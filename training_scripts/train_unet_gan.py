import multiprocessing
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import wandb
import time
from dataclasses import dataclass, asdict

from model import MainModel
from unet import Unet
from utils import rgb_to_lab, lab_to_rgb


@dataclass
class ModelSettings:
    use_pretrained: bool = False
    pretrained_path: str = ""
    l1_loss_weight: float = 100.0
    gan_loss_weight: float = 1.0
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    total_image_count: int = 991208
    validation_image_count: int = 24
    batch_size: int = 32
    num_steps: int = (total_image_count - validation_image_count) // batch_size
    warmup_steps: int = 1000
    validation_steps: int = 1000
    display_imgs: int = 12
    model_name: str = "Unet-GAN"
    gan_mode: str = "vanilla"

    def create_run_name(self):
        if self.use_pretrained:
            return f"{self.model_name}_pretrained_bs{self.batch_size}_steps{self.num_steps}"
        return f"{self.model_name}_bs{self.batch_size}_steps{self.num_steps}"


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith((".jpg")):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(np.array(image).astype(np.float32))
        l_chan, ab_chan = rgb_to_lab(image)
        return l_chan, ab_chan


class FastColorizationDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        iterator = super().__iter__()
        for data in iterator:
            with torch.cuda.stream(self.stream):
                yield [item.cuda(non_blocking=True) for item in data]


def validate_model(model, test_dataloader):
    torch.cuda.synchronize()
    val_start_time = time.time()
    model.eval()
    val_loss_G = 0.0
    val_loss_D = 0.0
    total_images = 0
    logged_images = 0

    with torch.no_grad():
        for l_chan, ab_chan in test_dataloader:
            l_chan, ab_chan = l_chan.to(device), ab_chan.to(device)

            model.setup_input(l_chan, ab_chan)
            model.forward()

            loss_D = model.compute_D_loss()
            loss_G = model.compute_G_loss()

            val_loss_G += loss_G.item()
            val_loss_D += loss_D.item()
            total_images += l_chan.size(0)

            if logged_images < settings.display_imgs:
                num_samples = min(
                    settings.display_imgs - logged_images, l_chan.shape[0]
                )
                l_chan_samples = l_chan[:num_samples]
                output_samples = model.fake_color[:num_samples]
                target_samples = ab_chan[:num_samples]

                greyscale_samples = lab_to_rgb(
                    l_chan_samples, torch.zeros_like(output_samples)
                )
                output_rgb_samples = lab_to_rgb(l_chan_samples, output_samples)
                target_rgb_samples = lab_to_rgb(l_chan_samples, target_samples)

                wandb.log(
                    {
                        "Examples": wandb.Image(
                            np.vstack(
                                [
                                    np.hstack(greyscale_samples.cpu().numpy()),
                                    np.hstack(output_rgb_samples.cpu().numpy()),
                                    np.hstack(target_rgb_samples.cpu().numpy()),
                                ]
                            ),
                            caption="Top: Grayscale, Middle: Predicted, Bottom: True",
                        )
                    },
                    commit=False,
                )

                logged_images += num_samples

    torch.cuda.synchronize()
    avg_val_loss_G = val_loss_G / (total_images / settings.batch_size)
    avg_val_loss_D = val_loss_D / (total_images / settings.batch_size)
    val_time = time.time() - val_start_time
    print(
        f"Average Validation Loss G: {avg_val_loss_G:.4f}, D: {avg_val_loss_D:.4f}, Validation Time: {val_time:.4f}s"
    )
    wandb.log(
        {
            "Validation Loss G": avg_val_loss_G,
            "Validation Loss D": avg_val_loss_D,
            "Validation Time": val_time,
        },
        commit=False,
    )
    return avg_val_loss_G + avg_val_loss_D


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    num_steps,
    validation_steps,
):
    torch.cuda.synchronize()
    start_time = time.time()
    model.train()
    step_times = []
    best_val_loss = float("inf")

    train_iter = iter(train_dataloader)

    for step in range(num_steps):
        try:
            l_chan, ab_chan = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            l_chan, ab_chan = next(train_iter)

        l_chan, ab_chan = l_chan.to(device), ab_chan.to(device)

        # Update Discriminator
        model.set_requires_grad(model.net_D, True)
        model.opt_D.zero_grad(set_to_none=True)
        model.setup_input(l_chan, ab_chan)
        model.forward()
        loss_D = model.compute_D_loss()
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(model.net_D.parameters(), max_norm=1.0)
        model.opt_D.step()

        # Update Generator
        model.set_requires_grad(model.net_D, False)
        model.opt_G.zero_grad(set_to_none=True)
        loss_G = model.compute_G_loss()
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(model.net_G.parameters(), max_norm=1.0)
        model.opt_G.step()

        torch.cuda.synchronize()
        step_end_time = time.time()
        step_times.append(step_end_time)

        if step > 0:
            step_time = step_times[-1] - step_times[-2]
        else:
            step_time = step_end_time - start_time

        total_time_spent = step_end_time - start_time
        avg_step_time = total_time_spent / (step + 1)
        etc_seconds = avg_step_time * (num_steps - (step + 1))
        etc_hours = int(etc_seconds // 3600)
        etc_minutes = int((etc_seconds % 3600) // 60)
        time_to_next_checkpoint = avg_step_time * (
            validation_steps - (step % validation_steps)
        )

        if model.loss_G_GAN is not None:
            loss_G_GAN = model.loss_G_GAN.item()
        else:
            loss_G_GAN = 0.0

        if model.loss_G_L1 is not None:
            loss_G_L1 = model.loss_G_L1.item()
        else:
            loss_G_L1 = 0.0

        print(
            f"Step [{step + 1}/{num_steps}], "
            f"Loss G: {loss_G.item():.4f}, "
            f"Loss D: {loss_D.item():.4f}, "
            f"Step Time: {step_time:.4f}s, "
            f"ETC: {etc_hours}h {etc_minutes}m, "
            f"Next Checkpoint: {time_to_next_checkpoint/60:.2f} minutes"
        )

        log_dict = {
            "Generator Loss": loss_G,
            "Generator Loss GAN": loss_G_GAN,
            "Generator Loss L1": loss_G_L1,
            "Discriminator Loss": loss_D,
            "Discriminator Loss Real": model.loss_D_real.item(),
            "Discriminator Loss Fake": model.loss_D_fake.item(),
            "Step": step + 1,
            "Step Time": step_time,
            "ETC (hours)": etc_seconds / 3600,
            "Time to Next Checkpoint (minutes)": time_to_next_checkpoint / 60,
        }

        wandb.log(log_dict)

        if (step + 1) % validation_steps == 0:
            model.eval()
            val_loss = validate_model(model, test_dataloader)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_G_state_dict": model.opt_G.state_dict(),
                        "optimizer_D_state_dict": model.opt_D.state_dict(),
                        "loss_G": loss_G.item(),
                        "loss_D": loss_D.item(),
                    },
                    f"best_checkpoint_unet_gan.pth",
                )
                
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_G_state_dict": model.opt_G.state_dict(),
                    "optimizer_D_state_dict": model.opt_D.state_dict(),
                    "loss_G": loss_G.item(),
                    "loss_D": loss_D.item(),
                },
                f"checkpoint_unet_gan_{step + 1}.pth",
            )

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    print(f"Total Time: {total_time:.4f}s")
    wandb.log({"Total Time": total_time})

    validate_model(model, test_dataloader)

    torch.save(model.state_dict(), "model_final_unet_gan.pth")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="unet-gan-colorizer")

    settings = ModelSettings()
    wandb.config.update(asdict(settings))
    wandb.run.name = settings.create_run_name()

    root_dir = "img_data"

    dataset = ColorizationDataset(root_dir=root_dir)
    train_size = settings.total_image_count - settings.validation_image_count
    test_size = settings.validation_image_count
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = FastColorizationDataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_dataloader = FastColorizationDataLoader(
        test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    pretrained_unet = Unet(input_c=1, output_c=2, n_down=8, num_filters=64)
    if settings.use_pretrained:
        state_dict = torch.load(settings.pretrained_path)
        if 'net_G' in state_dict:
            # The model was saved as part of MainModel
            pretrained_unet.load_state_dict({k.replace('net_G.', ''): v for k, v in state_dict['net_G'].items()})
        elif 'model_state_dict' in state_dict:
            # The model was saved with the entire state dict
            pretrained_unet.load_state_dict({k.replace('net_G.', ''): v for k, v in state_dict['model_state_dict'].items() if k.startswith('net_G.')})
        else:
            # Try loading directly
            pretrained_unet.load_state_dict({k.replace('net_G.', ''): v for k, v in state_dict.items() if k.startswith('net_G.')})
        print(f"Loaded pretrained model from {settings.pretrained_path}")

    model = MainModel(
        net_G=pretrained_unet,
        lr_G=settings.generator_lr,
        lr_D=settings.discriminator_lr,
        beta1=settings.beta1,
        beta2=settings.beta2,
        lambda_L1=settings.l1_loss_weight,
        lambda_GAN=settings.gan_loss_weight,
        gan_mode=settings.gan_mode,
    ).to(device)
    wandb.watch(model)

    print(model)

    print(f"Generator Parameters: {sum(p.numel() for p in model.net_G.parameters())}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.net_D.parameters())}")

    train_model(
        model,
        train_dataloader,
        test_dataloader,
        settings.num_steps,
        settings.validation_steps,
    )
