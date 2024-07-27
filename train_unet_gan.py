import multiprocessing
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import wandb
import time
from dataclasses import dataclass, asdict
from torch.optim.lr_scheduler import LambdaLR

from model import MainModel
from unet import Unet
from utils import rgb_to_lab, lab_to_rgb


@dataclass
class ModelSettings:
    use_pretrained: bool = True
    pretrained_path: str = "unet/model_unet_final_percpt.pth"
    perception_loss_weight: float = 0.1
    total_image_count: int = 1012019
    validation_image_count: int = 12
    batch_size: int = 32
    num_steps: int = (total_image_count - validation_image_count) // batch_size
    learning_rate: float = 0.0002
    min_lr: float = learning_rate / 10
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    validation_steps: int = 1000
    display_imgs: int = 12
    early_stopping_patience: int = 5
    model_name: str = "Unet-GAN"

    def create_run_name(self):
        return f"{self.model_name}_lr{self.learning_rate}_bs{self.batch_size}_steps{self.num_steps}"


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


def lr_lambda(current_step: int):
    if current_step < settings.warmup_steps:
        return float(current_step) / float(max(1, settings.warmup_steps))
    cosine_decay = 0.5 * (
        1
        + np.cos(
            np.pi
            * (current_step - settings.warmup_steps)
            / (settings.num_steps - settings.warmup_steps)
        )
    )
    return (
        settings.min_lr + (settings.learning_rate - settings.min_lr) * cosine_decay
    ) / settings.learning_rate


def validate_model(model, test_dataloader):
    torch.cuda.synchronize()
    val_start_time = time.time()
    model.eval()
    val_loss_G = 0.0
    val_loss_D = 0.0
    total_images = 0
    logged_images = 0

    with torch.inference_mode():
        for l_chan, ab_chan in test_dataloader:
            l_chan, ab_chan = l_chan.to(device), ab_chan.to(device)

            model.setup_input(l_chan, ab_chan)
            model.forward()

            # Compute losses without backward passes
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
    optimizer_G,
    optimizer_D,
    scheduler_G,
    scheduler_D,
    num_steps,
    validation_steps,
):
    torch.cuda.synchronize()
    start_time = time.time()
    model.train()
    step_times = []
    best_val_loss = float("inf")
    patience_counter = 0

    scaler = GradScaler()
    train_iter = iter(train_dataloader)

    for step in range(num_steps):
        try:
            l_chan, ab_chan = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            l_chan, ab_chan = next(train_iter)

        l_chan, ab_chan = l_chan.to(device), ab_chan.to(device)

        with autocast():
            model.setup_input(l_chan, ab_chan)
            model.forward()

        # Update D
        optimizer_D.zero_grad(set_to_none=True)
        with autocast():
            loss_D = model.compute_D_loss()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)

        # Update G
        optimizer_G.zero_grad(set_to_none=True)
        with autocast():
            loss_G = model.compute_G_loss()
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)

        scaler.update()

        # Step the schedulers after the optimizers
        scheduler_D.step()
        scheduler_G.step()

        torch.cuda.synchronize()
        step_end_time = time.time()
        step_times.append(step_end_time)

        if step > 0:
            step_time = step_times[-1] - step_times[-2]
        else:
            step_time = step_end_time - start_time

        total_time_spent = step_end_time - start_time
        avg_step_time = total_time_spent / (step + 1)
        etc = avg_step_time * (num_steps - (step + 1))
        time_to_next_checkpoint = avg_step_time * (
            validation_steps - (step % validation_steps)
        )

        print(
            f"Step [{step + 1}/{num_steps}], "
            f"Loss G: {loss_G.item():.4f}, "
            f"Loss D: {loss_D.item():.4f}, "
            f"Step Time: {step_time:.4f}s, "
            f"ETC: {etc/3600:.2f} hours, "
            f"Next Checkpoint: {time_to_next_checkpoint/60:.2f} minutes"
        )
        wandb.log(
            {
                "Training Loss G": loss_G.item(),
                "Training Loss D": loss_D.item(),
                "Step": step + 1,
                "Step Time": step_time,
                "Learning Rate G": scheduler_G.get_last_lr()[0],
                "Learning Rate D": scheduler_D.get_last_lr()[0],
                "ETC (hours)": etc / 3600,
                "Time to Next Checkpoint (minutes)": time_to_next_checkpoint / 60,
            }
        )

        if (step + 1) % validation_steps == 0:
            model.eval()
            val_loss = validate_model(model, test_dataloader)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_G_state_dict": optimizer_G.state_dict(),
                        "optimizer_D_state_dict": optimizer_D.state_dict(),
                        "scheduler_G_state_dict": scheduler_G.state_dict(),
                        "scheduler_D_state_dict": scheduler_D.state_dict(),
                        "loss_G": loss_G.item(),
                        "loss_D": loss_D.item(),
                    },
                    f"best_checkpoint_unet_gan.pth",
                )
            else:
                patience_counter += 1
                if patience_counter >= settings.early_stopping_patience:
                    print(
                        f"Early stopping triggered after {patience_counter} validations without improvement."
                    )
                    break

            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "scheduler_G_state_dict": scheduler_G.state_dict(),
                    "scheduler_D_state_dict": scheduler_D.state_dict(),
                    "loss_G": loss_G.item(),
                    "loss_D": loss_D.item(),
                },
                f"checkpoint_unet_gan.pth",
            )

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    print(f"Total Time: {total_time:.4f}s")
    wandb.log({"Total Time": total_time})

    validate_model(model, test_dataloader)

    torch.save(model.state_dict(), "model_unet_gan_final.pth")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="unet-gan-colorizer")

    settings = ModelSettings()
    wandb.config.update(asdict(settings))
    wandb.run.name = settings.create_run_name()

    root_dir = "new_data"

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
        pretrained_unet.load_state_dict(torch.load(settings.pretrained_path))
        print(f"Loaded pretrained model from {settings.pretrained_path}")

    model = MainModel(
        net_G=pretrained_unet, lambda_percep=settings.perception_loss_weight
    ).to(device)
    wandb.watch(model)

    print(model)

    optimizer_G = model.opt_G
    optimizer_D = model.opt_D

    scheduler_G = LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda)

    train_model(
        model,
        train_dataloader,
        test_dataloader,
        optimizer_G,
        optimizer_D,
        scheduler_G,
        scheduler_D,
        settings.num_steps,
        settings.validation_steps,
    )
