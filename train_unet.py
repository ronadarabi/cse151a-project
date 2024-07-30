from gc import disable
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
import multiprocessing
import gc

from unet import Unet
from utils import rgb_to_lab, lab_to_rgb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class ModelSettings:
    finetune: bool = True
    pretrained_model_path: str = "unet_attn_finetune/model_unet_final.pth"
    finetune_learning_rate: float = 0.00035 / 4
    total_image_count: int = 1012019
    validation_image_count: int = 1024
    batch_size: int = 64
    num_steps: int = (total_image_count - validation_image_count) // batch_size
    learning_rate: float = 0.0007  # Peak LR
    min_lr: float = learning_rate / 10  # Minimum LR
    weight_decay: float = 1e-5
    warmup_steps: int = 1000  # Linear warmup over n steps
    validation_steps: int = 1000  # Validate every n steps
    display_imgs: int = 12
    early_stopping_patience: int = 5
    loss_function: str = "L1Loss"
    optimizer: str = "AdamW"
    model_name: str = "Unet"

    def create_run_name(self):
        prefix = "finetune_" if self.finetune else ""
        learning_rate = self.finetune_learning_rate if self.finetune else self.learning_rate
        return f"{prefix}{self.model_name}_lr{learning_rate}_bs{self.batch_size}_steps{self.num_steps}_loss{self.loss_function}_opt{self.optimizer}"


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


# Linear warmup followed by cosine decay
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


def validate_model(model, test_dataloader, criterion):
    torch.cuda.empty_cache()
    gc.collect()
    val_start_time = time.time()
    model.eval()
    val_loss = 0.0
    total_images = 0
    logged_images = 0

    with torch.no_grad():
        for l_chan, ab_chan in test_dataloader:
            with autocast():
                outputs = model(l_chan)
                loss = criterion(outputs, ab_chan)

            val_loss += loss.item()
            total_images += l_chan.size(0)

            if logged_images < settings.display_imgs:
                num_samples = min(settings.display_imgs - logged_images, l_chan.shape[0])
                l_chan_samples = l_chan[:num_samples]
                output_samples = outputs[:num_samples]
                target_samples = ab_chan[:num_samples]

                l_rgb_samples = lab_to_rgb(l_chan_samples, torch.zeros_like(output_samples))
                output_rgb_samples = lab_to_rgb(l_chan_samples, output_samples)
                target_rgb_samples = lab_to_rgb(l_chan_samples, target_samples)

                l_rgb_samples = [sample.detach().cpu().numpy() for sample in l_rgb_samples]
                output_rgb_samples = [sample.detach().cpu().numpy() for sample in output_rgb_samples]
                target_rgb_samples = [sample.detach().cpu().numpy() for sample in target_rgb_samples]

                stacked_L_rgb = np.hstack(l_rgb_samples)
                stacked_output_rgb = np.hstack(output_rgb_samples)
                stacked_target_rgb = np.hstack(target_rgb_samples)

                stacked_images = np.vstack(
                    (stacked_L_rgb, stacked_output_rgb, stacked_target_rgb)
                )

                wandb.log(
                    {
                        "Examples": wandb.Image(
                            stacked_images,
                            caption="For each column: Top: Grayscale, Middle: Predicted, Bottom: True",
                        )
                    },
                    commit=False,
                )

                logged_images += l_chan.size(0)

    avg_val_loss = val_loss / (total_images / settings.batch_size)
    val_time = time.time() - val_start_time
    print(
        f"Average Validation Loss: {avg_val_loss:.4f}, Validation Time: {val_time:.4f}s"
    )
    wandb.log(
        {"Average Validation Loss": avg_val_loss, "Validation Time": val_time},
        commit=False,
    )
    return avg_val_loss


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_steps,
    validation_steps,
):
    torch.cuda.synchronize()
    start_time = time.time()
    model.train()
    running_loss = 0.0
    step_times = []
    best_val_loss = float("inf")
    patience_counter = 0

    train_iter = iter(train_dataloader)

    for step in range(num_steps):
        try:
            l_chan, ab_chan = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            l_chan, ab_chan = next(train_iter)

        l_chan, ab_chan = l_chan.to("cuda"), ab_chan.to("cuda")

        with autocast():
            outputs = model(l_chan)
            loss = criterion(outputs, ab_chan)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        step_end_time = time.time()
        step_times.append(step_end_time)

        running_loss += loss.item()

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

        print(
            f"Step [{step + 1}/{num_steps}], "
            f"Loss: {loss.item():.4f}, "
            f"Step Time: {step_time:.4f}s, "
            f"ETC: {etc_hours}h {etc_minutes}m, "
            f"Next Checkpoint: {time_to_next_checkpoint/60:.2f} minutes"
        )
        wandb.log(
            {
                "Training Loss": loss.item(),
                "Step": step + 1,
                "Step Time": step_time,
                "Learning Rate": scheduler.get_last_lr()[0],
                "ETC (hours)": etc_seconds / 3600,
                "Time to Next Checkpoint (minutes)": time_to_next_checkpoint / 60,
            }
        )

        if (step + 1) % validation_steps == 0:
            val_loss = validate_model(model, test_dataloader, criterion)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss,
                    },
                    f"best_checkpoint_unet.pth",
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                },
                f"checkpoint_unet.pth",
            )

    torch.cuda.synchronize()
    avg_train_loss = running_loss / num_steps
    total_time = time.time() - start_time

    print(f"Average Training Loss: {avg_train_loss:.4f}, Total Time: {total_time:.4f}s")
    wandb.log({"Average Training Loss": avg_train_loss, "Total Time": total_time})

    validate_model(model, test_dataloader, criterion)

    torch.save(model.state_dict(), "model_unet_final.pth")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    wandb.init(project="unet-colorizer")

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

    model = Unet().to("cuda")

    if settings.finetune:
        print(f"Loading pre-trained model from {settings.pretrained_model_path}")
        model.load_state_dict(torch.load(settings.pretrained_model_path))

    print(f"The model has {count_parameters(model):,} trainable parameters")
    wandb.watch(model)

    print(model)

    scaler = GradScaler()
    criterion = getattr(nn, settings.loss_function)()

    if settings.finetune:
        optimizer = getattr(optim, settings.optimizer)(
            model.parameters(),
            lr=settings.finetune_learning_rate,
            weight_decay=settings.weight_decay,
        )
    else:
        optimizer = getattr(optim, settings.optimizer)(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    train_model(
        model,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        scheduler,
        settings.num_steps,
        settings.validation_steps,
    )
