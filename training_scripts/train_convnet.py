import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import wandb
import time
from dataclasses import dataclass, asdict
from torch.optim.lr_scheduler import LambdaLR

wandb.init(project="convnet-colorizer")

@dataclass
class ModelSettings:
    total_image_count: int = 1012019
    validation_image_count: int = 1024
    batch_size: int = 10
    num_steps: int = (total_image_count - validation_image_count) // batch_size
    learning_rate: float = 0.0007 # Peak LR
    min_lr: float = learning_rate / 10  # Minimum LR
    weight_decay: float = 1e-5
    warmup_steps: int = 1000 # Linear warmup over n steps
    validation_steps: int = 1000 # Validate every n steps
    loss_function: str = "HuberLoss"
    optimizer: str = "Adam"
    model_name: str = "ConvNet"

    def create_run_name(self):
        return f"{self.model_name}_lr{self.learning_rate}_bs{self.batch_size}_steps{self.num_steps}_loss{self.loss_function}_opt{self.optimizer}"

settings = ModelSettings()

class ImageFolderRGBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg')):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        gray_image = transforms.functional.to_grayscale(image, num_output_channels=1)
        gray_image = torch.tensor(np.array(gray_image)).unsqueeze(0).float() / 255.0
        rgb_image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        return gray_image, rgb_image


root_dir = 'new_data'

transform = transforms.Compose([
    transforms.Resize((512, 512)),
])

dataset = ImageFolderRGBDataset(root_dir=root_dir, transform=transform)
train_size = settings.total_image_count - settings.validation_image_count
test_size = settings.validation_image_count
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=False, pin_memory=True)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),

            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = ConvNet().to("cuda")
wandb.watch(model)

criterion = getattr(nn, settings.loss_function)()
optimizer = getattr(optim, settings.optimizer)(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)


# Linear warmup followed by cosine decay
def lr_lambda(current_step: int):
    if current_step < settings.warmup_steps:
        return float(current_step) / float(max(1, settings.warmup_steps))
    cosine_decay = 0.5 * (1 + np.cos(np.pi * (current_step - settings.warmup_steps) / (settings.num_steps - settings.warmup_steps)))
    return (settings.min_lr + (settings.learning_rate - settings.min_lr) * cosine_decay) / settings.learning_rate

scheduler = LambdaLR(optimizer, lr_lambda)


wandb.config.update(asdict(settings))
wandb.run.name = settings.create_run_name()

def validate_model(model, test_dataloader, criterion):
    val_start_time = time.time()
    model.eval()
    val_loss = 0.0
    total_images = 0
    logged_images = 0

    with torch.no_grad():
        for i, (L, RGB) in enumerate(test_dataloader):
            L, RGB = L.to("cuda"), RGB.to("cuda")

            outputs = model(L)
            loss = criterion(outputs, RGB)
            val_loss += loss.item()
            total_images += L.size(0)

            if logged_images < 8:
                num_samples = min(8 - logged_images, L.shape[0])
                L_samples = L[:num_samples].cpu().numpy()
                output_samples = outputs[:num_samples].cpu().numpy()
                target_samples = RGB[:num_samples].cpu().numpy()

                L_rgb_samples = [
                    np.repeat(L_samples[j], 3, axis=0).transpose(1, 2, 0)
                    for j in range(num_samples)
                ]
                output_rgb_samples = [
                    output_samples[j].transpose(1, 2, 0)
                    for j in range(num_samples)
                ]
                target_rgb_samples = [
                    target_samples[j].transpose(1, 2, 0)
                    for j in range(num_samples)
                ]

                stacked_L_rgb = np.hstack(L_rgb_samples)
                stacked_output_rgb = np.hstack(output_rgb_samples)
                stacked_target_rgb = np.hstack(target_rgb_samples)

                stacked_images = np.vstack(
                    (stacked_L_rgb, stacked_output_rgb, stacked_target_rgb)
                )

                wandb.log(
                    {
                        "Examples": wandb.Image(
                            stacked_images,
                            caption="Top: Grayscale, Middle: Predicted, Bottom: True",
                        )
                    }, commit=False
                )
                
                logged_images += num_samples

    avg_val_loss = val_loss / (total_images / settings.batch_size)
    val_time = time.time() - val_start_time
    print(f"Average Validation Loss: {avg_val_loss:.4f}, Validation Time: {val_time:.4f}s")

    wandb.log({"Average Validation Loss": avg_val_loss, "Validation Time": val_time}, commit=False)

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_steps, validation_steps):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    step_times = []

    train_iter = iter(train_dataloader)

    for step in range(num_steps):
        try:
            L, RGB = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            L, RGB = next(train_iter)

        L, RGB = L.to("cuda"), RGB.to("cuda")

        outputs = model(L)
        loss = criterion(outputs, RGB)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
        etc = avg_step_time * (num_steps - (step + 1))

        print(
            f"Step [{step + 1}/{num_steps}], Loss: {loss.item():.4f}, Step Time: {step_time:.4f}s, ETC: {etc/3600:.2f} hours"
        )
        wandb.log({
            "Training Loss": loss.item(), 
            "Step": step + 1, 
            "Step Time": step_time, 
            "Learning Rate": scheduler.get_last_lr()[0],
            "ETC (hours)": etc / 3600
        })

        if (step + 1) % validation_steps == 0:
            validate_model(model, test_dataloader, criterion)

            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, f'checkpoint.pth')

    avg_train_loss = running_loss / num_steps
    total_time = time.time() - start_time

    print(
        f"Average Training Loss: {avg_train_loss:.4f}, Total Time: {total_time:.4f}s"
    )
    wandb.log({"Average Training Loss": avg_train_loss, "Total Time": total_time})

    validate_model(model, test_dataloader, criterion)

    torch.save(model.state_dict(), "model_final.pth")

train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, settings.num_steps, settings.validation_steps)
