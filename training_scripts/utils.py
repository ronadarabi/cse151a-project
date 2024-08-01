# The conversions and numbers in this file are magic for the love of god don't touch them
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def rgb_to_lab(rgb_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        RGB_TO_XYZ = torch.tensor(
            [
                [0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227],
            ],
            dtype=torch.float32,
            device=rgb_image.device,
        )
        XYZ_REF = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32, device=rgb_image.device)
        
        # Convert RGB to linear RGB
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0

        mask = rgb_image > 0.04045
        rgb_linear = torch.where(mask, torch.pow(((rgb_image + 0.055) / 1.055), 2.4), rgb_image / 12.92)
        
        # Convert linear RGB to XYZ
        xyz = torch.matmul(rgb_linear, RGB_TO_XYZ.t())
        
        # Normalize XYZ values
        xyz_scaled = xyz / XYZ_REF
        
        # XYZ to LAB conversion
        epsilon = 0.008856
        kappa = 903.3
        
        f = torch.where(xyz > epsilon, xyz_scaled.pow(1/3), (kappa * xyz_scaled + 16) / 116)

        x, y, z = f[..., 0], f[..., 1], f[..., 2]
        l = (116 * y - 16).unsqueeze(0)
        a = (500 * (x - y)).unsqueeze(0)
        b = (200 * (y - z)).unsqueeze(0)
        ab = torch.cat([a, b], dim=0)
        
        # Normalize to [-1, 1]
        l = (l / 50.0) - 1.0
        ab = ab / 110.0
        
        return l, ab

@torch.jit.script
def lab_to_rgb(l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        XYZ_TO_RGB = torch.tensor(
            [
                [ 3.24048134, -1.53715152, -0.49853633],
                [-0.96925495, 1.87599, 0.04155593],
                [ 0.05564664, -0.20404134, 1.05731107],
            ],
            dtype=torch.float32,
            device=l.device,
        )
        XYZ_REF = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32, device=l.device).view(1, 3, 1, 1)

        if l.dim() == 3:
            l = l.unsqueeze(0)
        if ab.dim() == 3:
            ab = ab.unsqueeze(0)
        
        # Denormalize from [-1, 1]
        l = (l + 1.0) * 50.0
        ab = ab * 110.0
        
        y = (l + 16) / 116
        x = ab[:, 0:1] / 500 + y
        z = y - ab[:, 1:2] / 200

        xyz = torch.cat([x, y, z], dim=1)
        
        mask = xyz > 0.2068966
        xyz = torch.where(mask, xyz.pow(3), (xyz - 16 / 116) / 7.787)
        
        xyz = xyz * XYZ_REF
        
        batch_size, _, height, width = xyz.shape
        xyz_reshaped = xyz.view(batch_size, 3, -1)

        rgb_linear = torch.bmm(XYZ_TO_RGB.expand(batch_size, -1, -1), xyz_reshaped)
        
        rgb_linear = rgb_linear.view(batch_size, 3, height, width)
        
        mask = rgb_linear > 0.0031308
        rgb = torch.where(mask, 1.055 * rgb_linear.pow(1 / 2.4) - 0.055, 12.92 * rgb_linear)
    
        return rgb.clamp(0, 1).permute(0, 2, 3, 1)