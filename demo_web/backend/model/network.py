from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_CHECKPOINT = Path(__file__).resolve().with_name("preference_controlled_cnn.pt")


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        h = self.relu(self.conv1(x))
        h = self.conv2(h)
        return self.relu(h + residual)


class GlobalContextBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.pool(x).view(b, c)
        g = self.fc1(g)
        g = self.relu(g)
        g = self.fc2(g)
        g = g.view(b, c, 1, 1)
        return x + g.expand(-1, -1, h, w)


class PreferenceControlledCNN(nn.Module):
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT):
        super().__init__()
        self.input_conv = nn.Conv2d(7, 64, 3, padding=1)
        self.block1 = ResidualBlock()
        self.block2 = ResidualBlock()
        self.block3 = ResidualBlock()
        self.global_context1 = GlobalContextBranch()
        self.global_context2 = GlobalContextBranch()
        self.output_conv = nn.Conv2d(64, 3, 3, padding=1)
        nn.init.normal_(self.output_conv.weight, std=1e-3)
        nn.init.zeros_(self.output_conv.bias)

        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("cnn"), dict):
                raise ValueError("checkpoint must be a mapping containing a 'cnn' state_dict")
            self.load_state_dict(checkpoint["cnn"], strict=True)
        except Exception as error:
            raise RuntimeError(f"CNN checkpoint is incompatible: {checkpoint_path}") from error
        self.checkpoint_path = checkpoint_path

    def forward(self, image_tensor, diffmap_tensor):
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise ValueError(f"image_tensor must be [B,3,H,W], got {tuple(image_tensor.shape)}")
        if diffmap_tensor.ndim != 4 or diffmap_tensor.shape[1] != 4:
            raise ValueError(f"diffmap_tensor must be [B,4,H,W], got {tuple(diffmap_tensor.shape)}")
        if image_tensor.shape[0] != diffmap_tensor.shape[0] or image_tensor.shape[2:] != diffmap_tensor.shape[2:]:
            raise ValueError("image and diffmap batch/spatial dimensions must match")
        x = torch.cat([image_tensor, diffmap_tensor], dim=1)
        features = self.input_conv(x)
        features = self.block1(features)
        features = self.global_context1(features)
        features = self.block2(features)
        features = self.block3(features)
        features = self.global_context2(features)
        return self.output_conv(features)


@torch.no_grad()
def run_inference(model, image_tensor, diffmap_tensor):
    model.eval()
    delta = model(image_tensor, diffmap_tensor)
    return torch.clamp(image_tensor + delta, 0.0, 1.0)
