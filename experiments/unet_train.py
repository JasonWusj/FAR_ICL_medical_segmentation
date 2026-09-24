"""2D U-Net supervised baseline using the same ISIC split and 128px scoring grid."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from far_icl.config import load_config, seed_all, write_json
from far_icl.data import load_case, read_manifest

from .score_masks import score


class ISICDataset(Dataset):
    def __init__(self, cases, cfg, augment=False):
        self.cases, self.cfg, self.augment = cases, cfg, augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        data = load_case(self.cases[index], self.cfg)
        x, y = data["rgb"], data["mask"]
        if self.augment:
            if torch.rand(()) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(()) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y


def block(inp, out):
    return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.InstanceNorm2d(out), nn.ReLU(inplace=True),
                         nn.Conv2d(out, out, 3, padding=1), nn.InstanceNorm2d(out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.down1, self.down2, self.down3, self.down4 = (
            block(3, base), block(base, base * 2), block(base * 2, base * 4),
            block(base * 4, base * 8))
        self.center = block(base * 8, base * 16)
        self.up4 = block(base * 24, base * 8)
        self.up3 = block(base * 12, base * 4)
        self.up2 = block(base * 6, base * 2)
        self.up1 = block(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2))
        c = self.down3(F.max_pool2d(b, 2))
        d = self.down4(F.max_pool2d(c, 2))
        z = self.center(F.max_pool2d(d, 2))
        for skip, layer in ((d, self.up4), (c, self.up3), (b, self.up2), (a, self.up1)):
            z = layer(torch.cat([F.interpolate(z, size=skip.shape[-2:], mode="bilinear",
                                                align_corners=False), skip], dim=1))
        return self.head(z)


def loss_fn(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    prob = logits.sigmoid()
    overlap = (prob * mask).sum((1, 2, 3))
    dice_loss = 1 - ((2 * overlap + 1) / (prob.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean()
    return bce + dice_loss


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    values = []
    for images, masks in loader:
        prediction = model(images.to(device)).sigmoid() > 0.5
        truth = masks.to(device) > 0.5
        intersection = (prediction & truth).sum((1, 2, 3)).float()
        values.extend(((2 * intersection + 1e-8) /
                       (prediction.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1e-8)).cpu().tolist())
    return float(np.mean(values))


def train_eval(cfg, epochs, patience, batch_size):
    seed_all(cfg["seed"], cfg["deterministic"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cases = read_manifest(cfg["manifest"], cfg["identity_scope"])
    train = [c for c in cases if c.split == "train"]
    val = [c for c in cases if c.split == "val"]
    test = [c for c in cases if c.split == "test"]
    if not train or not val or not test:
        raise ValueError("Need nonempty train, val and test sets")
    workers = min(4, int(__import__("os").environ.get("DATA_WORKERS", "4")))
    train_loader = DataLoader(ISICDataset(train, cfg, True), batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ISICDataset(val, cfg), batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda")
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best_unet.pt"
    best, stale = -math.inf, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_dice = validate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice}
        print(record, flush=True)
        history.append(record)
        if val_dice > best + 1e-5:
            best, stale = val_dice, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best}, best_path)
        else:
            stale += 1
        if stale >= patience:
            break
    write_json(history, out / "train_history.json")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    val_dir = out / "predictions_val"
    test_dir = out / "predictions_test"
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    for case in val + test:
        data = load_case(case, cfg, with_mask=False)
        with torch.no_grad():
            mask = (model(data["rgb"][None].to(device)).sigmoid()[0, 0] > 0.5)
        target = val_dir if case.split == "val" else test_dir
        Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255).save(target / f"{case.case_id}.png")
    score(cfg, val_dir, "unet", "val")
    score(cfg, test_dir, "unet", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isic.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience and batch-size must be positive")
    train_eval(load_config(args.config, args.set), args.epochs, args.patience, args.batch_size)


if __name__ == "__main__":
    main()
