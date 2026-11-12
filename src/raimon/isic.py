"""Fed-ISIC2019 embeddings: a frozen ImageNet EfficientNet-B0 applied once to every image and corruption level.

python -I scripts/raimon_cli.py isic-embed --raw data/isic/raw --out data/isic/embeddings.npz --device mps

Images are those of FLamby's Fed-ISIC2019 split (6 centres). Preprocessing follows FLamby: the shorter side is
resized to 224 and Shades-of-Gray colour constancy (power 6) is applied; we then take a 224 centre crop and the
ImageNet normalisation. Each image is embedded under 12 views, indexed [level, noise]:

  level 0      clean
  level 1..5   Gaussian blur with sigma (0.5, 1.0, 1.5, 2.0, 6.0), used by V6 (ramp) and V1 (level 5)
  noise 1      additive Gaussian noise, std 0.01 in [0, 1] (about 2.5/255), used by R5

Strengths were chosen on a 1,500-image subset with a linear probe trained on the other images: the noise costs
about 2.5 accuracy points, and blur at sigma 0.5, 1.0, 1.5 and 2.0 costs about 1, 7, 11 and 22 points, so V6 (whose
shift reaches only deployment inputs on this dataset) degrades gradually. Level 5 (sigma 6) is V1's shift, which also
reaches training; the dermatology pilot showed weaker blur is absorbed by the head. ImageNet-C noise (0.08) cost 11 points.

The noise draw is fixed per image (seeded by its row index), so the same perturbation is used in every run.
Training never touches the backbone, which is why its outputs can be computed once.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CENTRES = ["BCN", "HAM_vidir_molemax", "HAM_vidir_modern", "HAM_rosendahl", "MSK", "HAM_vienna_dias"]
BLUR_SIGMAS = (0.5, 1.0, 1.5, 2.0, 6.0)
NOISE_STD = 0.01
MALIGNANT = ("MEL", "BCC", "SCC")
SIZE = 224


def color_constancy(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades of Gray, as in FLamby's preprocessing. img is HxWx3 uint8."""
    x = img.astype(np.float32)
    rgb = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)
    rgb = rgb / np.sqrt(np.sum(rgb ** 2))
    x = x * (1.0 / (rgb * np.sqrt(3)))
    return np.clip(x, 0, 255).astype(np.uint8)


def load_image(raw: bytes) -> np.ndarray:
    import io

    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    x = color_constancy(np.asarray(im))
    h, w = x.shape[:2]
    top, left = (h - SIZE) // 2, (w - SIZE) // 2
    return x[top:top + SIZE, left:left + SIZE]


class ZipImages(torch.utils.data.Dataset):
    def __init__(self, zip_path: Path, images: list[str]):
        self.zip_path, self.images, self.zf = zip_path, images, None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path)
            self.names = {Path(n).stem: n for n in self.zf.namelist() if n.endswith(".jpg")}
        x = load_image(self.zf.read(self.names[self.images[i]]))
        return torch.from_numpy(x).permute(2, 0, 1), i


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur as gb
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return gb(x, kernel_size=[k, k], sigma=[sigma, sigma])


def views(x: torch.Tensor, noise: torch.Tensor) -> list[tuple[int, int, torch.Tensor]]:
    """x: [B,3,H,W] in [0,1]. Returns (level, noise_flag, image) for all 12 views."""
    out = []
    for level in range(len(BLUR_SIGMAS) + 1):
        xb = x if level == 0 else gaussian_blur(x, BLUR_SIGMAS[level - 1])
        out.append((level, 0, xb))
        out.append((level, 1, (xb + NOISE_STD * noise).clamp(0, 1)))
    return out


def metadata(raw: Path) -> pd.DataFrame:
    split = pd.read_csv(raw / "flamby_train_test_split.csv")
    md = pd.read_csv(raw / "ISIC_2019_Training_Metadata.csv")
    df = split.merge(md, on="image", how="left", validate="one_to_one")
    df["y"] = df[list(MALIGNANT)].sum(axis=1).round().astype(int)
    df["sex_known"] = df["sex"].isin(["male", "female"])
    df["g"] = (df["sex"] == "female").astype(int)
    df["age_known"] = df["age_approx"].notna()
    df["age60"] = (df["age_approx"] >= 60).astype(int)
    return df


def embed(raw: Path, out: Path, device: str, batch: int = 64, workers: int = 8) -> None:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    df = metadata(raw)
    gate = {"images": len(df), "sex_known": float(df.sex_known.mean()), "age_known": float(df.age_known.mean())}
    print("metadata join", gate, flush=True)
    if gate["sex_known"] < 0.90:
        raise SystemExit("sex is known for fewer than 90% of images")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    net = efficientnet_b0(weights=weights)
    net.classifier = torch.nn.Identity()
    net = net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    n, L = len(df), len(BLUR_SIGMAS) + 1
    emb = np.zeros((L, 2, n, 1280), np.float16)
    ds = ZipImages(raw / "ISIC_2019_Training_Input.zip", df["image"].tolist())
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=False,
                                     persistent_workers=workers > 0)
    t0 = time.time()
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            x = x.to(device).float() / 255.0
            noise = torch.stack([torch.randn(3, SIZE, SIZE, generator=torch.Generator().manual_seed(int(i)))
                                 for i in idx]).to(device)
            for level, nf, v in views(x, noise):
                emb[level, nf, idx.numpy()] = net((v - mean) / std).float().cpu().numpy().astype(np.float16)
            if b % 20 == 0:
                done = min(n, (b + 1) * batch)
                print(f"{done}/{n} images, {time.time() - t0:.0f}s", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, emb=emb, image=df["image"].to_numpy(), center=df["center"].to_numpy(), y=df["y"].to_numpy(),
             g=df["g"].to_numpy(), sex_known=df["sex_known"].to_numpy(), age60=df["age60"].to_numpy(),
             age_known=df["age_known"].to_numpy())
    tmp.rename(out)
    receipt = {**gate, "weights": str(weights), "torch": torch.__version__, "device": device,
               "blur_sigmas": BLUR_SIGMAS, "noise_std": NOISE_STD, "malignant": MALIGNANT,
               "seconds": time.time() - t0}
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=1))
    print("wrote", out, receipt, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    embed(Path(a.raw), Path(a.out), a.device, a.batch, a.workers)


if __name__ == "__main__":
    main()
