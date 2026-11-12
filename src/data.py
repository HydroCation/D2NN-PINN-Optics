"""
Data pipeline and procedural benchmark generator for diffractive optical evaluation.

Supports:
1. Automated torchvision MNIST downloading and preprocessing into 8x8 complex fields.
2. Procedural generator fallback for offline environments creating 1,000 synthetic
   geometric digit-like masks (ellipses, vertical/horizontal strokes for digits 0-9).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("d2nn.data")


def generate_synthetic_digit(
    digit: int,
    fov_size: int = 8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Procedurally generate a geometric digit-like mask (0-9) on a square grid.

    Constructs clean geometric patterns mimicking handwritten digits:
    - 0: Elliptical ring
    - 1: Vertical center stroke
    - 2: Top bar, upper right stroke, middle bar, lower left stroke, bottom bar
    - 3: Top/mid/bottom bars with right vertical backbone
    - 4: Left upright, crossbar, and right vertical stem
    - 5: Top bar, upper left stroke, middle bar, lower right loop, bottom bar
    - 6: Upper left arc with enclosed bottom loop
    - 7: Top horizontal bar and diagonal stroke
    - 8: Double stacked elliptical loops (figure-8)
    - 9: Upper enclosed loop with right vertical stroke

    Args:
        digit: Integer class in [0, 9].
        fov_size: Grid size (default 8).
        seed: Optional RNG seed.

    Returns:
        np.ndarray of shape (fov_size, fov_size) with values in [0.0, 1.0].
    """
    rng = np.random.RandomState(seed)
    base_size = 8
    img = np.zeros((base_size, base_size), dtype=np.float32)
    y, x = np.mgrid[0:base_size, 0:base_size].astype(np.float32)
    cx, cy = (base_size - 1.0) / 2.0, (base_size - 1.0) / 2.0

    # Slight random geometric jitter
    dx = rng.uniform(-0.3, 0.3)
    dy = rng.uniform(-0.3, 0.3)
    cx_j, cy_j = cx + dx, cy + dy

    if digit == 0:
        r = np.sqrt(((x - cx_j) / 2.4) ** 2 + ((y - cy_j) / 3.0) ** 2)
        img[(r >= 0.55) & (r <= 1.25)] = 1.0
    elif digit == 1:
        c = int(round(cx_j))
        c = max(1, min(base_size - 2, c))
        img[1 : base_size - 1, c] = 1.0
        img[1, max(0, c - 1)] = 0.6  # Top serif
    elif digit == 2:
        img[1, 1 : base_size - 1] = 1.0
        img[2:4, base_size - 2] = 1.0
        img[4, 2 : base_size - 2] = 1.0
        img[5, 1:3] = 1.0
        img[base_size - 2, 1 : base_size - 1] = 1.0
    elif digit == 3:
        img[1, 1 : base_size - 1] = 1.0
        cy_int = int(round(cy_j))
        img[cy_int, 2 : base_size - 1] = 1.0
        img[base_size - 2, 1 : base_size - 1] = 1.0
        img[1 : base_size - 1, base_size - 2] = 1.0
    elif digit == 4:
        img[1:4, 1] = 1.0
        img[4, 1 : base_size - 1] = 1.0
        img[1 : base_size - 1, base_size - 3] = 1.0
    elif digit == 5:
        img[1, 1 : base_size - 1] = 1.0
        img[2:4, 1] = 1.0
        img[4, 1 : base_size - 2] = 1.0
        img[4 : base_size - 1, base_size - 2] = 1.0
        img[base_size - 2, 1 : base_size - 2] = 1.0
    elif digit == 6:
        r = np.sqrt(((x - cx_j) / 2.2) ** 2 + ((y - (cy_j + 1.2)) / 1.7) ** 2)
        img[(r >= 0.5) & (r <= 1.2)] = 1.0
        img[1 : base_size - 3, 1] = 1.0
    elif digit == 7:
        img[1, 1 : base_size - 1] = 1.0
        for i in range(1, base_size - 1):
            col = max(1, min(base_size - 2, int(round((base_size - 1) - i * 0.75))))
            img[i, col] = 1.0
    elif digit == 8:
        r1 = np.sqrt(((x - cx_j) / 2.0) ** 2 + ((y - (cy_j - 1.4)) / 1.5) ** 2)
        r2 = np.sqrt(((x - cx_j) / 2.2) ** 2 + ((y - (cy_j + 1.4)) / 1.6) ** 2)
        img[(r1 >= 0.45) & (r1 <= 1.15)] = 1.0
        img[(r2 >= 0.50) & (r2 <= 1.25)] = 1.0
    elif digit == 9:
        r = np.sqrt(((x - cx_j) / 2.2) ** 2 + ((y - (cy_j - 1.2)) / 1.7) ** 2)
        img[(r >= 0.5) & (r <= 1.2)] = 1.0
        img[3 : base_size - 1, base_size - 2] = 1.0

    # Add subtle organic noise and ensure strict dynamic range [0.0, 1.0]
    img += rng.uniform(0.0, 0.05, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    # Resize if target fov_size != base_size
    if fov_size != base_size:
        t_in = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        t_out = F.interpolate(t_in, size=(fov_size, fov_size), mode="area")
        img = t_out.squeeze(0).squeeze(0).numpy()
        img = np.clip(img, 0.0, 1.0)

    return img


def generate_synthetic_digits_dataset(
    n_samples: int = 1000,
    fov_size: int = 8,
    seed: int = 42,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct n_samples synthetic geometric digit-like masks rendered on fov_size x fov_size.

    Args:
        n_samples: Total number of procedural samples (default 1000).
        fov_size: Field of view dimension (default 8).
        seed: Random seed for procedural generation.
        device: PyTorch device.

    Returns:
        Tuple of (complex_tensors, labels):
        - complex_tensors: (n_samples, fov_size, fov_size) in torch.complex64, range [0.0, 1.0].
        - labels: (n_samples,) integer class labels in [0, 9].
    """
    images = np.zeros((n_samples, fov_size, fov_size), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        digit = i % 10
        labels[i] = digit
        images[i] = generate_synthetic_digit(
            digit=digit,
            fov_size=fov_size,
            seed=seed + i,
        )

    tensor_real = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    # Cast to complex64 with zero imaginary component: u = A * exp(j * 0)
    complex_tensor = torch.complex(tensor_real, torch.zeros_like(tensor_real))
    labels_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.long)
    return complex_tensor, labels_tensor


def load_mnist_benchmark_dataset(
    n_samples: int = 1000,
    fov_size: int = 8,
    root: Union[str, Path] = "data/mnist",
    force_synthetic: bool = False,
    device: Union[torch.device, str] = "cpu",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Load test images from torchvision MNIST downsampled to fov_size, with procedural fallback.

    Args:
        n_samples: Number of benchmark images to return (default 1000).
        fov_size: Target square FOV resolution (default 8).
        root: Storage path for MNIST dataset.
        force_synthetic: If True, bypasses torchvision and generates procedural digits.
        device: Target tensor device.
        seed: Deterministic seed.

    Returns:
        Tuple of (images, labels, source_name):
        - images: (n_samples, fov_size, fov_size) torch.complex64 normalized to [0.0, 1.0].
        - labels: (n_samples,) torch.long labels.
        - source_name: 'torchvision_mnist' or 'procedural_synthetic'.
    """
    if force_synthetic:
        logger.info(f"Generating {n_samples} procedural synthetic digits (forced).")
        imgs, labels = generate_synthetic_digits_dataset(
            n_samples=n_samples, fov_size=fov_size, seed=seed, device=device
        )
        return imgs, labels, "procedural_synthetic"

    try:
        import torchvision
        import torchvision.transforms as T

        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading MNIST test dataset from '{root_path}'...")
        mnist_ds = torchvision.datasets.MNIST(
            root=str(root_path),
            train=False,
            download=True,
            transform=T.ToTensor(),
        )

        n_available = min(n_samples, len(mnist_ds))
        loader = torch.utils.data.DataLoader(
            mnist_ds,
            batch_size=n_available,
            shuffle=False,
        )
        batch_x, batch_y = next(iter(loader))  # (N, 1, 28, 28)

        # Downsample 28x28 -> fov_size x fov_size using area interpolation (energy conserving)
        x_downsampled = F.interpolate(
            batch_x,
            size=(fov_size, fov_size),
            mode="area",
        ).squeeze(1)  # (N, fov_size, fov_size)

        # Normalize per-image dynamic range to [0.0, 1.0]
        max_vals = torch.amax(x_downsampled, dim=(-2, -1), keepdim=True)
        min_vals = torch.amin(x_downsampled, dim=(-2, -1), keepdim=True)
        norm_range = max_vals - min_vals
        norm_range = torch.where(norm_range > 1e-6, norm_range, torch.ones_like(norm_range))
        x_norm = (x_downsampled - min_vals) / norm_range
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        # Cast to torch.complex64 on target device
        x_real = x_norm.to(device=device, dtype=torch.float32)
        complex_imgs = torch.complex(x_real, torch.zeros_like(x_real))
        labels_tensor = batch_y[:n_available].to(device=device, dtype=torch.long)

        logger.info(
            f"Successfully loaded {n_available} MNIST test images downsampled to {fov_size}x{fov_size}."
        )
        return complex_imgs, labels_tensor, "torchvision_mnist"

    except Exception as exc:
        logger.warning(
            f"Failed to load torchvision MNIST ({exc}). Falling back to procedural synthetic dataset."
        )
        imgs, labels = generate_synthetic_digits_dataset(
            n_samples=n_samples, fov_size=fov_size, seed=seed, device=device
        )
        return imgs, labels, "procedural_synthetic"
