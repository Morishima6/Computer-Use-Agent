from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROJECT_CACHE_ROOT = PROJECT_ROOT / ".cache"
TORCH_HOME = PROJECT_CACHE_ROOT / "torch"
TORCH_HUB_DIR = TORCH_HOME / "hub"


def configure_model_cache() -> None:
    PROJECT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_ROOT))
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))


configure_model_cache()


DEFAULT_PHASH_WEIGHT = 0.20
DEFAULT_MS_SSIM_WEIGHT = 0.35
DEFAULT_LPIPS_WEIGHT = 0.45
DEFAULT_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


@dataclass
class SimilarityReport:
    similarity: float
    phash_similarity: float
    phash_hamming_distance: int
    ms_ssim_similarity: float
    lpips_similarity: Optional[float]
    lpips_distance: Optional[float]
    aligned_size: Tuple[int, int]
    used_metrics: List[str]
    skipped_metrics: List[str]


def clamp_similarity(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    if abs(value - 1.0) < 1e-12:
        return 1.0
    if abs(value) < 1e-12:
        return 0.0
    return value


def normalize_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def resize_pair_to_common_size(
    image_a: Image.Image, image_b: Image.Image
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    if image_a.size == image_b.size:
        return image_a, image_b, image_a.size

    common_size = (
        min(image_a.width, image_b.width),
        min(image_a.height, image_b.height),
    )
    resized_a = image_a.resize(common_size, Image.Resampling.BICUBIC)
    resized_b = image_b.resize(common_size, Image.Resampling.BICUBIC)
    return resized_a, resized_b, common_size


def image_to_grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def compute_phash_bits(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    size = hash_size * highfreq_factor
    resized = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_coeff = dct(dct_rows, axis=1, norm="ortho")
    low_freq = dct_coeff[:hash_size, :hash_size]
    median = float(np.median(low_freq[1:, 1:]))
    return (low_freq > median).astype(np.uint8).reshape(-1)


def compute_phash_similarity(image_a: Image.Image, image_b: Image.Image) -> Tuple[float, int]:
    bits_a = compute_phash_bits(image_a)
    bits_b = compute_phash_bits(image_b)
    hamming_distance = int(np.count_nonzero(bits_a != bits_b))
    similarity = 1.0 - (hamming_distance / float(bits_a.size))
    return clamp_similarity(similarity), hamming_distance


def average_pool_2x(image: np.ndarray) -> np.ndarray:
    if image.shape[0] % 2 == 1:
        image = np.pad(image, ((0, 1), (0, 0)), mode="edge")
    if image.shape[1] % 2 == 1:
        image = np.pad(image, ((0, 0), (0, 1)), mode="edge")
    return (
        image[0::2, 0::2]
        + image[1::2, 0::2]
        + image[0::2, 1::2]
        + image[1::2, 1::2]
    ) / 4.0


def compute_ssim_and_cs(
    image_a: np.ndarray,
    image_b: np.ndarray,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 255.0,
) -> Tuple[float, float]:
    image_a = image_a.astype(np.float64, copy=False)
    image_b = image_b.astype(np.float64, copy=False)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_a = gaussian_filter(image_a, sigma=sigma)
    mu_b = gaussian_filter(image_b, sigma=sigma)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = gaussian_filter(image_a * image_a, sigma=sigma) - mu_a_sq
    sigma_b_sq = gaussian_filter(image_b * image_b, sigma=sigma) - mu_b_sq
    sigma_ab = gaussian_filter(image_a * image_b, sigma=sigma) - mu_ab

    sigma_a_sq = np.maximum(sigma_a_sq, 0.0)
    sigma_b_sq = np.maximum(sigma_b_sq, 0.0)

    luminance = (2.0 * mu_ab + c1) / (mu_a_sq + mu_b_sq + c1)
    contrast_structure = (2.0 * sigma_ab + c2) / (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = luminance * contrast_structure

    ssim_score = float(np.mean(ssim_map))
    cs_score = float(np.mean(contrast_structure))
    return ssim_score, cs_score


def compute_ms_ssim_similarity(
    image_a: Image.Image,
    image_b: Image.Image,
    weights: Tuple[float, ...] = DEFAULT_MS_SSIM_WEIGHTS,
) -> float:
    gray_a = image_to_grayscale_array(image_a)
    gray_b = image_to_grayscale_array(image_b)

    min_dim = min(gray_a.shape[0], gray_a.shape[1], gray_b.shape[0], gray_b.shape[1])
    max_levels = max(1, min(len(weights), int(np.floor(np.log2(max(min_dim, 1)))) - 1))
    effective_weights = weights[:max_levels]

    mcs_scores: List[float] = []
    current_a = gray_a
    current_b = gray_b

    for level in range(max_levels):
        ssim_score, cs_score = compute_ssim_and_cs(current_a, current_b)
        ssim_score = float(np.clip(ssim_score, 0.0, 1.0))
        cs_score = float(np.clip(cs_score, 0.0, 1.0))

        if level == max_levels - 1:
            msssim = 1.0
            for idx, cs_value in enumerate(mcs_scores):
                msssim *= cs_value ** effective_weights[idx]
            msssim *= ssim_score ** effective_weights[-1]
            return clamp_similarity(msssim)

        mcs_scores.append(cs_score)
        current_a = average_pool_2x(current_a)
        current_b = average_pool_2x(current_b)

    return 1.0


class LPIPSEvaluator:
    def __init__(self, net: str, device: str) -> None:
        import lpips
        import torch

        torch.hub.set_dir(str(TORCH_HUB_DIR))

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for LPIPS, but torch.cuda.is_available() is False.")

        self._torch = torch
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def _to_tensor(self, image: Image.Image):
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = np.transpose(array, (2, 0, 1))[None, ...]
        return self._torch.from_numpy(tensor).to(self.device)

    def distance(self, image_a: Image.Image, image_b: Image.Image) -> float:
        tensor_a = self._to_tensor(image_a)
        tensor_b = self._to_tensor(image_b)
        with self._torch.no_grad():
            value = self.model(tensor_a, tensor_b)
        return float(value.detach().cpu().item())


def lpips_distance_to_similarity(distance: float) -> float:
    return clamp_similarity(1.0 / (1.0 + max(distance, 0.0)))


def build_lpips_evaluator(
    lpips_net: str = "alex",
    device: str = "cpu",
    require_lpips: bool = False,
) -> Optional[LPIPSEvaluator]:
    return _build_lpips_evaluator_cached(lpips_net, device, require_lpips)


@lru_cache(maxsize=8)
def _build_lpips_evaluator_cached(
    lpips_net: str,
    device: str,
    require_lpips: bool,
) -> Optional[LPIPSEvaluator]:
    try:
        return LPIPSEvaluator(net=lpips_net, device=device)
    except Exception as exc:
        if require_lpips:
            raise RuntimeError(
                "LPIPS is required but unavailable. Install 'torch' and 'lpips', or set require_lpips=False."
            ) from exc
        print(
            f"[warn] LPIPS skipped: {exc}. Install 'torch' and 'lpips' to enable it.",
            file=sys.stderr,
        )
        return None


def weighted_similarity(metric_scores: dict[str, Optional[float]], metric_weights: dict[str, float]) -> Tuple[float, List[str], List[str]]:
    used_metrics: List[str] = []
    skipped_metrics: List[str] = []
    weight_sum = 0.0
    score_sum = 0.0

    for name, weight in metric_weights.items():
        score = metric_scores.get(name)
        if score is None:
            skipped_metrics.append(name)
            continue
        if weight <= 0:
            skipped_metrics.append(name)
            continue
        used_metrics.append(name)
        weight_sum += weight
        score_sum += weight * score

    if weight_sum <= 0:
        raise ValueError("No valid metric is available to compute the final similarity score.")

    return clamp_similarity(score_sum / weight_sum), used_metrics, skipped_metrics

def compare_screenshot_similarity_report(
    image_a_path: Union[str, Path],
    image_b_path: Union[str, Path],
    *,
    phash_weight: float = DEFAULT_PHASH_WEIGHT,
    ms_ssim_weight: float = DEFAULT_MS_SSIM_WEIGHT,
    lpips_weight: float = DEFAULT_LPIPS_WEIGHT,
    lpips_net: str = "alex",
    device: str = "cpu",
    require_lpips: bool = False,
) -> SimilarityReport:
    image_a = load_rgb_image(normalize_path(image_a_path))
    image_b = load_rgb_image(normalize_path(image_b_path))
    aligned_a, aligned_b, aligned_size = resize_pair_to_common_size(image_a, image_b)

    phash_similarity, phash_hamming_distance = compute_phash_similarity(image_a, image_b)
    ms_ssim_similarity = compute_ms_ssim_similarity(aligned_a, aligned_b)

    lpips_similarity: Optional[float] = None
    lpips_distance: Optional[float] = None
    lpips_evaluator = build_lpips_evaluator(
        lpips_net=lpips_net,
        device=device,
        require_lpips=require_lpips,
    )
    if lpips_evaluator is not None:
        lpips_distance = lpips_evaluator.distance(aligned_a, aligned_b)
        lpips_similarity = lpips_distance_to_similarity(lpips_distance)

    similarity, used_metrics, skipped_metrics = weighted_similarity(
        metric_scores={
            "phash": phash_similarity,
            "ms_ssim": ms_ssim_similarity,
            "lpips": lpips_similarity,
        },
        metric_weights={
            "phash": phash_weight,
            "ms_ssim": ms_ssim_weight,
            "lpips": lpips_weight,
        },
    )

    return SimilarityReport(
        similarity=clamp_similarity(similarity),
        phash_similarity=phash_similarity,
        phash_hamming_distance=phash_hamming_distance,
        ms_ssim_similarity=ms_ssim_similarity,
        lpips_similarity=lpips_similarity,
        lpips_distance=lpips_distance,
        aligned_size=aligned_size,
        used_metrics=used_metrics,
        skipped_metrics=skipped_metrics,
    )


def compare_screenshot_similarity(
    image_a_path: Union[str, Path],
    image_b_path: Union[str, Path],
    *,
    phash_weight: float = DEFAULT_PHASH_WEIGHT,
    ms_ssim_weight: float = DEFAULT_MS_SSIM_WEIGHT,
    lpips_weight: float = DEFAULT_LPIPS_WEIGHT,
    lpips_net: str = "alex",
    device: str = "cpu",
    require_lpips: bool = False,
) -> float:
    report = compare_screenshot_similarity_report(
        image_a_path=image_a_path,
        image_b_path=image_b_path,
        phash_weight=phash_weight,
        ms_ssim_weight=ms_ssim_weight,
        lpips_weight=lpips_weight,
        lpips_net=lpips_net,
        device=device,
        require_lpips=require_lpips,
    )
    return report.similarity
