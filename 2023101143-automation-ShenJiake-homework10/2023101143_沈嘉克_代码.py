"""
第12次课实验：实现并比较 Sinusoidal Position Encoding 与 RoPE.

运行方式：
    python 2023101143_沈嘉克_代码.py

脚本会在 outputs/ 中生成数值结果、图像和摘要 JSON。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def sinusoidal_position_encoding(seq_len: int, dim: int) -> np.ndarray:
    """生成 Transformer 论文中的 sinusoidal position encoding."""
    if dim % 2 != 0:
        raise ValueError("dim 必须为偶数，便于两两组成 sin/cos 通道。")

    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    pair_indices = np.arange(0, dim, 2, dtype=np.float64)
    div_terms = np.exp(-np.log(10000.0) * pair_indices / dim)

    pe = np.zeros((seq_len, dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(positions * div_terms)
    pe[:, 1::2] = np.cos(positions * div_terms)
    return pe


def rotate_2d(vector: np.ndarray, theta: float) -> np.ndarray:
    """将二维向量逆时针旋转 theta 弧度."""
    if vector.shape != (2,):
        raise ValueError("rotate_2d 只接受形状为 (2,) 的二维向量。")
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    return rotation @ vector


def rope_frequencies(dim: int) -> np.ndarray:
    """RoPE 每个二维子空间使用的角频率."""
    if dim % 2 != 0:
        raise ValueError("dim 必须为偶数。")
    return 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))


def apply_rope(x: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    """
    对最后一维应用 RoPE.

    x 的形状可以是 (seq_len, dim) 或 (..., seq_len, dim)。本实验主要使用前者。
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("最后一维 dim 必须为偶数。")

    seq_len = x.shape[-2]
    dim = x.shape[-1]
    if positions is None:
        positions = np.arange(seq_len, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] != seq_len:
        raise ValueError("positions 长度必须等于序列长度。")

    freqs = rope_frequencies(dim)
    angles = positions[:, None] * freqs[None, :]
    cos = np.cos(angles)
    sin = np.sin(angles)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = np.empty_like(x, dtype=np.float64)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    return rotated


def attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """计算缩放前的 attention score 矩阵 QK^T."""
    return q @ k.T


def verify_rope_relative_property(
    seq_len: int = 12, dim: int = 8, seed: int = 2023101143
) -> dict[str, float]:
    """
    验证 RoPE 的相对位置性质。

    如果同一个 q_base 和 k_base 被放在不同绝对位置 m,n 上，
    RoPE 后的点积只依赖相对距离 n-m，而不依赖 m 和 n 的绝对值。
    """
    rng = np.random.default_rng(seed)
    q_base = rng.normal(size=dim)
    k_base = rng.normal(size=dim)

    scores_by_delta: dict[int, list[float]] = {}
    for m in range(seq_len):
        for n in range(seq_len):
            q_m = apply_rope(q_base[None, :], positions=np.array([m]))[0]
            k_n = apply_rope(k_base[None, :], positions=np.array([n]))[0]
            scores_by_delta.setdefault(n - m, []).append(float(q_m @ k_n))

    max_std = max(float(np.std(values)) for values in scores_by_delta.values())
    max_range = max(float(np.ptp(values)) for values in scores_by_delta.values())

    return {
        "max_std_same_delta": max_std,
        "max_range_same_delta": max_range,
        "num_deltas": float(len(scores_by_delta)),
    }


def run_experiment() -> dict[str, object]:
    """运行全部数值实验并保存图像。"""
    rng = np.random.default_rng(2023101143)
    seq_len = 16
    dim = 8

    token_embeddings = rng.normal(size=(seq_len, dim))
    pe = sinusoidal_position_encoding(seq_len, dim)
    e_plus_pos = token_embeddings + pe

    q = rng.normal(size=(seq_len, dim))
    k = rng.normal(size=(seq_len, dim))
    q_rope = apply_rope(q)
    k_rope = apply_rope(k)

    raw_scores = attention_scores(q, k)
    rope_scores = attention_scores(q_rope, k_rope)
    epos_scores = attention_scores(e_plus_pos, e_plus_pos)

    relative_result = verify_rope_relative_property(seq_len=20, dim=dim)

    vector = np.array([1.0, 0.0])
    theta = np.pi / 3
    rotated_vector = rotate_2d(vector, theta)

    save_heatmap(pe, "Sinusoidal Position Encoding", OUTPUT_DIR / "sinusoidal_pe.png")
    save_heatmap(rope_scores, "RoPE Attention Scores", OUTPUT_DIR / "rope_scores.png")
    save_heatmap(epos_scores, "E + pos Attention Scores", OUTPUT_DIR / "epos_scores.png")
    save_relative_plot(dim, OUTPUT_DIR / "rope_relative_curve.png")

    summary = {
        "seq_len": seq_len,
        "dim": dim,
        "sinusoidal_shape": list(pe.shape),
        "rotated_vector_60deg": rotated_vector.round(6).tolist(),
        "raw_score_mean": float(raw_scores.mean()),
        "rope_score_mean": float(rope_scores.mean()),
        "epos_score_mean": float(epos_scores.mean()),
        "relative_property": relative_result,
        "figures": [
            "outputs/sinusoidal_pe.png",
            "outputs/epos_scores.png",
            "outputs/rope_scores.png",
            "outputs/rope_relative_curve.png",
        ],
    }

    with (OUTPUT_DIR / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def save_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("dimension / key position")
    plt.ylabel("position / query position")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_relative_plot(dim: int, path: Path) -> None:
    rng = np.random.default_rng(7)
    q_base = rng.normal(size=dim)
    k_base = rng.normal(size=dim)
    deltas = np.arange(-12, 13)
    scores = []
    for delta in deltas:
        q_0 = apply_rope(q_base[None, :], positions=np.array([0]))[0]
        k_delta = apply_rope(k_base[None, :], positions=np.array([delta]))[0]
        scores.append(q_0 @ k_delta)

    plt.figure(figsize=(6, 4))
    plt.plot(deltas, scores, marker="o", linewidth=1.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.title("RoPE Dot Product Changes with Relative Position")
    plt.xlabel("relative position n - m")
    plt.ylabel("dot product")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result, ensure_ascii=False, indent=2))
