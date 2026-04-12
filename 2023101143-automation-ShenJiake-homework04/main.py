import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re

# =========================
# Matplotlib 中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 生成测试图
# =========================
def generate_checkerboard(size=512, num_checks=16):
    """
    生成棋盘格图像，像素范围 [0,255]
    """
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    block = size // num_checks
    board = ((xx // block) + (yy // block)) % 2
    img = (board * 255).astype(np.uint8)
    return img


def generate_chirp(size=512, f0=2, f1=60):
    """
    生成二维 chirp 图像，频率从左到右逐渐变高
    像素范围 [0,255]
    """
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    freq = f0 + (f1 - f0) * xx
    phase = 2 * np.pi * freq * yy
    img = 0.5 + 0.5 * np.sin(phase)
    img = (img * 255).astype(np.uint8)
    return img


# =========================
# 2. 下采样与滤波
# =========================
def downsample_direct(img, M):
    """
    直接抽样下采样
    """
    return img[::M, ::M]


def gaussian_then_downsample(img, M, sigma):
    """
    先高斯滤波，再下采样
    OpenCV 的 sigmaX=sigma，ksize=(0,0) 表示自动根据 sigma 算核大小
    """
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    ds = blurred[::M, ::M]
    return blurred, ds


def resize_back_nearest(img_small, target_shape):
    """
    最近邻上采样回原尺寸，便于误差比较
    """
    H, W = target_shape[:2]
    return cv2.resize(img_small, (W, H), interpolation=cv2.INTER_NEAREST)


def resize_back_linear(img_small, target_shape):
    """
    双线性上采样回原尺寸
    """
    H, W = target_shape[:2]
    return cv2.resize(img_small, (W, H), interpolation=cv2.INTER_LINEAR)


# =========================
# 3. FFT 频谱
# =========================
def fft_spectrum(img):
    """
    计算频谱图，用于观察高频成分
    """
    img_f = img.astype(np.float32)
    dft = np.fft.fft2(img_f)
    dft_shift = np.fft.fftshift(dft)
    mag = np.log1p(np.abs(dft_shift))
    return mag


# =========================
# 4. 工具函数
# =========================
def sanitize_filename(text):
    """
    将中文标题或特殊字符转换成适合做文件名的字符串
    """
    text = str(text).strip()
    text = text.replace('\n', '_')
    text = re.sub(r'[\\/:*?"<>|]', '_', text)
    text = re.sub(r'\s+', '_', text)
    return text


def save_single_image(img, title, save_dir, index, cmap='gray'):
    """
    单独保存每一张图
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{index:02d}_{sanitize_filename(title)}.png"
    filepath = os.path.join(save_dir, filename)

    plt.figure(figsize=(5, 5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"已保存: {filepath}")


def show_images(images, titles, cmap='gray', figsize=(16, 8), save_dir=None, group_name="默认分组"):
    """
    显示拼图 + 每张子图单独保存
    """
    if save_dir is None:
        save_dir = "results"

    os.makedirs(save_dir, exist_ok=True)

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    # 1. 显示拼图
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, i)
        if len(img.shape) == 2:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=12)
        plt.axis('off')
    plt.tight_layout()

    collage_path = os.path.join(save_dir, f"{sanitize_filename(group_name)}_拼图.png")
    plt.savefig(collage_path, dpi=200, bbox_inches='tight')
    print(f"已保存拼图: {collage_path}")
    plt.show()
    plt.close()

    # 2. 每张单独保存
    for i, (img, title) in enumerate(zip(images, titles), 1):
        save_single_image(img, title, save_dir, i, cmap=cmap)


# =========================
# 5. 01第一部分：混叠观察
# =========================
def experiment_part1(img, M=4, sigma=None, name="测试图"):
    """
    01第一部分：
    - 直接下采样
    - 高斯后下采样
    - 频谱对比
    """
    if sigma is None:
        sigma = 0.45 * M

    direct_ds = downsample_direct(img, M)
    blurred, blur_ds = gaussian_then_downsample(img, M, sigma)

    spec_orig = fft_spectrum(img)
    spec_direct_ds = fft_spectrum(direct_ds)
    spec_blur = fft_spectrum(blurred)
    spec_blur_ds = fft_spectrum(blur_ds)

    save_dir1 = os.path.join("results", f"{name}_01第一部分_图像对比")
    show_images(
        [img, direct_ds, blurred, blur_ds],
        [
            f"{name}-原图",
            f"{name}-直接下采样 M={M}",
            f"{name}-高斯滤波 σ={sigma:.2f}",
            f"{name}-滤波后下采样"
        ],
        figsize=(14, 8),
        save_dir=save_dir1,
        group_name=f"{name}_01第一部分_图像对比"
    )

    save_dir2 = os.path.join("results", f"{name}_01第一部分_频谱对比")
    show_images(
        [spec_orig, spec_direct_ds, spec_blur, spec_blur_ds],
        [
            f"{name}-原图频谱",
            f"{name}-直接下采样后频谱 M={M}",
            f"{name}-高斯滤波后频谱 σ={sigma:.2f}",
            f"{name}-滤波后下采样频谱"
        ],
        figsize=(14, 8),
        save_dir=save_dir2,
        group_name=f"{name}_01第一部分_频谱对比"
    )


# =========================
# 6. 02第二部分：固定 M=4，测试不同 sigma
# =========================
def mse(img1, img2):
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    return np.mean((a - b) ** 2)


def experiment_part2(img, M=4, sigma_list=(0.5, 1.0, 2.0, 4.0), name="测试图"):
    """
    固定 M=4，测试不同 σ
    做法：
    - 先滤波再下采样
    - 再上采样回原图大小
    - 与原图比较误差
    """
    results = []
    errors = []

    for sigma in sigma_list:
        blurred, ds = gaussian_then_downsample(img, M, sigma)
        recon = resize_back_linear(ds, img.shape)
        err = mse(img, recon)

        results.append(recon)
        errors.append(err)

    theory_sigma = 0.45 * M
    blurred_t, ds_t = gaussian_then_downsample(img, M, theory_sigma)
    recon_t = resize_back_linear(ds_t, img.shape)
    err_t = mse(img, recon_t)

    images = [img] + results + [recon_t]
    titles = ["原图"] + [f"σ={s}, MSE={e:.2f}" for s, e in zip(sigma_list, errors)] + \
             [f"理论值 σ={theory_sigma:.2f}, MSE={err_t:.2f}"]

    save_dir = os.path.join("results", f"{name}_02第二部分_sigma实验")
    show_images(
        images,
        titles,
        figsize=(18, 10),
        save_dir=save_dir,
        group_name=f"{name}_02第二部分_sigma实验"
    )

    print(f"\n{name}：固定 M=4 时各 sigma 的误差：")
    for s, e in zip(sigma_list, errors):
        print(f"sigma={s:<4} -> MSE={e:.4f}")
    print(f"理论值 sigma≈0.45M={theory_sigma:.2f} -> MSE={err_t:.4f}")


# =========================
# 7. 03第三部分：自适应下采样（重写）
# =========================
def gradient_map(img):
    """
    用 Sobel 计算梯度强度图
    """
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag


def estimate_block_gradient_map(img, block_size=32):
    """
    计算每个块的平均梯度
    返回:
        grad_map: 原图大小梯度图
        block_grad: 每个块的平均梯度图
    """
    grad_map = gradient_map(img)
    H, W = img.shape

    bh = H // block_size
    bw = W // block_size

    block_grad = np.zeros((bh, bw), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            y0 = i * block_size
            y1 = (i + 1) * block_size
            x0 = j * block_size
            x1 = (j + 1) * block_size

            block = grad_map[y0:y1, x0:x1]
            block_grad[i, j] = np.mean(block)

    return grad_map, block_grad


def gradient_to_M_map(block_grad, M_values=(2, 4, 6)):
    """
    根据块平均梯度映射局部 M
    梯度大 -> M 小
    梯度小 -> M 大
    """
    g = block_grad.astype(np.float32)
    gmin, gmax = g.min(), g.max()
    g_norm = (g - gmin) / (gmax - gmin + 1e-8)

    M_map = np.zeros_like(g, dtype=np.int32)

    # 高频区域：保留更多采样点 -> 小M
    M_map[g_norm > 0.60] = M_values[0]   # 2
    M_map[(g_norm > 0.25) & (g_norm <= 0.60)] = M_values[1]  # 4
    M_map[g_norm <= 0.25] = M_values[2]  # 6

    return M_map, g_norm


def adaptive_downsample_reconstruct(img, block_size=32, M_values=(2, 4, 6)):
    """
    按块进行自适应下采样重建：
    1) 梯度估计局部复杂度
    2) 由复杂度决定局部 M
    3) sigma = 0.45 * M
    4) 每块分别滤波 -> 下采样 -> 上采样重建
    """
    H, W = img.shape

    grad_map, block_grad = estimate_block_gradient_map(img, block_size=block_size)
    M_map, g_norm = gradient_to_M_map(block_grad, M_values=M_values)

    bh, bw = M_map.shape
    recon = np.zeros_like(img, dtype=np.uint8)

    sigma_map_vis = np.zeros((H, W), dtype=np.float32)
    M_map_vis = np.zeros((H, W), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            y0 = i * block_size
            y1 = (i + 1) * block_size
            x0 = j * block_size
            x1 = (j + 1) * block_size

            block = img[y0:y1, x0:x1]
            M_local = int(M_map[i, j])
            sigma_local = 0.45 * M_local

            # 先局部滤波
            blurred = cv2.GaussianBlur(
                block,
                (0, 0),
                sigmaX=sigma_local,
                sigmaY=sigma_local
            )

            # 局部下采样
            ds = blurred[::M_local, ::M_local]

            # 为了和原图比较，重建回块原大小
            rec_block = cv2.resize(
                ds,
                (block.shape[1], block.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            recon[y0:y1, x0:x1] = rec_block
            sigma_map_vis[y0:y1, x0:x1] = sigma_local
            M_map_vis[y0:y1, x0:x1] = M_local

    # 可视化
    sigma_vis = 255 * (sigma_map_vis - sigma_map_vis.min()) / (sigma_map_vis.max() - sigma_map_vis.min() + 1e-8)
    M_vis = 255 * (M_map_vis - M_map_vis.min()) / (M_map_vis.max() - M_map_vis.min() + 1e-8)

    return grad_map, M_vis.astype(np.uint8), sigma_vis.astype(np.uint8), recon


def experiment_part3(img, M=4, name="测试图"):
    """
    03第三部分：
    - 用梯度分析估计局部 M 值
    - 对不同区域用不同滤波
    - 和全图统一下采样对比误差图
    """
    # ===== 自适应方案 =====
    grad, M_vis, sigma_vis, adaptive_rec = adaptive_downsample_reconstruct(
        img,
        block_size=32,
        M_values=(2, 4, 6)
    )

    # ===== 统一方案 =====
    uniform_sigma = 0.45 * M
    uniform_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=uniform_sigma, sigmaY=uniform_sigma)
    uniform_ds = uniform_blur[::M, ::M]
    uniform_rec = resize_back_linear(uniform_ds, img.shape)

    # ===== 误差图 =====
    err_uniform = cv2.absdiff(img, uniform_rec)
    err_adaptive = cv2.absdiff(img, adaptive_rec)

    save_dir1 = os.path.join("results", f"{name}_03第三部分_局部M与滤波")
    show_images(
        [img, grad, M_vis, sigma_vis],
        ["原图", "梯度图", "局部M图", "局部sigma图"],
        figsize=(14, 8),
        save_dir=save_dir1,
        group_name=f"{name}_03第三部分_局部M与滤波"
    )

    save_dir2 = os.path.join("results", f"{name}_03第三部分_重建与误差图")
    show_images(
        [uniform_rec, adaptive_rec, err_uniform, err_adaptive],
        [
            f"统一滤波重建图 MSE={mse(img, uniform_rec):.2f}",
            f"自适应滤波重建图 MSE={mse(img, adaptive_rec):.2f}",
            "统一滤波误差图",
            "自适应滤波误差图"
        ],
        figsize=(14, 8),
        save_dir=save_dir2,
        group_name=f"{name}_03第三部分_重建与误差图"
    )

    print(f"{name} 统一滤波 MSE = {mse(img, uniform_rec):.4f}")
    print(f"{name} 自适应滤波 MSE = {mse(img, adaptive_rec):.4f}")


# =========================
# 8. 主程序
# =========================
def main():
    os.makedirs("results", exist_ok=True)

    checker = generate_checkerboard(size=512, num_checks=16)
    chirp = generate_chirp(size=512, f0=2, f1=60)

    # 01第一部分
    print("==== 01第一部分：混叠观察 ====")
    experiment_part1(checker, M=4, name="棋盘格")
    experiment_part1(chirp, M=4, sigma=3.0, name="Chirp图")

    # 02第二部分
    print("\n==== 02第二部分：固定 M=4 测试不同 sigma ====")
    experiment_part2(checker, M=4, sigma_list=(0.5, 1.0, 2.0, 4.0), name="棋盘格")
    experiment_part2(chirp, M=4, sigma_list=(0.5, 1.0, 2.0, 4.0), name="Chirp图")

    # 03第三部分
    print("\n==== 03第三部分：自适应下采样 ====")
    experiment_part3(checker, M=4, name="棋盘格")
    experiment_part3(chirp, M=4, name="Chirp图")


if __name__ == "__main__":
    main()