import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# =========================
# 中文字体设置（适配你当前 WSL 环境）
# =========================
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 工具函数
# =========================
def read_gray_image(img_path):
    """读取灰度图像"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    return img


def downsample_image(img, scale=0.5, use_gaussian=False, ksize=(5, 5), sigma=1.0):
    """
    下采样
    use_gaussian=False: 不做预滤波直接缩小
    use_gaussian=True : 先高斯平滑再缩小
    """
    src = img.copy()
    if use_gaussian:
        src = cv2.GaussianBlur(src, ksize, sigma)

    new_w = int(src.shape[1] * scale)
    new_h = int(src.shape[0] * scale)

    small = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small


def restore_image(img_small, target_size, method='nearest'):
    """
    恢复图像到原始尺寸
    method: nearest / bilinear / bicubic
    """
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC
    }
    if method not in interp_map:
        raise ValueError("method 必须是 'nearest' / 'bilinear' / 'bicubic'")

    restored = cv2.resize(img_small, target_size, interpolation=interp_map[method])
    return restored


def calc_mse(img1, img2):
    """计算 MSE"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean((img1 - img2) ** 2)


def calc_psnr(img1, img2):
    """计算 PSNR"""
    mse = calc_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse)


def fft_spectrum(img):
    """
    计算二维傅里叶频谱
    返回：
    - fshift: 频谱中心化后的复数频域
    - magnitude_spectrum: 对数幅度谱
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1 + np.abs(fshift))
    return fshift, magnitude_spectrum


def dct2(img):
    """二维 DCT"""
    img_f = np.float32(img)
    return cv2.dct(img_f)


def log_display_dct(dct_coeff):
    """DCT 系数对数显示"""
    return np.log(1 + np.abs(dct_coeff))


def low_freq_energy_ratio(dct_coeff, ratio=0.25):
    """
    统计左上角低频区域能量占总能量比例
    ratio=0.25 表示取左上角 1/4 x 1/4 区域
    """
    h, w = dct_coeff.shape
    h_l = max(1, int(h * ratio))
    w_l = max(1, int(w * ratio))

    total_energy = np.sum(dct_coeff ** 2)
    low_energy = np.sum(dct_coeff[:h_l, :w_l] ** 2)

    if total_energy == 0:
        return 0.0
    return low_energy / total_energy


def save_single_image(img, save_path, cmap='gray'):
    """单独保存一张图片"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"图像已保存: {save_path}")
    plt.close(fig)


def show_and_save_images(title_list, image_list, save_path, cols=3, cmap='gray', figsize=(15, 8)):
    """批量保存图像（不弹窗）"""
    n = len(image_list)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=figsize)

    for i, (title, img) in enumerate(zip(title_list, image_list), 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存: {save_path}")
    plt.close(fig)


def save_bar_chart(x, y1, y2, width, xtick_labels, ylabel, title, label1, label2, save_path, figsize=(8, 5)):
    """保存双柱状图（不弹窗）"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, y1, width, label=label1)
    ax.bar(x + width / 2, y2, width, label=label2)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存: {save_path}")
    plt.close(fig)


def evaluate_restoration(original, restored_dict, group_name):
    """打印一组恢复结果的 MSE / PSNR"""
    print("=" * 60)
    print(group_name)
    print("=" * 60)
    for name, rec_img in restored_dict.items():
        mse = calc_mse(original, rec_img)
        psnr = calc_psnr(original, rec_img)
        print(f"{name}:")
        print(f"  MSE  = {mse:.4f}")
        print(f"  PSNR = {psnr:.4f} dB")
        print("-" * 40)


# =========================
# 主程序
# =========================
def main():
    # ========= 参数设置 =========
    img_path = "test.jpg"       # 改成你的图片路径
    scale = 0.5                 # 可改成 0.25
    low_freq_ratio = 0.25       # DCT 左上角低频区域比例
    gaussian_ksize = (5, 5)
    gaussian_sigma = 1.0

    # ========= 获取保存目录 =========
    img_abs_path = os.path.abspath(img_path)
    save_dir = os.path.dirname(img_abs_path)
    if save_dir == "":
        save_dir = "."

    print(f"原图路径: {img_abs_path}")
    print(f"输出目录: {save_dir}")

    # ========= 读取图像 =========
    img = read_gray_image(img_path)
    h, w = img.shape
    target_size = (w, h)

    # ========= 两种下采样 =========
    img_small_no_filter = downsample_image(
        img, scale=scale, use_gaussian=False
    )

    img_small_gaussian = downsample_image(
        img, scale=scale, use_gaussian=True,
        ksize=gaussian_ksize, sigma=gaussian_sigma
    )

    # ========= 不做预滤波：三种恢复 =========
    restored_no_filter = {
        '最近邻恢复(无预滤波)': restore_image(img_small_no_filter, target_size, method='nearest'),
        '双线性恢复(无预滤波)': restore_image(img_small_no_filter, target_size, method='bilinear'),
        '双三次恢复(无预滤波)': restore_image(img_small_no_filter, target_size, method='bicubic')
    }

    # ========= 高斯预滤波：三种恢复 =========
    restored_gaussian = {
        '最近邻恢复(高斯预滤波)': restore_image(img_small_gaussian, target_size, method='nearest'),
        '双线性恢复(高斯预滤波)': restore_image(img_small_gaussian, target_size, method='bilinear'),
        '双三次恢复(高斯预滤波)': restore_image(img_small_gaussian, target_size, method='bicubic')
    }

    # ========= 空间域评价 =========
    evaluate_restoration(img, restored_no_filter, "不做预滤波直接缩小：空间域评价指标")
    evaluate_restoration(img, restored_gaussian, "高斯平滑后缩小：空间域评价指标")

    # ========= 单图保存 =========
    save_single_image(img, os.path.join(save_dir, "00_原图.png"))
    save_single_image(img_small_no_filter, os.path.join(save_dir, "00_直接缩小.png"))
    save_single_image(img_small_gaussian, os.path.join(save_dir, "00_高斯后缩小.png"))

    save_single_image(restored_no_filter['最近邻恢复(无预滤波)'], os.path.join(save_dir, "00_最近邻恢复_无预滤波.png"))
    save_single_image(restored_no_filter['双线性恢复(无预滤波)'], os.path.join(save_dir, "00_双线性恢复_无预滤波.png"))
    save_single_image(restored_no_filter['双三次恢复(无预滤波)'], os.path.join(save_dir, "00_双三次恢复_无预滤波.png"))

    save_single_image(restored_gaussian['最近邻恢复(高斯预滤波)'], os.path.join(save_dir, "00_最近邻恢复_高斯预滤波.png"))
    save_single_image(restored_gaussian['双线性恢复(高斯预滤波)'], os.path.join(save_dir, "00_双线性恢复_高斯预滤波.png"))
    save_single_image(restored_gaussian['双三次恢复(高斯预滤波)'], os.path.join(save_dir, "00_双三次恢复_高斯预滤波.png"))

    # ========= 保存与显示：下采样和恢复结果 =========
    save_path_1 = os.path.join(save_dir, "01_下采样与恢复结果.png")
    show_and_save_images(
        [
            "原图",
            f"直接缩小(scale={scale})",
            f"高斯后缩小(scale={scale})",
            "最近邻恢复(无预滤波)",
            "双线性恢复(无预滤波)",
            "双三次恢复(无预滤波)",
            "最近邻恢复(高斯预滤波)",
            "双线性恢复(高斯预滤波)",
            "双三次恢复(高斯预滤波)"
        ],
        [
            img,
            img_small_no_filter,
            img_small_gaussian,
            restored_no_filter['最近邻恢复(无预滤波)'],
            restored_no_filter['双线性恢复(无预滤波)'],
            restored_no_filter['双三次恢复(无预滤波)'],
            restored_gaussian['最近邻恢复(高斯预滤波)'],
            restored_gaussian['双线性恢复(高斯预滤波)'],
            restored_gaussian['双三次恢复(高斯预滤波)']
        ],
        save_path=save_path_1,
        cols=3,
        figsize=(16, 12)
    )

    # ========= 傅里叶频谱分析 =========
    _, spec_original = fft_spectrum(img)
    _, spec_small_no_filter = fft_spectrum(img_small_no_filter)
    _, spec_small_gaussian = fft_spectrum(img_small_gaussian)
    _, spec_bilinear_no_filter = fft_spectrum(restored_no_filter['双线性恢复(无预滤波)'])
    _, spec_bilinear_gaussian = fft_spectrum(restored_gaussian['双线性恢复(高斯预滤波)'])

    save_path_2 = os.path.join(save_dir, "02_傅里叶频谱对比.png")
    show_and_save_images(
        [
            "原图频谱(对数显示)",
            "直接缩小频谱",
            "高斯后缩小频谱",
            "双线性恢复频谱(无预滤波)",
            "双线性恢复频谱(高斯预滤波)"
        ],
        [
            spec_original,
            spec_small_no_filter,
            spec_small_gaussian,
            spec_bilinear_no_filter,
            spec_bilinear_gaussian
        ],
        save_path=save_path_2,
        cols=3,
        figsize=(15, 8)
    )

    # ========= DCT 分析 =========
    dct_original = dct2(img)

    dct_no_filter = {
        '最近邻恢复(无预滤波)': dct2(restored_no_filter['最近邻恢复(无预滤波)']),
        '双线性恢复(无预滤波)': dct2(restored_no_filter['双线性恢复(无预滤波)']),
        '双三次恢复(无预滤波)': dct2(restored_no_filter['双三次恢复(无预滤波)'])
    }

    dct_gaussian = {
        '最近邻恢复(高斯预滤波)': dct2(restored_gaussian['最近邻恢复(高斯预滤波)']),
        '双线性恢复(高斯预滤波)': dct2(restored_gaussian['双线性恢复(高斯预滤波)']),
        '双三次恢复(高斯预滤波)': dct2(restored_gaussian['双三次恢复(高斯预滤波)'])
    }

    save_path_3 = os.path.join(save_dir, "03_DCT对比.png")
    show_and_save_images(
        [
            "原图DCT(对数显示)",
            "最近邻DCT(无预滤波)",
            "双线性DCT(无预滤波)",
            "双三次DCT(无预滤波)",
            "最近邻DCT(高斯预滤波)",
            "双线性DCT(高斯预滤波)",
            "双三次DCT(高斯预滤波)"
        ],
        [
            log_display_dct(dct_original),
            log_display_dct(dct_no_filter['最近邻恢复(无预滤波)']),
            log_display_dct(dct_no_filter['双线性恢复(无预滤波)']),
            log_display_dct(dct_no_filter['双三次恢复(无预滤波)']),
            log_display_dct(dct_gaussian['最近邻恢复(高斯预滤波)']),
            log_display_dct(dct_gaussian['双线性恢复(高斯预滤波)']),
            log_display_dct(dct_gaussian['双三次恢复(高斯预滤波)'])
        ],
        save_path=save_path_3,
        cols=3,
        figsize=(15, 12)
    )

    # ========= DCT 低频能量占比 =========
    print("=" * 60)
    print(f"DCT 左上角低频区域能量占比（区域比例={low_freq_ratio}）")
    print("=" * 60)

    energy_original = low_freq_energy_ratio(dct_original, ratio=low_freq_ratio)

    energy_no_filter = {
        name: low_freq_energy_ratio(coeff, ratio=low_freq_ratio)
        for name, coeff in dct_no_filter.items()
    }

    energy_gaussian = {
        name: low_freq_energy_ratio(coeff, ratio=low_freq_ratio)
        for name, coeff in dct_gaussian.items()
    }

    print(f"原图: {energy_original:.6f}\n")

    print("无预滤波：")
    for name, value in energy_no_filter.items():
        print(f"{name}: {value:.6f}")

    print("\n高斯预滤波：")
    for name, value in energy_gaussian.items():
        print(f"{name}: {value:.6f}")

    # ========= 柱状图：MSE 对比 =========
    method_names = ['最近邻', '双线性', '双三次']

    mse_no_filter = [
        calc_mse(img, restored_no_filter['最近邻恢复(无预滤波)']),
        calc_mse(img, restored_no_filter['双线性恢复(无预滤波)']),
        calc_mse(img, restored_no_filter['双三次恢复(无预滤波)'])
    ]

    mse_gaussian = [
        calc_mse(img, restored_gaussian['最近邻恢复(高斯预滤波)']),
        calc_mse(img, restored_gaussian['双线性恢复(高斯预滤波)']),
        calc_mse(img, restored_gaussian['双三次恢复(高斯预滤波)'])
    ]

    x = np.arange(len(method_names))
    width = 0.35

    save_path_4 = os.path.join(save_dir, "04_MSE对比柱状图.png")
    save_bar_chart(
        x=x,
        y1=mse_no_filter,
        y2=mse_gaussian,
        width=width,
        xtick_labels=method_names,
        ylabel="MSE",
        title="不同恢复方法的 MSE 对比",
        label1="无预滤波",
        label2="高斯预滤波",
        save_path=save_path_4
    )

    # ========= 柱状图：PSNR 对比 =========
    psnr_no_filter = [
        calc_psnr(img, restored_no_filter['最近邻恢复(无预滤波)']),
        calc_psnr(img, restored_no_filter['双线性恢复(无预滤波)']),
        calc_psnr(img, restored_no_filter['双三次恢复(无预滤波)'])
    ]

    psnr_gaussian = [
        calc_psnr(img, restored_gaussian['最近邻恢复(高斯预滤波)']),
        calc_psnr(img, restored_gaussian['双线性恢复(高斯预滤波)']),
        calc_psnr(img, restored_gaussian['双三次恢复(高斯预滤波)'])
    ]

    save_path_5 = os.path.join(save_dir, "05_PSNR对比柱状图.png")
    save_bar_chart(
        x=x,
        y1=psnr_no_filter,
        y2=psnr_gaussian,
        width=width,
        xtick_labels=method_names,
        ylabel="PSNR (dB)",
        title="不同恢复方法的 PSNR 对比",
        label1="无预滤波",
        label2="高斯预滤波",
        save_path=save_path_5
    )

    # ========= 柱状图：DCT低频能量占比 =========
    energy_no_filter_list = [
        energy_no_filter['最近邻恢复(无预滤波)'],
        energy_no_filter['双线性恢复(无预滤波)'],
        energy_no_filter['双三次恢复(无预滤波)']
    ]

    energy_gaussian_list = [
        energy_gaussian['最近邻恢复(高斯预滤波)'],
        energy_gaussian['双线性恢复(高斯预滤波)'],
        energy_gaussian['双三次恢复(高斯预滤波)']
    ]

    save_path_6 = os.path.join(save_dir, "06_DCT低频能量占比对比柱状图.png")
    save_bar_chart(
        x=x,
        y1=energy_no_filter_list,
        y2=energy_gaussian_list,
        width=width,
        xtick_labels=method_names,
        ylabel="低频能量占比",
        title="不同恢复方法的 DCT 低频能量占比对比",
        label1="无预滤波",
        label2="高斯预滤波",
        save_path=save_path_6
    )



if __name__ == "__main__":
    main()