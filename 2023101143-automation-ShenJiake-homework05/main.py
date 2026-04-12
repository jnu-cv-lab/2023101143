import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================
# Matplotlib 中文显示设置
# =============================
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


# =============================
# 基础工具函数
# =============================
def ensure_output_dir():
    os.makedirs("outputs", exist_ok=True)


def save_image(name, img):
    ensure_output_dir()
    cv2.imwrite(f"outputs/{name}.png", img)


def show_image(title, img):
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_and_save(name, img, show=True):
    save_image(name, img)
    if show:
        show_image(name, img)


def print_analysis(name):
    print(f"\n==== {name} 几何性质分析 ====")
    print("1. 直线是否保持为直线：保持")

    if name == "相似变换":
        print("2. 平行线是否仍保持平行：保持")
        print("3. 两条垂直线是否仍垂直：保持")
        print("4. 圆是否仍保持为圆：保持")
    elif name == "仿射变换":
        print("2. 平行线是否仍保持平行：保持")
        print("3. 两条垂直线是否仍垂直：一般不保持")
        print("4. 圆是否仍保持为圆：一般不保持，通常变为椭圆")
    elif name == "透视变换":
        print("2. 平行线是否仍保持平行：一般不保持")
        print("3. 两条垂直线是否仍垂直：一般不保持")
        print("4. 圆是否仍保持为圆：一般不保持，通常变为圆锥曲线")


def resize_for_display(img, max_side=1400):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img.copy(), 1.0
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized, scale


def order_points(pts):
    """
    输入4个点，输出顺序：
    左上、右上、右下、左下
    """
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    rect[0] = pts[np.argmin(s)]     # 左上
    rect[2] = pts[np.argmax(s)]     # 右下
    rect[1] = pts[np.argmax(diff)]  # 右上
    rect[3] = pts[np.argmin(diff)]  # 左下

    return rect


# =============================
# 自动扩展画布，避免裁剪
# =============================
def warp_affine_full(img, M, border_value=(255, 255, 255)):
    h, w = img.shape[:2]

    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    transformed = cv2.transform(np.array([corners]), M)[0]

    min_x = int(np.floor(np.min(transformed[:, 0])))
    max_x = int(np.ceil(np.max(transformed[:, 0])))
    min_y = int(np.floor(np.min(transformed[:, 1])))
    max_y = int(np.ceil(np.max(transformed[:, 1])))

    new_w = max_x - min_x + 1
    new_h = max_y - min_y + 1

    M_adj = M.copy()
    M_adj[0, 2] -= min_x
    M_adj[1, 2] -= min_y

    warped = cv2.warpAffine(
        img,
        M_adj,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped


def warp_perspective_full(img, M, border_value=(255, 255, 255)):
    h, w = img.shape[:2]

    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    transformed = cv2.perspectiveTransform(np.array([corners]), M)[0]

    min_x = int(np.floor(np.min(transformed[:, 0])))
    max_x = int(np.ceil(np.max(transformed[:, 0])))
    min_y = int(np.floor(np.min(transformed[:, 1])))
    max_y = int(np.ceil(np.max(transformed[:, 1])))

    new_w = max_x - min_x + 1
    new_h = max_y - min_y + 1

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    M_adj = T @ M

    warped = cv2.warpPerspective(
        img,
        M_adj,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped


# =============================
# 白纸检测：先分割，再从轮廓极值提四角
# =============================
def detect_paper_contour(image):
    """
    返回：
        paper_quad: 4x2 float32，顺序为 左上、右上、右下、左下
        vis_mask: 掩膜图
        vis_contour: 轮廓与角点可视化图
    """
    img = image.copy()

    # 缩放处理，提高速度
    proc, scale = resize_for_display(img, max_side=1400)
    ph, pw = proc.shape[:2]

    # HSV分割白纸：低饱和、高亮度
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 80, 255]))

    # 形态学优化
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    vis_mask = mask.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, vis_mask, proc

    # 最大外轮廓，通常就是整张纸
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    if area < ph * pw * 0.1:
        return None, vis_mask, proc

    contour_vis = proc.copy()
    cv2.drawContours(contour_vis, [c], -1, (0, 255, 0), 2)

    pts = c.reshape(-1, 2).astype(np.float32)

    # 从轮廓点中找四个角点
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]   # 左上
    br = pts[np.argmax(s)]   # 右下
    tr = pts[np.argmax(d)]   # 右上
    bl = pts[np.argmin(d)]   # 左下

    paper_quad_proc = np.array([tl, tr, br, bl], dtype=np.float32)

    # 画角点与四边形
    draw_quad = paper_quad_proc.astype(np.int32)
    cv2.polylines(contour_vis, [draw_quad], True, (0, 0, 255), 3)

    for i, p in enumerate(draw_quad):
        cv2.circle(contour_vis, tuple(p), 8, (255, 0, 0), -1)
        cv2.putText(
            contour_vis,
            f"P{i + 1}",
            (p[0] + 10, p[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    # 映射回原图坐标
    paper_quad = paper_quad_proc / scale
    return paper_quad.astype(np.float32), vis_mask, contour_vis


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return warped, rect


def enhance_document(warp):
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    enhanced = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )
    return gray, enhanced


# =============================
# 1. 读取测试图
# =============================
img = cv2.imread("几何图形测试图.png")
if img is None:
    raise FileNotFoundError("未找到文件：几何图形测试图.png")

h, w = img.shape[:2]
show_and_save("原始图像", img, show=True)

# =============================
# 2. 相似变换
# =============================
angle = 25
scale = 0.85
center = (w // 2, h // 2)

M_sim = cv2.getRotationMatrix2D(center, angle, scale)
M_sim[:, 2] += np.array([120, 80], dtype=np.float32)

sim_img = warp_affine_full(img, M_sim, border_value=(255, 255, 255))
show_and_save("相似变换", sim_img, show=True)
print_analysis("相似变换")

# =============================
# 3. 仿射变换
# =============================
pts1_aff = np.float32([
    [120, 120],
    [w - 150, 120],
    [150, h - 150]
])

pts2_aff = np.float32([
    [80, 180],
    [w - 50, 100],
    [250, h - 80]
])

M_aff = cv2.getAffineTransform(pts1_aff, pts2_aff)
aff_img = warp_affine_full(img, M_aff, border_value=(255, 255, 255))
show_and_save("仿射变换", aff_img, show=True)
print_analysis("仿射变换")

# =============================
# 4. 透视变换
# =============================
pts1_pers = np.float32([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
])

pts2_pers = np.float32([
    [120, 80],
    [w - 60, 20],
    [w - 180, h - 40],
    [80, h - 120]
])

M_pers = cv2.getPerspectiveTransform(pts1_pers, pts2_pers)
pers_img = warp_perspective_full(img, M_pers, border_value=(255, 255, 255))
show_and_save("透视变换", pers_img, show=True)
print_analysis("透视变换")

# =============================
# 5. 透视校正
# =============================
doc_img = cv2.imread("存在透视畸变的平面图.jpg")
if doc_img is None:
    raise FileNotFoundError("未找到文件：存在透视畸变的平面图.jpg")

paper_quad, mask_vis, contour_vis = detect_paper_contour(doc_img)

show_and_save("纸张分割掩膜", mask_vis, show=True)
show_and_save("检测到的纸张轮廓", contour_vis, show=True)

if paper_quad is None:
    raise RuntimeError("未能检测到整张纸，请检查图片光照或边缘是否过暗。")

warped_doc, ordered_rect = four_point_transform(doc_img, paper_quad)

# 在原图上画最终角点
drawn = doc_img.copy()
pts_int = ordered_rect.astype(np.int32)
cv2.polylines(drawn, [pts_int], True, (0, 255, 0), 5)

for i, p in enumerate(pts_int):
    cv2.circle(drawn, tuple(p), 12, (0, 0, 255), -1)
    cv2.putText(
        drawn,
        f"P{i + 1}",
        (p[0] + 10, p[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

show_and_save("最终检测角点", drawn, show=True)
show_and_save("透视校正结果", warped_doc, show=True)

gray_doc, binary_doc = enhance_document(warped_doc)
show_and_save("校正后灰度图", gray_doc, show=True)
show_and_save("校正后二值增强图", binary_doc, show=True)

print("\n透视校正完成，结果已保存到 outputs 文件夹。")