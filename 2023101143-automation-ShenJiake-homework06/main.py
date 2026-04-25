import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


BOX_PATH = "box.png"
SCENE_PATH = "box_in_scene.png"
OUTPUT_DIR = "outputs"
RANSAC_THRESHOLD = 5.0


@dataclass
class ExperimentResult:
    method: str
    nfeatures: int | None
    kp_box_count: int
    kp_scene_count: int
    descriptor_shape_box: tuple[int, ...] | None
    descriptor_shape_scene: tuple[int, ...] | None
    match_count: int
    inlier_count: int
    inlier_ratio: float
    homography: np.ndarray | None
    located: bool
    elapsed_ms: float


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def imwrite(name: str, image: np.ndarray) -> None:
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, name)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def draw_keypoints(gray: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    return cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)


def detect_orb(gray: np.ndarray, nfeatures: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


def match_orb(desc_box: np.ndarray, desc_scene: np.ndarray) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_box, desc_scene)
    return sorted(matches, key=lambda m: m.distance)


def compute_homography(
    kp_box: list[cv2.KeyPoint],
    kp_scene: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray | None, np.ndarray | None, int, float]:
    if len(matches) < 4:
        return None, None, 0, 0.0

    src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if mask is None:
        return homography, None, 0, 0.0

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(matches) if matches else 0.0
    return homography, inlier_mask, inlier_count, inlier_ratio


def draw_inlier_matches(
    box_gray: np.ndarray,
    kp_box: list[cv2.KeyPoint],
    scene_gray: np.ndarray,
    kp_scene: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray | None,
) -> np.ndarray:
    matches_mask = None if inlier_mask is None else inlier_mask.astype(int).tolist()
    return cv2.drawMatches(
        box_gray,
        kp_box,
        scene_gray,
        kp_scene,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def locate_box(scene_gray: np.ndarray, box_shape: tuple[int, int], homography: np.ndarray | None) -> tuple[np.ndarray, bool]:
    scene_color = cv2.cvtColor(scene_gray, cv2.COLOR_GRAY2BGR)
    if homography is None:
        return scene_color, False

    h, w = box_shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography)
    if not np.isfinite(projected).all():
        return scene_color, False

    pts = projected.reshape(-1, 2)
    scene_h, scene_w = scene_gray.shape[:2]
    area = cv2.contourArea(pts.astype(np.float32))
    located = (
        area > 1000
        and np.all(pts[:, 0] > -scene_w * 0.5)
        and np.all(pts[:, 0] < scene_w * 1.5)
        and np.all(pts[:, 1] > -scene_h * 0.5)
        and np.all(pts[:, 1] < scene_h * 1.5)
    )

    color = (0, 255, 0) if located else (0, 0, 255)
    cv2.polylines(scene_color, [np.int32(projected)], True, color, 4, cv2.LINE_AA)
    return scene_color, located


def run_orb_experiment(
    box_gray: np.ndarray,
    scene_gray: np.ndarray,
    nfeatures: int,
    save_main_outputs: bool = False,
) -> ExperimentResult:
    start = time.perf_counter()
    kp_box, desc_box = detect_orb(box_gray, nfeatures)
    kp_scene, desc_scene = detect_orb(scene_gray, nfeatures)
    if desc_box is None or desc_scene is None:
        raise RuntimeError("ORB failed to compute descriptors.")

    matches = match_orb(desc_box, desc_scene)
    homography, inlier_mask, inlier_count, inlier_ratio = compute_homography(kp_box, kp_scene, matches)
    located_image, located = locate_box(scene_gray, box_gray.shape, homography)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if save_main_outputs:
        imwrite("task1_box_orb_keypoints.png", draw_keypoints(box_gray, kp_box))
        imwrite("task1_box_in_scene_orb_keypoints.png", draw_keypoints(scene_gray, kp_scene))

        all_initial_matches = cv2.drawMatches(
            box_gray,
            kp_box,
            scene_gray,
            kp_scene,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        imwrite("task2_orb_all_initial_matches.png", all_initial_matches)

        top_n = min(50, len(matches))
        top_matches = cv2.drawMatches(
            box_gray,
            kp_box,
            scene_gray,
            kp_scene,
            matches[:top_n],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        imwrite("task2_orb_top50_initial_matches.png", top_matches)

        ransac_matches = draw_inlier_matches(box_gray, kp_box, scene_gray, kp_scene, matches, inlier_mask)
        imwrite("task3_orb_ransac_inlier_matches.png", ransac_matches)
        imwrite("task4_orb_object_location.png", located_image)

    imwrite(f"task6_orb_nfeatures_{nfeatures}_location.png", located_image)

    return ExperimentResult(
        method="ORB",
        nfeatures=nfeatures,
        kp_box_count=len(kp_box),
        kp_scene_count=len(kp_scene),
        descriptor_shape_box=desc_box.shape,
        descriptor_shape_scene=desc_scene.shape,
        match_count=len(matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        homography=homography,
        located=located,
        elapsed_ms=elapsed_ms,
    )


def run_sift_experiment(box_gray: np.ndarray, scene_gray: np.ndarray) -> ExperimentResult | None:
    if not hasattr(cv2, "SIFT_create"):
        return None

    start = time.perf_counter()
    sift = cv2.SIFT_create()
    kp_box, desc_box = sift.detectAndCompute(box_gray, None)
    kp_scene, desc_scene = sift.detectAndCompute(scene_gray, None)
    if desc_box is None or desc_scene is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(desc_box, desc_scene, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda m: m.distance)
    homography, inlier_mask, inlier_count, inlier_ratio = compute_homography(kp_box, kp_scene, good_matches)
    located_image, located = locate_box(scene_gray, box_gray.shape, homography)
    elapsed_ms = (time.perf_counter() - start) * 1000

    top_n = min(50, len(good_matches))
    sift_good_matches = cv2.drawMatches(
        box_gray,
        kp_box,
        scene_gray,
        kp_scene,
        good_matches[:top_n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    sift_ransac_matches = draw_inlier_matches(box_gray, kp_box, scene_gray, kp_scene, good_matches, inlier_mask)
    imwrite("optional_sift_top50_good_matches.png", sift_good_matches)
    imwrite("optional_sift_ransac_inlier_matches.png", sift_ransac_matches)
    imwrite("optional_sift_object_location.png", located_image)

    return ExperimentResult(
        method="SIFT",
        nfeatures=None,
        kp_box_count=len(kp_box),
        kp_scene_count=len(kp_scene),
        descriptor_shape_box=desc_box.shape,
        descriptor_shape_scene=desc_scene.shape,
        match_count=len(good_matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        homography=homography,
        located=located,
        elapsed_ms=elapsed_ms,
    )


def format_matrix(matrix: np.ndarray | None) -> str:
    if matrix is None:
        return "Homography 估计失败"
    return "\n".join("[" + ", ".join(f"{value:.6f}" for value in row) + "]" for row in matrix)


def success_text(value: bool) -> str:
    return "成功" if value else "失败"


def print_console_report(orb_1000: ExperimentResult, orb_results: list[ExperimentResult], sift_result: ExperimentResult | None) -> None:
    print("\n========== 任务 1：ORB 特征检测与描述 ==========")
    print("1. ORB 创建方式：cv2.ORB_create(nfeatures=1000)")
    print("2. detectAndCompute()：已得到关键点和描述子")
    print(f"3. box.png 关键点数量：{orb_1000.kp_box_count}")
    print(f"4. box_in_scene.png 关键点数量：{orb_1000.kp_scene_count}")
    print(f"5. box.png 描述子维度：{orb_1000.descriptor_shape_box}")
    print(f"6. box_in_scene.png 描述子维度：{orb_1000.descriptor_shape_scene}")
    print("7. 输出图：outputs/task1_box_orb_keypoints.png")
    print("8. 输出图：outputs/task1_box_in_scene_orb_keypoints.png")

    print("\n========== 任务 2：ORB 特征匹配 ==========")
    print("1. 匹配器：cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)")
    print("2. 匹配结果：已按 distance 从小到大排序")
    print(f"3. 总匹配数量：{orb_1000.match_count}")
    print("4. 全部初始匹配图：outputs/task2_orb_all_initial_matches.png")
    print("5. 前 50 个匹配图：outputs/task2_orb_top50_initial_matches.png")

    print("\n========== 任务 3：RANSAC 剔除错误匹配 ==========")
    print("1. Homography 估计：cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)")
    print(f"2. 总匹配数量：{orb_1000.match_count}")
    print(f"3. RANSAC 内点数量：{orb_1000.inlier_count}")
    print(f"4. 内点比例：{orb_1000.inlier_ratio:.3f}")
    print("5. RANSAC 后匹配图：outputs/task3_orb_ransac_inlier_matches.png")
    print("6. Homography 矩阵：")
    print(format_matrix(orb_1000.homography))

    print("\n========== 任务 4：目标定位 ==========")
    print("1. 已获取 box.png 四个角点")
    print("2. 已使用 cv2.perspectiveTransform() 投影角点")
    print("3. 已使用 cv2.polylines() 绘制目标边框")
    print(f"4. 定位是否成功：{success_text(orb_1000.located)}")
    print("5. 目标定位图：outputs/task4_orb_object_location.png")

    print("\n========== 任务 6：参数对比实验 ==========")
    print("nfeatures\t模板图关键点数\t场景图关键点数\t匹配数量\tRANSAC内点数\t内点比例\t是否成功定位")
    for r in orb_results:
        print(f"{r.nfeatures}\t\t{r.kp_box_count}\t\t{r.kp_scene_count}\t\t{r.match_count}\t\t{r.inlier_count}\t\t{r.inlier_ratio:.3f}\t\t{success_text(r.located)}")
    print("结论：匹配数量通常随 nfeatures 增大而增加，但内点比例不一定提高，特征点越多不一定定位越好。")

    print("\n========== 选做任务：SIFT 特征匹配 ==========")
    if sift_result is None:
        print("本机 OpenCV 不支持 SIFT_create，未完成 SIFT 选做实验。")
    else:
        print("1. SIFT 创建方式：cv2.SIFT_create()")
        print("2. 匹配方式：BFMatcher(cv2.NORM_L2) + knnMatch(k=2) + Lowe ratio test(0.75)")
        print(f"3. SIFT good matches 数量：{sift_result.match_count}")
        print(f"4. SIFT RANSAC 内点数量：{sift_result.inlier_count}")
        print(f"5. SIFT 内点比例：{sift_result.inlier_ratio:.3f}")
        print(f"6. SIFT 定位是否成功：{success_text(sift_result.located)}")
        print("7. SIFT 前 50 个 good matches 图：outputs/optional_sift_top50_good_matches.png")
        print("8. SIFT RANSAC 匹配图：outputs/optional_sift_ransac_inlier_matches.png")
        print("9. SIFT 目标定位图：outputs/optional_sift_object_location.png")


def main() -> None:
    ensure_output_dir()
    box_gray = read_gray(BOX_PATH)
    scene_gray = read_gray(SCENE_PATH)

    orb_results = []
    for nfeatures in (500, 1000, 2000):
        result = run_orb_experiment(
            box_gray,
            scene_gray,
            nfeatures=nfeatures,
            save_main_outputs=(nfeatures == 1000),
        )
        orb_results.append(result)

    orb_1000 = next(r for r in orb_results if r.nfeatures == 1000)
    sift_result = run_sift_experiment(box_gray, scene_gray)
    print_console_report(orb_1000, orb_results, sift_result)


if __name__ == "__main__":
    main()

