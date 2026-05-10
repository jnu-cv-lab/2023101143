import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
OUTPUT_DIR = Path("assignment_outputs")


def save_sample_images(images, labels, output_path):
    """保存若干张样本图像及其真实标签。"""
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
    for ax, image, label in zip(axes.ravel(), images[:10], labels[:10]):
        ax.imshow(image, cmap="gray_r")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    fig.suptitle("Digits Dataset Sample Images", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, output_path):
    """保存混淆矩阵图像。"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix of Best Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return cm


def save_misclassified_examples(images, y_true, y_pred, output_path, max_examples=12):
    """保存若干个错误分类样本。"""
    wrong_indices = np.where(y_true != y_pred)[0]
    selected = wrong_indices[:max_examples]

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
    for ax in axes.ravel():
        ax.axis("off")

    for ax, idx in zip(axes.ravel(), selected):
        ax.imshow(images[idx], cmap="gray_r")
        ax.set_title(f"True: {y_true[idx]}  Pred: {y_pred[idx]}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Misclassified Examples", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return wrong_indices


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 任务 1：数据准备
    digits = load_digits()
    images = digits.images
    X = digits.data
    y = digits.target

    print("任务 1：数据准备")
    print(f"图像数量：{len(images)}")
    print(f"每张图像大小：{images[0].shape[0]} x {images[0].shape[1]}")
    print(f"类别标签：{digits.target_names.tolist()}")
    print(f"特征矩阵形状：{X.shape}，表示 {X.shape[0]} 张图像，每张图像 64 个像素特征")

    sample_path = OUTPUT_DIR / "sample_digits.png"
    save_sample_images(images, y, sample_path)
    print(f"样本图像已保存：{sample_path}")

    # 任务 2：数据划分
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X,
        y,
        images,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n任务 2：数据划分")
    print(f"训练集大小：{X_train.shape[0]}")
    print(f"测试集大小：{X_test.shape[0]}")
    # 任务 3：特征表示
    print("\n任务 3：特征表示")

    # 任务 4：模型训练
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    results = []
    predictions = {}

    print("\n任务 4 和任务 5：模型训练与结果比较")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))
        predictions[name] = y_pred
        print(f"{name:20s} 测试准确率：{acc:.4f}")

    results = sorted(results, key=lambda item: item[1], reverse=True)
    best_model_name, best_acc = results[0]
    worst_model_name, worst_acc = results[-1]

    accuracy_table_path = OUTPUT_DIR / "accuracy_table.csv"
    with accuracy_table_path.open("w", encoding="utf-8") as f:
        f.write("模型,测试准确率\n")
        for name, acc in results:
            f.write(f"{name},{acc:.4f}\n")

    print(f"\n准确率表格已保存：{accuracy_table_path}")
    print(f"准确率最高模型：{best_model_name}，准确率：{best_acc:.4f}")
    print(f"准确率最低模型：{worst_model_name}，准确率：{worst_acc:.4f}")

    # 任务 6：错误样本分析
    best_pred = predictions[best_model_name]
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    error_path = OUTPUT_DIR / "misclassified_examples.png"
    cm = save_confusion_matrix(y_test, best_pred, cm_path)
    wrong_indices = save_misclassified_examples(images_test, y_test, best_pred, error_path)

    print("\n任务 6：错误样本分析")
    print(f"用于错误分析的模型：{best_model_name}")
    print(f"混淆矩阵已保存：{cm_path}")
    print(f"错误分类样本图已保存：{error_path}")
    print(f"错误样本数量：{len(wrong_indices)}")

    # 找出最常见的混淆对，不统计对角线。
    confusion_pairs = []
    for true_label in range(10):
        for pred_label in range(10):
            if true_label != pred_label and cm[true_label, pred_label] > 0:
                confusion_pairs.append((true_label, pred_label, cm[true_label, pred_label]))
    confusion_pairs.sort(key=lambda item: item[2], reverse=True)

    print("最常见的混淆对（真实标签 -> 预测标签：次数）：")
    if confusion_pairs:
        for true_label, pred_label, count in confusion_pairs[:5]:
            print(f"{true_label} -> {pred_label}: {count}")
    else:
        print("该模型在测试集上没有错误分类样本。")


if __name__ == "__main__":
    main()
