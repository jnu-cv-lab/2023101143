# 第13课作业：羽毛球动作识别

本项目完成“基于 MediaPipe Pose 与骨架序列 Transformer 的羽毛球击球动作识别”实验。程序将羽毛球击球视频转换为人体骨架时间序列，并使用轻量级 Transformer Encoder 完成 6 类动作分类。

## 实验环境

| 名称 | 版本 |
| --- | --- |
| Python | 3.14.4 |
| MediaPipe | 0.10.35 |
| PyTorch | 2.12.0+cpu |
| OpenCV | 4.13.0 |
| NumPy | 2.4.4 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |

MediaPipe Tasks 使用的模型文件：

```text
models/pose_landmarker_lite.task
```

## 文件说明

| 提交项 | 文件 | 内容 |
| --- | --- | --- |
| 预处理代码 | `01_preprocess.py` | 读取视频，使用 MediaPipe Pose 提取关键点，重采样、归一化并保存 `.npy` |
| 训练代码 | `02_train.py` | 定义 `Dataset`、`DataLoader`、Skeleton Transformer 模型和训练循环 |
| 测试与推理代码 | `03_test_infer.py` | 输出测试准确率、混淆矩阵、分类报告、单样本 logits 和 softmax 概率 |
| 可选项 | `04_visualize_skeleton.py` | 生成 1-2 张视频片段的骨架可视化图片 |
| 公共模块 | `hw13_common.py` | MediaPipe 提取、归一化、模型、训练、评估、绘图和报告生成等公共函数 |




