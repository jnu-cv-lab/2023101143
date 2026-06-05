import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img_path = os.path.join(os.path.dirname(__file__), "test1.jpg")
img = cv2.imread(img_path)

# 2. 输出基本信息
print("图像尺寸:", img.shape)   # (高, 宽, 通道)
print("数据类型:", img.dtype)

# 3. 显示原图（OpenCV是BGR，需要转RGB）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# 4. 转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Gray Image")
plt.axis("off")
plt.show()

# 5. 保存灰度图
graypath=os.path.join(os.path.dirname(__file__), "gray.jpg")
cv2.imwrite(graypath, gray)

# 6. NumPy操作（裁剪左上角）
crop = img[0:100, 0:100]
croppath=os.path.join(os.path.dirname(__file__), "crop.jpg")
cv2.imwrite(croppath, crop)

# 输出一个像素值
print("左上角像素:", img[0,0])