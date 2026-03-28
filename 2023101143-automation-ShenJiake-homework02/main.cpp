#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 任务1：读取图像
    Mat img = imread("./input1.jpg");  // 确保替换为正确的文件路径
    if (img.empty()) {
        cerr << "无法打开图像文件!" << endl;
        return -1;
    }

    // 任务2：输出图像基本信息
    cout << "图像宽度: " << img.cols << ", 图像高度: " << img.rows << ", 图像通道数: " << img.channels() << endl;

    // 任务3：显示原图
    namedWindow("原图", WINDOW_NORMAL);
    imshow("原图", img);

    // 任务4：转换为灰度图，并显示
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_NORMAL);
    imshow("灰度图", gray);

    // 任务5：保存处理结果
    imwrite("output_gray.jpg", gray);
    cout << "灰度图已保存为 output_gray.jpg" << endl;

    // 任务6：使用 NumPy 做简单操作，输出某个像素值（例如：左上角像素）
    Vec3b pixel = img.at<Vec3b>(0, 0);  // 获取左上角像素
    cout << "左上角像素值 (B, G, R): " << "B=" << int(pixel[0]) << ", G=" << int(pixel[1]) << ", R=" << int(pixel[2]) << endl;

    return 0;
}