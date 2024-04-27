#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
void print_data(const cv::Mat &img, const std::string &path)
{
    std::ofstream outputFile(path);
    // 将矩阵数据逐行写入文件
    if (outputFile.is_open())
    {
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                outputFile << img.at<double>(i, j) << " ";
            }
            outputFile << std::endl;
        }
        outputFile.close();
    }
    else
    {
        std::cout << "Failed to open file" << std::endl;
    }
}

// 图片归一化
cv::Mat normalizeImg(const cv::Mat &data)
{
    cv::Mat image32F;

    // 将CV_64F图像转换为CV_32F图像
    data.convertTo(image32F, CV_32F);
    cv::Mat normalizedData;

    double minValue, maxValue;
    cv::minMaxLoc(image32F, &minValue, &maxValue);

    if (maxValue != minValue)
    {
        cv::normalize(image32F, normalizedData, 0, 255, cv::NORM_MINMAX);
    }
    else
    {
        double value;
        if (maxValue > 255)
        {
            value = 255;
        }
        else if (maxValue < 0)
        {
            value = 0;
        }
        else
        {
            value = maxValue;
        }
        normalizedData = cv::Mat(image32F.size(), image32F.type(), cv::Scalar(value));
    }

    cv::Mat img;
    normalizedData.convertTo(img, CV_8U);

    return img;
}

// 输出图片
void print_img(cv::Mat data, const std::string &path)
{
    cv::Mat img = normalizeImg(data);
    cv::imwrite(path, img);
}

void print_m(const cv::Mat &matrix)
{
    print_data(matrix, "matrix.txt");
    print_img(matrix, "output.jpg");
}

// 添加结果线条
cv::Mat addRedLine(const cv::Mat &img, double angle, int central_not_print_num = 0)
{
    cv::Mat img_rgb;
    if (img.channels() == 1)
    {
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);
    }
    else if (img.channels() == 3)
    {
        img_rgb = img.clone();
    }
    else
    {
        // Handle unsupported image format (not grayscale or 3-channel)
        return cv::Mat();
    }

    int maxr = std::min(img.rows, img.cols) / 2;
    int h = img.rows / 2;
    int w = img.cols / 2;

    for (int r = 0; r < maxr; r++)
    {
        if (r < central_not_print_num)
        {
            continue;
        }
        int x1 = w + static_cast<int>(r * sin(angle / 360 * 2 * CV_PI));
        int y1 = h + static_cast<int>(r * cos(angle / 360 * 2 * CV_PI));
        int x2 = w - static_cast<int>(r * sin(angle / 360 * 2 * CV_PI));
        int y2 = h - static_cast<int>(r * cos(angle / 360 * 2 * CV_PI));

        cv::line(img_rgb, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
    }

    return img_rgb;
}

// 路径拼接
const std::string join(const std::string &dir, const std::string &name)
{
    fs::path path1 = dir;
    fs::path path2 = name;
    return path1 / path2;
}

// 输出列表
void print_vector(const std::vector<double> &sumList, const std::string &path)
{
    std::ofstream file(path);

    // 检查文件是否成功打开
    if (file.is_open())
    {
        // 使用迭代器遍历std::vector中的元素
        for (auto it = sumList.begin(); it != sumList.end(); ++it)
        {
            // 将当前元素写入文件流，并添加空格分隔符
            file << *it << " ";
        }

        // 关闭文件流
        file.close();
    }
}