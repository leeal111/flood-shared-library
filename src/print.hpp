#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>

void print_data(const cv::Mat &img)
{
    std::ofstream outputFile("matrix.txt");
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

cv::Mat normalizeImg(const cv::Mat &data)
{
    cv::Mat normalizedData;

    double minValue, maxValue;
    cv::minMaxLoc(data, &minValue, &maxValue);

    if (maxValue != minValue)
    {
        cv::normalize(data, normalizedData, 0, 255, cv::NORM_MINMAX);
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
        normalizedData = cv::Mat(data.size(), data.type(), cv::Scalar(value));
    }

    cv::Mat img;
    normalizedData.convertTo(img, CV_8U);

    return img;
}

void print_img(cv::Mat data)
{
    cv::Mat img = normalizeImg(data);
    cv::imwrite("output.jpg", img);
}

void print_m(const cv::Mat &matrix)
{
    print_data(matrix);
    print_img(matrix);
}

void print_vector(const std::vector<double> &sumList)
{
    std::ofstream file("data.txt");

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

        std::cout << "数据已成功保存到文件中。" << std::endl;
    }
    else
    {
        std::cout << "无法打开文件。" << std::endl;
    }
}