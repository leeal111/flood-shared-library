#include "bitSTIV.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>

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

void save(cv::Mat sti)
{
    sti = normalizeImg(sti);
    cv::imwrite("output.jpg", sti);
}

void printMatType(const cv::Mat &matrix)
{
    int type = matrix.type();

    if (type == CV_8UC1)
    {
        std::cout << "8位无符号单通道图像" << std::endl;
    }
    else if (type == CV_8UC3)
    {
        std::cout << "8位无符号三通道图像" << std::endl;
    }
    else if (type == CV_8SC1)
    {
        std::cout << "8位有符号单通道图像" << std::endl;
    }
    else if (type == CV_32FC1)
    {
        std::cout << "32位浮点单通道图像" << std::endl;
    }
    else if (type == CV_32SC1)
    {
        std::cout << "32位有符号单通道图像" << std::endl;
    }
    else
    {
        std::cout << "未知类型" << std::endl;
    }
}

void print(const cv::Mat &matrix)
{
    printMatType(matrix);

    save(matrix);

    // 数据保存
    const std::string &filename = "output.txt";
    std::ofstream outputFile(filename); // 创建一个输出流
    if (!outputFile.is_open())
    {
        std::cout << "无法打开文件：" << filename << std::endl;
        return;
    }
    int elemSize = matrix.elemSize();
    int elemType = matrix.type();

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            if (elemType == CV_8UC1)
            {
                outputFile << static_cast<int>(matrix.at<uchar>(i, j)) << " ";
            }
            else if (elemType == CV_8SC1)
            {
                outputFile << static_cast<int>(matrix.at<char>(i, j)) << " ";
            }
            // 其他单字节元素类型（如CV_8UC3、CV_8SC3等）可以根据需要进行类似的处理
        }
        outputFile << std::endl; // 每行结束后换行
    }
}

cv::Mat std_filter(cv::Mat sti)
{
    cv::Mat normalizedImage = sti.clone();
    std::vector<cv::Mat> vm;
    for (int col = 0; col < normalizedImage.cols; col++)
    {
        cv::Mat column = normalizedImage.col(col);

        // 计算每一列的均值和标准差
        cv::Scalar meanScalar, varianceScalar;
        cv::meanStdDev(column, meanScalar, varianceScalar);
        double mean = meanScalar[0];
        double variance = varianceScalar[0];

        // 对每一列执行标准差归一化
        column = (column - mean) / variance;
        vm.push_back(column);
    }
    cv::hconcat(vm, normalizedImage);

    return normalizedImage;
}

cv::Mat xycrd2polarcrd(cv::Mat img, float res = 45, float theta = 45, float precision = 1, int rangeV = 0, float rangedivR = 2, int zeroNum = 0)
{
    int maxr = std::round(std::min(img.rows, img.cols) / rangedivR);
    int maxa = std::round(2 * theta / precision);
    int h = img.rows / 2;
    int w = img.cols / 2;

    cv::Mat dst = cv::Mat::zeros(maxa, maxr, CV_32FC1);
    for (int a = 0; a < maxa; a++)
    {
        float angle = (res - theta) + (a * precision);
        for (int r = 0; r < maxr; r++)
        {
            int h0 = h + static_cast<int>(r * std::sin(angle * CV_PI / 180));
            int w0 = w + static_cast<int>(r * std::cos(angle * CV_PI / 180));
            dst.at<float>(a, r) += img.at<float>(h0, w0);
            for (int i = 0; i < rangeV; i++)
            {
                float dangle = (i + 1) * precision;
                dst.at<float>(a, r) += img.at<float>(h + static_cast<int>(r * std::sin((angle - dangle) * CV_PI / 180)), w + static_cast<int>(r * std::cos((angle - dangle) * CV_PI / 180)));
                dst.at<float>(a, r) += img.at<float>(h + static_cast<int>(r * std::sin((angle + dangle) * CV_PI / 180)), w + static_cast<int>(r * std::cos((angle + dangle) * CV_PI / 180)));
            }
        }
    }

    cv::Mat zeroRegion = dst.colRange(0, zeroNum);
    zeroRegion.setTo(0);

    return dst;
}

void lowFreqFilter(cv::Mat &image)
{
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;

    image.col(center_x) = cv::Scalar(0, 0, 0);
    image.row(center_y) = cv::Scalar(0, 0, 0);
}

cv::Mat absFFTshift(cv::Mat image)
{
    cv::Mat complexImage;
    cv::dft(image, complexImage, cv::DFT_COMPLEX_OUTPUT);

    // 分割实部和虚部
    cv::Mat planes[2];
    cv::split(complexImage, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);

    cv::Mat result = magnitude;

    int cx = result.cols / 2;
    int cy = result.rows / 2;

    cv::Mat q0(result, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(result, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(result, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(result, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    return result;
}

cv::Mat verticalDelete(cv::Mat image)
{
    cv::Mat image_l, image_v, image_r, image_r_flip;

    if (image.cols % 2 == 0)
    {
        image_l = image(cv::Rect(1, 0, image.cols / 2 - 1, image.rows));
    }
    else
    {
        image_l = image(cv::Rect(0, 0, image.cols / 2, image.rows));
    }

    image_v = image(cv::Rect(image.cols / 2, 0, 1, image.rows));

    image_r = image(cv::Rect(image.cols / 2 + 1, 0, image.cols - (image.cols / 2 + 1), image.rows));
    cv::flip(image_r, image_r_flip, 1);

    cv::Mat min_value = cv::min(image_l, image_r_flip);
    cv::Mat image_l_res = image_l - min_value;
    cv::Mat image_r_res;
    cv::flip(image_r_flip - min_value, image_r_res, 1);

    image_v = cv::Mat::zeros(image_v.size(), image_v.type());

    cv::Mat result;
    cv::hconcat(image_l_res, image_v, result);
    cv::hconcat(result, image_r_res, result);

    return result;
}

cv::Mat imgPow(cv::Mat image, double powNum = -1)
{
    cv::Mat img_clr = verticalDelete(image);
    double maxv = 0;
    cv::minMaxLoc(img_clr, nullptr, &maxv);

    if (powNum < 0)
    {
        powNum = std::log(255) / std::log(maxv);
    }

    cv::pow(image, powNum, image);

    return image;
}

cv::Mat partSobel(cv::Mat image)
{
    cv::Mat img;
    cv::Sobel(image, img, CV_32FC1, 1, 1, 3);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<float>(i, j) < 0)
                img.at<float>(i, j) = 0;
        }
    }
    return img;
}

cv::Mat imgCrop(cv::Mat image, int rangedivR = 10)
{
    int h = image.rows;
    int w = image.cols;
    int l = std::min(h, w) / rangedivR;

    // 计算图像中心坐标
    int center_x = w / 2;
    int center_y = h / 2;

    // 计算正方形的左上角和右下角坐标
    int x1 = center_x - l / 2;
    int y1 = center_y - l / 2;
    int x2 = center_x + l / 2;
    int y2 = center_y + l / 2;

    // 创建剪裁掩膜
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    cv::rectangle(mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), -1);

    // 将图像与掩膜相乘，将正方形之外的像素置为 0
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);

    return result;
}

double sti2angle_IFFT(cv::Mat img)
{
    ::print(img);
    cv::Mat img_std = std_filter(img);
    // ::print(img_std);
    cv::Mat img_clr = partSobel(img_std);
    // ::print(img_clr);
    cv::Mat img_fft = absFFTshift(img_clr);
    // ::print(img_fft);
    lowFreqFilter(img_fft);
    // ::print(img_fft);
    cv::Mat img_fft_clr = verticalDelete(img_fft);
    // ::print(img_fft_clr);
    cv::Mat img_fft_pow = imgPow(img_fft_clr, 2);
    // save(img_fft_pow);
    cv::Mat img_fft_crop = imgCrop(img_fft_pow);
    // save(img_fft_crop);
    cv::Mat img_fe = absFFTshift(img_fft_crop);
    // save(img_fe);
    lowFreqFilter(img_fe);
    // save(img_fe);
    cv::Mat img_fe_clr = verticalDelete(img_fe);
    // save(img_fe_clr);
    cv::Mat img_fe_ = img_fe_clr;

    float res = 45;
    float theta = 45;
    float precision = 1;
    int rangeV = 1;
    float rangedivR = 2.5;
    int zeroNum = 20;
    // float res = (res - theta) + (np.argmax(sum_list_2) * precision);
    cv::Mat polar = xycrd2polarcrd(img_fe_, res, theta, precision, rangeV, rangedivR, zeroNum);
    int rows = polar.rows;
    int cols = polar.cols;

    // std::cout << "矩阵的行数：" << rows << std::endl;
    // std::cout << "矩阵的列数：" << cols << std::endl;
    cv::Mat sum_list;
    cv::reduce(polar, sum_list, 1, cv::REDUCE_SUM);
    std::vector<double> sum_list_vec;
    sum_list.copyTo(sum_list_vec);

    int maxIndex = 0; // 假设最大元素的序号为0
    for (int i = 1; i < sum_list_vec.size(); i++)
    {
        if (sum_list_vec[i] > sum_list_vec[maxIndex])
        {
            maxIndex = i;
        }
    }
    res = (res - theta) + (maxIndex * precision);

    if (res < 2)
    {
        res = 2;
    }
    if (res > 88)
    {
        res = 88;
    }

    theta = 2;
    precision = 0.1;
    cv::Mat polar_2 = xycrd2polarcrd(img_fe_, res, theta, precision, rangeV, rangedivR, zeroNum);
    cv::Mat sum_list_2;
    cv::reduce(polar_2, sum_list_2, 1, cv::REDUCE_SUM);

    std::vector<double> sum_list_2_vec;
    sum_list_2.copyTo(sum_list_2_vec);

    maxIndex = 0; // 假设最大元素的序号为0
    for (int i = 1; i < sum_list_2_vec.size(); i++)
    {
        if (sum_list_2_vec[i] > sum_list_2_vec[maxIndex])
        {
            maxIndex = i;
        }
    }
    res = (res - theta) + (maxIndex * precision);

    double result = 90 - res;
    if (res > 90)
    {
        result = res - 90;
    }

    return result;
}
double sti2angle(cv::Mat img)
{
    if (img.channels() > 1)
    {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // 将图像转换为灰度图像
    }
    img.convertTo(img, CV_32FC1);
    return sti2angle_IFFT(img);
}

#ifdef __cplusplus
extern "C"
{
#endif
    double BIT_sti2angle_path(int ifR2L, const char *path)
    {
        cv::Mat image = cv::imread(std::string(path));
        if (ifR2L == 1)
            cv::flip(image, image, 1);
        return sti2angle(image);
    }
#ifdef __cplusplus
}
#endif
