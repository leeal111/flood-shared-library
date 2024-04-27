#include "bitSTIV.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>
#include "print.hpp"

double calculateScore(const std::vector<double> &sumList, int range_len = 5)
{
    int maxIndex = std::distance(sumList.begin(), std::max_element(sumList.begin(), sumList.end()));
    double total = sumList[maxIndex];

    for (int i = 1; i <= range_len; i++)
    {
        int index = maxIndex - i >= 0 ? maxIndex - i : 0;
        total += sumList[index];

        index = maxIndex + i < sumList.size() ? maxIndex + i : sumList.size() - 1;
        total += sumList[index];
    }

    return total / std::accumulate(sumList.begin(), sumList.end(), 0.0);
}

// 求取竖直差
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

// 转换特征到极坐标
cv::Mat xycrd2polarcrd(cv::Mat img, double res = 45, double theta = 45, double precision = 1, int rangeV = 0, double rangedivR = 2, int zeroNum = 0)
{
    int maxr = std::round(std::min(img.rows, img.cols) / rangedivR);
    int maxa = std::round(2 * theta / precision);
    int h = img.rows / 2;
    int w = img.cols / 2;

    cv::Mat dst = cv::Mat::zeros(maxa, maxr, CV_64FC1);
    for (int a = 0; a < maxa; a++)
    {
        double angle = (res - theta) + (a * precision);
        for (int r = 0; r < maxr; r++)
        {
            int h0 = h + static_cast<int>(r * std::sin(angle * CV_PI / 180));
            int w0 = w + static_cast<int>(r * std::cos(angle * CV_PI / 180));
            dst.at<double>(a, r) += img.at<double>(h0, w0);
            for (int i = 0; i < rangeV; i++)
            {
                double dangle = (i + 1) * precision;
                dst.at<double>(a, r) += img.at<double>(h + static_cast<int>(r * std::sin((angle - dangle) * CV_PI / 180)), w + static_cast<int>(r * std::cos((angle - dangle) * CV_PI / 180)));
                dst.at<double>(a, r) += img.at<double>(h + static_cast<int>(r * std::sin((angle + dangle) * CV_PI / 180)), w + static_cast<int>(r * std::cos((angle + dangle) * CV_PI / 180)));
            }
        }
    }

    cv::Mat zeroRegion = dst.colRange(0, zeroNum);
    zeroRegion.setTo(0);

    return dst;
}

// 过滤0频率值
void lowFreqFilter(cv::Mat &image)
{
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;

    image.col(center_x) = cv::Scalar(0, 0, 0);
    image.row(center_y) = cv::Scalar(0, 0, 0);
}

// 傅里叶变换
cv::Mat absFFTshift(cv::Mat image)
{
    cv::Mat complexImage;
    cv::dft(image, complexImage, cv::DFT_COMPLEX_OUTPUT);

    // 分割实部和虚部
    cv::Mat planes[2];
    cv::split(complexImage, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    // 以下的操作是移动图像  (零频移到中心)
    int cx = magnitude.cols / 2;
    int cy = magnitude.rows / 2;

    cv::Mat part1_r(magnitude, cv::Rect(0, 0, cx, cy)); // 元素坐标表示为(cx, cy)
    cv::Mat part2_r(magnitude, cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_r(magnitude, cv::Rect(0, cy, cx, cy));
    cv::Mat part4_r(magnitude, cv::Rect(cx, cy, cx, cy));

    cv::Mat temp;
    part1_r.copyTo(temp); // 左上与右下交换位置(实部)
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp); // 右上与左下交换位置(实部)
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    return magnitude;
}

// 图像轴对称删除
cv::Mat verticalDelete(cv::Mat image)
{
    cv::Mat image_l, image_v, image_r, image_r_flip;

    if (image.cols % 2 == 0)
    {
        image_l = image(cv::Rect(1, 0, image.cols / 2 - 1, image.rows));
        image_r = image(cv::Rect(image.cols / 2 + 1, 0, image.cols / 2 - 1, image.rows));
    }
    else
    {
        image_l = image(cv::Rect(0, 0, image.cols / 2, image.rows));
        image_r = image(cv::Rect(image.cols / 2 + 1, 0, image.cols / 2, image.rows));
    }
    image_v = image(cv::Rect(image.cols / 2, 0, 1, image.rows));
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

// 图像值幂运算
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

// Sobel算子
cv::Mat partSobel(cv::Mat image)
{
    cv::Mat img;
    cv::Sobel(image, img, CV_64FC1, 1, 1, 3);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<double>(i, j) < 0)
                img.at<double>(i, j) = 0;
        }
    }
    return img;
}

// 裁剪图像中心区域
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

// 简单检查参数
void exam_args(int ifR2L, const char *path, double *score, const char *result_path)
{
    if (ifR2L != 0 && ifR2L != 1)
    {
        std::cout << "ifR2L should be 0 or 1!" << std::endl;
    }
    if (path == nullptr)
    {
        std::cout << "path should not be nullptr!" << std::endl;
    }
    if (score == nullptr)
    {
        std::cout << "score should not be nullptr!" << std::endl;
    }
}

double sti2angle_IFFT(int ifR2L, const char *path, double *score, const char *result_path)
{
    // 读取图片、翻转图片并转换到灰度图
    exam_args(ifR2L, path, score, result_path);
    cv::Mat img = cv::imread(std::string(path));
    if (ifR2L)
        cv::flip(img, img, 1);

    if (img.channels() > 1)
    {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // 将图像转换为灰度图像
    }
    img.convertTo(img, CV_64FC1);

    // 消除不同位置的竖直亮度差异。
    cv::Mat img_std = std_filter(img).clone();

    // 提取倾斜特征，使得fft的特征更明显，同时也是归一化
    cv::Mat img_clr = partSobel(img_std).clone();

    // 傅里叶变换
    cv::Mat img_fft = absFFTshift(img_clr).clone();
    lowFreqFilter(img_fft); // 删除0频率值

    // 过滤由于partSobel产生的噪声
    cv::Mat img_fft_clr = verticalDelete(img_fft).clone();

    // 幂运算
    cv::Mat img_fft_pow = imgPow(img_fft_clr, 2).clone();

    // 仅取中心部分
    cv::Mat img_fft_crop = imgCrop(img_fft_pow).clone();

    // 傅里叶变换并取幅值
    cv::Mat img_fe = absFFTshift(img_fft_crop).clone();
    lowFreqFilter(img_fe);

    // 更严格的取向判断
    cv::Mat img_fe_clr = verticalDelete(img_fe.clone());
    cv::Mat img_fe_ = img_fe_clr;

    // 后续流程根据特征图转换极坐标并求取最大值，与一阶段一致
    double res = 45;
    double theta = 45;
    double precision = 1;
    int rangeV = 1;
    double rangedivR = 2.5;
    int zeroNum = 20;
    cv::Mat polar = xycrd2polarcrd(img_fe_, res, theta, precision, rangeV, rangedivR, zeroNum);
    cv::Mat sum_list;
    cv::reduce(polar, sum_list, 1, cv::REDUCE_SUM);
    std::vector<double> sum_list_vec;
    sum_list.copyTo(sum_list_vec);
    // 计算STI分数值
    *score = calculateScore(sum_list_vec);
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

    if (result_path != nullptr)
    {
        fs::create_directories(std::string(result_path));
        print_img(img, join(result_path, "ORIGIN.jpg"));
        print_img(img_std, join(result_path, "std.jpg"));
        print_img(img_clr, join(result_path, "clr.jpg"));
        print_img(img_fft, join(result_path, "fft.jpg"));
        print_img(img_fft_clr, join(result_path, "fftclr.jpg"));
        print_img(img_fft_crop, join(result_path, "fftcrop.jpg"));
        print_img(img_fe, join(result_path, "ifft.jpg"));
        print_img(img_fe_clr, join(result_path, "ifftclr.jpg"));
        print_vector(sum_list_vec, join(result_path, "sumlist.txt"));
        print_vector(sum_list_2_vec, join(result_path, "sumlist_2.txt"));
        print_img(addRedLine(normalizeImg(img_fft), -(90 - result)), join(result_path, "FFTRES.jpg"));
        print_img(addRedLine(normalizeImg(img_fe), result), join(result_path, "IFFTRES.jpg"));
    }

    return result;
}

#ifdef __cplusplus
extern "C"
{
#endif
    double BIT_sti2angle_path(int ifR2L, const char *path)
    {
        double score;
        return sti2angle_IFFT(ifR2L, path, &score, nullptr);
    }

    double BIT_sti2score_path(int ifR2L, const char *path)
    {
        double score;
        sti2angle_IFFT(ifR2L, path, &score, nullptr);
        return score;
    }

    double BIT_sti2angleWithscore_path(int ifR2L, const char *path, double *score)
    {
        return sti2angle_IFFT(ifR2L, path, score, nullptr);
    }

    double BIT_sti2angleWithscoreAndeval_path(int ifR2L, const char *path, double *score, const char *result_dir)
    {
        return sti2angle_IFFT(ifR2L, path, score, result_dir);
    }
#ifdef __cplusplus
}
#endif
