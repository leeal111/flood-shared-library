/**
 * @file bitSTIV.h
 * @brief STIV算法C++头文件：包含了相关函数的接口
 * @version 0.1
 * @date 2024-04-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef _BIT_STIV_H_
#define _BIT_STIV_H_

#ifdef __cplusplus
extern "C"
{
#endif
    /**
     * @brief 求解给定STI图像纹理角度
     *
     * @param ifR2L 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入1
     * @param path STI图像的路径
     * @return double 以角度制为单位的浮点数，即纹理角度结果
     */
    double BIT_sti2angle_path(int ifR2L, const char *path);

    /**
     * @brief 求解给定STI图像的分数
     * 
     * @param ifR2L 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入1
     * @param path 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入1
     * @return double 0-1范围内的浮点小数，即图像分数结果
     */
    double BIT_sti2score_path(int ifR2L, const char *path);
#ifdef __cplusplus
}
#endif

#endif