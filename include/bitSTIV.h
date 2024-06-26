/**
 * @file bitSTIV.h
 * @brief STIV算法C++头文件：包含了相关函数的接口
 * @version 0.3
 * @date 2024-04-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef _BIT_STIV_H_
#define _BIT_STIV_H_

#ifdef _WIN32

#ifdef DLL_EXPORTS
#define DLL_DECL_EXPORT __declspec(dllexport)
#else
#define DLL_DECL_EXPORT __declspec(dllimport)
#endif

#else

#define DLL_DECL_EXPORT

#endif

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
    double DLL_DECL_EXPORT BIT_sti2angle_path(int ifR2L, const char *path);

    /**
     * @brief 求解给定STI图像的分数
     *
     * @param ifR2L 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入1
     * @param path STI图像的路径
     * @return double 0-1范围内的浮点小数，即图像分数结果
     */
    double DLL_DECL_EXPORT BIT_sti2score_path(int ifR2L, const char *path);

    /**
     * @brief 求解给定STI图像的纹理角度和分数
     *
     * @param ifR2L 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入1
     * @param path STI图像的路径
     * @param score 出参：0-1范围内的浮点小数，即图像分数结果
     * @return double 以角度制为单位的浮点数，即纹理角度结果
     */
    double DLL_DECL_EXPORT BIT_sti2angleWithscore_path(int ifR2L, const char *path, double *score);

    /**
     * @brief 求解给定STI图像的纹理角度和分数,并在
     *
     * @param ifR2L 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入1
     * @param path STI图像的路径
     * @param score 出参：0-1范围内的浮点小数，即图像分数结果
     * @param result_dir 中间结果保存目录。如果不存在，则会创建。如果传入空指针，退化到BIT_sti2angleWithscore_path函数。
     * @return double double 以角度制为单位的浮点数，即纹理角度结果
     */
    double DLL_DECL_EXPORT BIT_sti2angleWithscoreAndeval_path(int ifR2L, const char *img_path, double *score, const char *result_dir);

#ifdef __cplusplus
}
#endif
#undef DLL_DECL_EXPORT
#endif