#ifndef _BIT_STIV_Tradition_H_
#define _BIT_STIV_Tradition_H_

#ifdef __cplusplus
extern "C"
{
#endif

    // 参数意义：
    // ifR2L: 指定水流方向，如果为从左往右流则输入0，如果从右往左流则输入非零数
    // path: STI图像的路径
    //
    // 返回值：
    // 角度制的浮点数结果
    double BIT_sti2angle_path(int ifR2L, const char *path);
#ifdef __cplusplus
}
#endif

#endif