#include "bitSTIV.h"
#include <stdio.h>
void ComputeAngle(int i,const char * str)
{
    printf("%lf\n", BIT_sti2angle_path(i,str));
}
void ComputeScore(int i, const char* str)
{
    printf("%lf\n", BIT_sti2score_path(i,str));
}
