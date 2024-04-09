#include "bitSTIV.h"
#include <stdio.h>
void ComputeAngle()
{
    printf("%lf\n", BIT_sti2angle_path(1, "../../tests/imgs/test.jpg"));
}
void ComputeScore()
{
    printf("%lf\n", BIT_sti2score_path(0, "../../tests/imgs/test.jpg"));
}
