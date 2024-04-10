#include <stdlib.h>
#include "bitSTIV.h"
#include <stdio.h>
int main(int argc, char **argv)
{
    int i = atoi(argv[1]);
    double score;
    printf("%lf\n", BIT_sti2angle_path(i, argv[2]));
    printf("%lf\n", BIT_sti2score_path(i, argv[2]));
    double angle = BIT_sti2angleWithscore_path(i, argv[2], &score);
    printf("%lf %lf\n", angle, score);
}