#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/imgproc.hpp"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include "ocv_util.h"

using namespace cv;
using namespace std;

static cv::Mat bytesToMat1(u8 *bytes, int width, int height)
{
    cv::Mat image = cv::Mat(height,width,CV_8UC1,bytes).clone(); // make a copy
    return image;
}

static cv::Mat bytesToMat3(u8 *bytes, int width, int height)
{
    cv::Mat image = cv::Mat(height,width,CV_8UC3,bytes).clone(); // make a copy
    return image;
}

int OcvUtil_BytesToImag1(u8 *bytes, int width, int height, const char *fileName)
{
    cv::Mat image = cv::Mat(height,width,CV_8UC1,bytes).clone(); // make a copy
    cv::imwrite(fileName, image);
    return 0;
}

int OcvUtil_BytesToImag3(u8 *bytes, int width, int height, const char *fileName)
{
    cv::Mat image = cv::Mat(height,width,CV_8UC3,bytes).clone(); // make a copy
    cv::imwrite(fileName, image);
    return 0;
}

void ocv_util()
{
    printf("%s()\n", __func__);

    {
        u8 *ptr = (u8 *)calloc(3*80*80, 1);
        for (int i = 0; i < 3*80*80; i+=3) {
            ptr[i] = 128;
        }

        cv::Mat image = bytesToMat3(ptr, 80, 80);

        cv::imwrite("my.jpg", image);
    }
    {
        u8 *ptr = (u8 *)calloc(1*80*80, 1);
        for (int i = 0; i < 1*80*80; i+=1) {
            ptr[i] = 128;
        }

        cv::Mat image = bytesToMat1(ptr, 80, 80);

        cv::imwrite("my2.jpg", image);
    }
}