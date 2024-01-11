#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "basic.h"
// #include "sys.h"
// #include "sys_pc_api.h"
#include <chrono>

#include "ocv_util.h"

extern int app_fitEllipse( int argc, char** argv );

//using namespace cv;
typedef unsigned char byte;

byte * matToBytes(cv::Mat image)
{
   int size = image.total() * image.elemSize();
   byte * bytes = new byte[size];  // you will have to delete[] that later
   memcpy(bytes, image.data, size * sizeof(byte));
   return bytes;
}

cv::Mat bytesToMat(byte *bytes, int width, int height)
{
    cv::Mat image = cv::Mat(height,width,CV_8UC3,bytes).clone(); // make a copy
    return image;
}

int app_image_to_bytes()
{
    /*
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    */
    cv::Mat image;
    //image = imread( argv[1], 1 );
    image = cv::imread("../../pupil_data/pieces/func3/pics/1.jpg", 1);
    if (image.elemSize() == 0) {
        image = cv::imread("../pupil_data/pieces/func3/pics/1.jpg", 1);
    }
    if (image.elemSize() == 0) {
        image = cv::imread("pupil_data/pieces/func3/pics/1.jpg", 1);
    }
    printf("h = %d\n", image.size().height);
    printf("w = %d\n", image.size().width);
    printf("total = %lu\n", image.total());
    printf("elemSize = %lu\n", image.elemSize());
    //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY); Flake: may use this w/o <opencv2/imgproc/types_c.h>
    cv::cvtColor(image, image, CV_BGR2GRAY);
    printf("h = %d\n", image.size().height);
    printf("w = %d\n", image.size().width);
    printf("total = %lu\n", image.total());
    printf("elemSize = %lu\n", image.elemSize());
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);
    return 0;
}

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

const int w = 500;
int levels = 3;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

static void on_trackbar(int, void*)
{
    Mat cnt_img = Mat::zeros(w, w, CV_8UC3);
    int _levels = levels - 3;
    drawContours( cnt_img, contours, _levels <= 0 ? 3 : -1, Scalar(128,255,255),
                  3, LINE_AA, hierarchy, std::abs(_levels) );

    imshow("contours", cnt_img);
}

int app_contour()
{
    Mat img = Mat::zeros(w, w, CV_8UC1);
    //Draw 6 faces
    for( int i = 0; i < 6; i++ )
    {
        int dx = (i%2)*250 - 30;
        int dy = (i/2)*150;
        const Scalar white = Scalar(255);
        const Scalar black = Scalar(0);

        if( i == 0 )
        {
            for( int j = 0; j <= 10; j++ )
            {
                double angle = (j+5)*CV_PI/21;
                line(img, Point(cvRound(dx+100+j*10-80*cos(angle)),
                    cvRound(dy+100-90*sin(angle))),
                    Point(cvRound(dx+100+j*10-30*cos(angle)),
                    cvRound(dy+100-30*sin(angle))), white, 1, 8, 0);
            }
        }

        ellipse( img, Point(dx+150, dy+100), Size(100,70), 0, 0, 360, white, -1, 8, 0 );
        ellipse( img, Point(dx+115, dy+70), Size(30,20), 0, 0, 360, black, -1, 8, 0 );
        ellipse( img, Point(dx+185, dy+70), Size(30,20), 0, 0, 360, black, -1, 8, 0 );
        ellipse( img, Point(dx+115, dy+70), Size(15,15), 0, 0, 360, white, -1, 8, 0 );
        ellipse( img, Point(dx+185, dy+70), Size(15,15), 0, 0, 360, white, -1, 8, 0 );
        ellipse( img, Point(dx+115, dy+70), Size(5,5), 0, 0, 360, black, -1, 8, 0 );
        ellipse( img, Point(dx+185, dy+70), Size(5,5), 0, 0, 360, black, -1, 8, 0 );
        ellipse( img, Point(dx+150, dy+100), Size(10,5), 0, 0, 360, black, -1, 8, 0 );
        ellipse( img, Point(dx+150, dy+150), Size(40,10), 0, 0, 360, black, -1, 8, 0 );
        ellipse( img, Point(dx+27, dy+100), Size(20,35), 0, 0, 360, white, -1, 8, 0 );
        ellipse( img, Point(dx+273, dy+100), Size(20,35), 0, 0, 360, white, -1, 8, 0 );
    }
    //show the faces
    namedWindow( "image", 1 );
    imshow( "image", img );
    //Extract the contours so that
    vector<vector<Point> > contours0;
    findContours( img, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    contours.resize(contours0.size());
    for( size_t k = 0; k < contours0.size(); k++ )
        approxPolyDP(Mat(contours0[k]), contours[k], 3, true);

    namedWindow( "contours", 1 );
    createTrackbar( "levels+3", "contours", &levels, 7, on_trackbar );

    on_trackbar(0,0);
    waitKey();

    return 0;
}

void app_image_to_c_array(const char *file_with_path)
{
    //const char * file_with_path = "pupil_data/pieces/func2/data/data_func2b/0/145800.215907685.png";

    //char filename[256];
    //syspc_file_search(file_with_path, filename, ARRAY_LEN(filename));

    cv::Mat image, gray;
    //image = cv::imread(filename, 1);
    image = cv::imread(file_with_path, 1);
    if (image.elemSize() == 0) {
        BASIC_ASSERT(0);
    }

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    printf("h = %d\n", gray.size().height);
    printf("w = %d\n", gray.size().width);
    printf("total = %lu\n", gray.total());
    printf("elemSize = %lu\n", gray.elemSize());
    
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            printf("%d,", gray.at<uint8_t>(i, j));
        }
    }
    printf("\n");
}





void app_crop_resize_impl(const char * file_in, const char * file_out)
{
    //char filename[128];
    //syspc_file_search(file_with_path, filename, ARRAY_LEN(filename));

    cv::Mat img = cv::imread(file_in, 0);
    if (img.elemSize() == 0) {
        BASIC_ASSERT(0);
    }

    cv::Mat img2, img3;

    img(Rect(0,3,160,117)).copyTo(img2);

    cv::resize(img2, img3, cv::Size(160, 120), 0, 0, cv::INTER_LANCZOS4);

    //cv::imshow("img",img3);
    //cv::waitKey(0);

    cv::imwrite(file_out, img3);
}

void app_crop_resize()
{
    const char * f1  = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126973.286668160.jpg";
    const char * f1o = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126973.286668160_new.jpg";
    const char * f2  = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2b/0/145800.158732006.png";
    const char * f2o = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2b/0/145800.158732006_new.png";
    const char * f3  = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2c/0/146391.794837788.jpg";
    const char * f3o = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2c/0/146391.794837788_new.jpg";
    const char * f4  = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126975.914523917.jpg";
    const char * f4o = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126975.914523917_new.jpg";
    const char * f5  = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126974.718584326.jpg";
    const char * f5o = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126974.718584326_new.jpg";
    const char * f6  = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126976.290559499.jpg";
    const char * f6o = "/home/pupil/temp/p1/pupil_p1/pupil_data/pieces/func2/data/data_func2a/0/126976.290559499_new.jpg";

    app_crop_resize_impl(f1, f1o);
    app_crop_resize_impl(f2, f2o);
    app_crop_resize_impl(f3, f3o);
    app_crop_resize_impl(f4, f4o);
    app_crop_resize_impl(f5, f5o);
    app_crop_resize_impl(f6, f6o);
}



int main(int argc, char** argv)
{
    argc = argc;
    argv = argv;
    
    //app_fitEllipse(argc, argv);
    ocv_util();

    // app_contour(argc, argv);
    //app_image_to_c_array("pupil_data/pieces/func2/data/data_func2a/0/126973.286668160.jpg");
    // app_crop_resize();
    return 0;
}