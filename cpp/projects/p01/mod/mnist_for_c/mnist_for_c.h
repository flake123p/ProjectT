/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "basic.h"

#ifdef __cplusplus
extern "C" {
#endif

// set appropriate path for data
#define TRAIN_IMAGE "/home/john/ws/mnist/MNIST_for_C/data/train-images.idx3-ubyte"
#define TRAIN_LABEL "/home/john/ws/mnist/MNIST_for_C/data/train-labels.idx1-ubyte"
#define TEST_IMAGE  "/home/john/ws/mnist/MNIST_for_C/data/t10k-images.idx3-ubyte"
#define TEST_LABEL  "/home/john/ws/mnist/MNIST_for_C/data/t10k-labels.idx1-ubyte"

#define SIZE784 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

extern unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
extern int width[MAX_NUM_OF_IMAGES];
extern int height[MAX_NUM_OF_IMAGES];

extern int info_image[LEN_INFO_IMAGE];
extern int info_label[LEN_INFO_LABEL];

extern unsigned char train_image_char[NUM_TRAIN][SIZE784];
extern unsigned char test_image_char[NUM_TEST][SIZE784];
extern unsigned char train_label_char[NUM_TRAIN][1];
extern unsigned char test_label_char[NUM_TEST][1];

extern double train_image[NUM_TRAIN][SIZE784];
extern double test_image[NUM_TEST][SIZE784];
extern int  train_label[NUM_TRAIN];
extern int test_label[NUM_TEST];


void FlipLong(unsigned char * ptr);


void read_mnist_image_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][SIZE784], int info_arr[]);
void read_mnist_label_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][1], int info_arr[]);


void image_char2double(int num_data, unsigned char data_image_char[][SIZE784], double data_image[][SIZE784]);


void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[]);

void load_mnist();

void print_mnist_pixel(double data_image[][SIZE784], int num_data);
void print_mnist_label(int data_label[], int num_data);

// name: path for saving image (ex: "./images/sample.pgm")
void save_image(int n, char name[]);

// save mnist image (call for each image)
// store train_image[][] into image[][][]
void save_mnist_pgm(double data_image[][SIZE784], int index);
int mnist_for_c_example(void);

#ifdef __cplusplus
};//extern "C"
#endif