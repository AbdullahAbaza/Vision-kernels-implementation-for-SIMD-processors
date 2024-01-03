#include <opencv2/highgui/highgui.hpp>
#include <immintrin.h> // for AVX-2 intrinsics
#include <iostream>

using namespace cv;

// A function that multiplies two 8-bit gray scale images using AVX-2 intrinsics
void multiply_gray_scale_image_with_mask(const cv::Mat &img, const cv::Mat &mask, cv::Mat &result)
{

    if (img.rows != mask.rows || img.cols != mask.cols)
    {
        throw std::runtime_error("Input image and mask image have different dimensions.");
    }

    // Get the number of rows and columns of the images
    const int rows = img.rows;
    const int cols = img.cols;

    // Create a result matrix with the same size and type as the input images
    result.create(rows, cols, CV_8UC1);

    // Loop over each row of the images
    for (int i = 0; i < rows; i++)
    {
        // Get pointers to the data of each row
        const uchar *img_data = img.ptr<uchar>(i);
        const uchar *mask_data = mask.ptr<uchar>(i);
        uchar *result_data = result.ptr<uchar>(i);

        // Loop over each column of the images with a step of 32 bytes (256 bits)
        for (int j = 0; j < cols; j += 32)
        {
            // Load 32 bytes from each image into 256-bit registers
            __m256i img_vec = _mm256_loadu_si256((__m256i *)(img_data + j));
            __m256i mask_vec = _mm256_loadu_si256((__m256i *)(mask_data + j));

            // Multiply each pair of bytes using unsigned saturation arithmetic
            __m256i result_vec = _mm256_mullo_epi16(img_vec, mask_vec);

            // Store 32 bytes from the result register into the result image
            _mm256_storeu_si256((__m256i *)(result_data + j), result_vec);
        }
    }
}

// A main function that tests the above function with an example image and mask
int main(int argc, char **argv)
{
    // Read an example gray scale image from a file
    cv::Mat img = cv::imread("C:/Users/black/Downloads/1000 ML/FaceID/OpenCV c++/Vision_Kernel_AVX-2/test.jpg", cv::IMREAD_GRAYSCALE);

    if (img.empty())
    { // check if image is valid
        std::cerr << "Could not read image\n";
        return -1;
    }

    // Create an example mask with half intensity (128)
    cv::Mat mask = cv::Mat(img.size(), img.type(), cv::Scalar(256));
    // cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(128));

    // Create an empty matrix to store the result
    cv::Mat result;

    // Call the function to multiply the image with the mask
    multiply_gray_scale_image_with_mask(img, mask, result);

    cv::imshow("Input Image", img);     // display input image
    cv::imshow("Mask Image", mask);     // display mask image
    cv::imshow("Result Image", result); // display result image

    cv::imwrite("input.jpg", img);     // save input image as input.jpg
    cv::imwrite("mask.jpg", mask);     // save mask image as mask.jpg
    cv::imwrite("result.jpg", result); // save result image as result.jpg

    int key = cv::waitKey(0); // wait for any key press
    if (key == 27)
    {                            // if ESC is pressed
        cv::destroyAllWindows(); // close all windows
    }

    return 0;
}