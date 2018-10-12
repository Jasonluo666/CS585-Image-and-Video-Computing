#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void createGrayScale(Mat& image, string location = "") {

	// create an Mat object for grayscale image
	Mat gray_image;
	cv::cvtColor(image, gray_image, CV_BGR2GRAY);
	
	for (int row = 0; row < image.rows; row++) {
		unsigned char* rowPtr_origin = image.ptr<unsigned char>(row);
		unsigned char* rowPtr_grayscale = gray_image.ptr<unsigned char>(row);

		for (int col = 0; col < image.cols; col++) {
			int B_index = col * image.channels() + 0,
				G_index = col * image.channels() + 1,
				R_index = col * image.channels() + 2;	// BGR channel = 3

			// Method 3: Weigh them: 0.3*R + 0.6*G + 0.1*B
			rowPtr_grayscale[col] = 0.3 * rowPtr_origin[R_index] + 0.6 * rowPtr_origin[G_index] + 0.1 * rowPtr_origin[B_index];
		}
	}

	imwrite(location, gray_image);	// save the image
}

void horizontallyFlip(Mat& image, string location = "") {

	// create an Mat object for flipped image
	Mat flipped_image;
	// flipCode = 0 -> vertical
	// flipCode = 1 -> horizontal
	//cv::flip(image, flipped_image, 1);

	image.copyTo(flipped_image);

	for (int row = 0; row < image.rows; row++) {
		unsigned char* rowPtr_origin = image.ptr<unsigned char>(row);
		unsigned char* rowPtr_flipped = flipped_image.ptr<unsigned char>(row);

		for (int col = 0; col < image.cols; col++) {
			int original_B_index = col * image.channels() + 0,
				flipped_B_index = (image.cols - 1 - col) * image.channels() + 0;	// BGR channel = 3

			for (int color_index = 0; color_index < 3; color_index++)
				rowPtr_flipped[flipped_B_index + color_index] = rowPtr_origin[original_B_index + color_index];
		}
	}

	imwrite(location, flipped_image);	// save the image
}

void blurredImage(Mat& gray_image, string location = "", int loop_time = 1) {

	// create an Mat object for blurred image
	Mat preprocessed_image, blurred_image;
	gray_image.copyTo(preprocessed_image);
	gray_image.copyTo(blurred_image);
	int neighbors[3] = { 1, 0, -1 };

	while (loop_time--) {
		for (int row = 0; row < preprocessed_image.rows; row++) {
			unsigned char* rowPtr = blurred_image.ptr<unsigned char>(row);

			for (int col = 0; col < preprocessed_image.cols; col++) {
				// find neighbors
				unsigned char* rowPtr_temporary;
				int count = 0, value = 0;
				for (int x_index = 0; x_index < 3; x_index++)
					for (int y_index = 0; y_index < 3; y_index++) {
						// continue if out of range
						if ((neighbors[x_index] == 0 && neighbors[y_index] == 0) || row + neighbors[x_index] < 0 ||
							row + neighbors[x_index] >= preprocessed_image.rows || col + neighbors[y_index] < 0 ||
							col + neighbors[y_index] >= preprocessed_image.cols)
							continue;

						rowPtr_temporary = preprocessed_image.ptr<unsigned char>(row + neighbors[x_index]);
						value += rowPtr_temporary[col + neighbors[y_index]];
						count++;
					}

				if (count != 0)
					rowPtr[col] = value / count;
			}
		}
		// prepare for the next iteration
		blurred_image.copyTo(preprocessed_image);
	}

	imwrite(location, blurred_image);	// save the image
}

int main()
{
	Mat image;
	image = imread("my_picture.jpg", IMREAD_COLOR); // Read the file

	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return 0;
	}

	// grayscale image
	createGrayScale(image, "./grayscale.jpg");
	// flipped image
	horizontallyFlip(image, "./flipped.jpg");

	// blurred image -> 10 times iterations
	Mat grayscale_image = imread("grayscale.jpg", cv::IMREAD_GRAYSCALE); // Read the file
	blurredImage(grayscale_image, "./blurred.jpg", 10);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}