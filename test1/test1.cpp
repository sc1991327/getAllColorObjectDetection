#include "stdafx.h"

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// find all connect components
void FindBlobs(const Mat &binary, vector < vector<Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            Rect rect;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

            vector <Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

int main( int argc, char** argv )
{
	//capture the video from web cam
    VideoCapture cap(0); 
    if ( !cap.isOpened() )  
    {
		// if not success, exit program
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

	// ------ Control Window ------
	// Use to control the ROI Color.

	//create a window called "Control"
    namedWindow("Control",CV_WINDOW_AUTOSIZE);

	// H
	int iLowH = 38;
	int iHighH = 75;
	// S
	int iLowS = 90; 
	int iHighS = 150;
	// V
	int iLowV = 130;
	int iHighV = 250;
	
	// Area Size Threshold
	int iAreaSize = 10000;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 255);		//Hue (0 - 255)
	cvCreateTrackbar("HighH", "Control", &iHighH, 255);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255);		//Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255);		//Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	cvCreateTrackbar("AreaSizeThreshold", "Control", &iAreaSize, 100000);//Value (0 - 100000)

	// ------ Original Image Window / Result Image Window ------
	// Use to show original/result image.

    while (true){
		// read a new RGB frame from video
        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal); 
        if (!bSuccess){
			//if not success, break loop
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

		//Create a black image with the size as the camera output
		Mat imgFinger = Mat::zeros( imgOriginal.size(), CV_8UC3 );

		//Convert the captured frame from BGR to HSV
		Mat imgHSV;
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
	
		//Threshold the image
		Mat imgThresholded;
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

		//morphological opening (removes small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		//morphological closing (removes small holes from the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

		//Calculate the moments of the thresholded image
		Moments oMoments = moments(imgThresholded);
		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;
		if(dArea > iAreaSize){
			//calculate the position of the ball
			int posX = dM10 / dArea;
			int posY = dM01 / dArea;
			if (posX >= 0 && posY >= 0){
				//Draw a red line from the previous point to the current point
				circle(imgFinger, Point(posX, posY), 5 , Scalar(0,0,255), CV_FILLED);
			}
		}

		imgOriginal = imgOriginal + imgFinger;
		imshow("Thresholded Image", imgThresholded);	//show the thresholded image
		imshow("Original", imgOriginal);				//show the original image

		//wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        if (waitKey(30) == 27){
            cout << "esc key is pressed by user" << endl;
            break; 
		}
    }
	return 0;
}