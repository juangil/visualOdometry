#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


int thresh = 200;

using namespace cv;
using namespace std;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    Mat edges;
    Mat frame;
    cap >> frame; // get a new frame from camera
    namedWindow("edges",1);
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( frame.size(), CV_32FC1 );
    for(;;)
    {
        //Mat frame;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, edges, CV_BGR2GRAY);
        
        
        
        //Harris
        

        /// Detector parameters
        int blockSize = 2;
        int apertureSize = 3;
        double k = 0.04;

        /// Detecting corners
        cornerHarris( edges, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
        //cout << dst<< endl;
       
        /// Normalizing
        normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
        convertScaleAbs( dst_norm, dst_norm_scaled );

        /// Drawing a circle around corners
        for( int j = 0; j < dst_norm.rows ; j++ )
         { for( int i = 0; i < dst_norm.cols; i++ )
              {
                if( (int) dst_norm.at<float>(j,i) > thresh )
                  {
                   circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                  }
              }
         }
        imshow("edges", dst_norm_scaled);
        if(waitKey(100) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
