#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>



using namespace std;
using namespace cv;



Mat db;    

bool NonMaxSupression(Mat m, int x, int y, int nonMaxRadius){
    for(int i = -nonMaxRadius;  i <= nonMaxRadius; i++)
        for(int j = -nonMaxRadius; j <= nonMaxRadius; j++){
            int nx = x + i;
            int ny = y + j;
            if (nx < 0 || ny < 0 || nx >= m.rows || ny >= m.cols || (nx == x && ny == y))
                continue;
            if (m.at<float>(nx,ny) > m.at<float>(x,y))
                return false;
        }
    return true;
}


vector<pair<int,int> > GenFeature(Mat img,int blockSize = 2, int apertureSize = 3, double k = 0.06, int nonMaxRadius = 3, int thresh = 80.0){
    // se asume que img esta en escala de grises
    Mat dst = Mat::zeros( img.size(), CV_32FC1 );
    Mat dst_norm, dst_norm_scaled;
    cornerHarris( img, dst, blockSize, apertureSize, k, BORDER_DEFAULT );  
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    vector<pair<int,int> > features;    
    for( int i = 0; i < dst_norm.rows ; i++ ){
        for( int j = 0; j < dst_norm.cols; j++ ){
            if ( NonMaxSupression(dst_norm, i, j, nonMaxRadius) && ( dst_norm.at<float>(i,j) > thresh) ) {
                features.push_back(make_pair(i,j));
                circle( dst_norm_scaled, Point( j, i ), 5,  Scalar(0), 2, 8, 0 ); //
            }
        }
    }
    db = dst_norm_scaled;
    return features;
}

int main(int argc, char** argv){
    Mat image, image_gray;
    if( argc != 2){
     cout <<"Pasar el nombre de la imagen" << endl;
     return -1;
    }
    namedWindow("ventana",1);//
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cvtColor( image, image_gray, CV_BGR2GRAY );
    vector<pair<int,int> > fts = GenFeature(image_gray);
    namedWindow("edges",1);
    imshow("edges", db);
    cout<<fts.size()<<endl;
    waitKey(0);
    //for(int i = 0; i < fts.size(); ++i) cout<< fts[i] << end;
    return 0;
}
