#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sstream>


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


vector<pair<int,int> > GenFeature(Mat img,int blockSize = 2, int apertureSize = 3, double k = 0.04, int nonMaxRadius = 3, int upperLimitNormalization = 5000){
    // se asume que img esta en escala de grises
    Mat dst = Mat::zeros( img.size(), CV_32FC1 );
    Mat dst_norm, dst_norm_scaled;
    cornerHarris( img, dst, blockSize, apertureSize, k, BORDER_DEFAULT );  
    normalize( dst, dst_norm, 0, upperLimitNormalization, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( img, dst_norm_scaled );
    //seleccionando el Threshold adaptativamente
    int histogram[upperLimitNormalization + 1];
    for(int i = 0; i < upperLimitNormalization + 1; i++)
        histogram[i] = 0;
    for( int i = 0; i < dst_norm.rows ; i++ )
        for( int j = 0; j < dst_norm.cols; j++ )
            histogram[(int)dst_norm.at<float>(i,j)]++;
    int cumulativesum = 0;
    int total = dst_norm.rows * dst_norm.cols;
    int aim = total / 20;
    int thresh = upperLimitNormalization / 3;
    for(int i = 0; i < upperLimitNormalization; i++){
        cumulativesum += histogram[i];
        if ((total - cumulativesum) <= aim){
            thresh = i + 1;
            break;
        }
    }
    vector<pair<int,int> > features;    
    for( int i = 0; i < dst_norm.rows ; i++ ){
        for( int j = 0; j < dst_norm.cols; j++ ){
            if ( NonMaxSupression(dst_norm, i, j, nonMaxRadius) && ( ((int)dst_norm.at<float>(i,j)) >= thresh) ) {
                features.push_back(make_pair(i,j));
                circle( dst_norm_scaled, Point( j, i ), 5,  Scalar(0), 2, 8, 0 ); //
            }
        }
    }
    db = dst_norm_scaled;
    return features;
}

string convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int main(){//int argc, char** argv){
    Mat image, image_gray;
    /*if( argc != 2){
     cout <<"Pasar el nombre de la imagen" << endl;
     return -1;
    }*/
    //namedWindow("ventana",1);//
    
    for(int i = 0; i <= 1; ++i){
          string frame = "I1_000";//convertInt(0);
          string num = convertInt(i);
          if(num.size() == 1) num = "00"+num;
          else if(num.size() == 2) num = "0"+num; 
          frame = frame+num+".jpg";    
          //cout<<frame<<endl;
          image = imread(frame);//, CV_LOAD_IMAGE_GRAYSCALE);//, CV_LOAD_IMAGE_COLOR);
          //cvtColor( image, image_gray, CV_BGR2GRAY );
          vector<pair<int,int> > fts = GenFeature(image_gray);
          //namedWindow("edges",1);
          imwrite(num+".jpg", db);
          //cout<<fts.size()<<endl;
          //waitKey(0);
    }
    return 0;
}
