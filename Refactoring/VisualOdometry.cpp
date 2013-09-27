#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;


#include "Calibration.h"
#include "Normalizing.h"
#include "FeatureDetector.h"
#include "Matching.h"
#include "MotionEstimator.h"
#include "RobustEstimator.h"
#include "disambiguating.h"


string toString(int a){
     stringstream ss;
     ss << a;
     return ss.str();
}


void debugging(Mat img1, Mat img2,  vector<pair<int,int> > &fts1,  vector<pair<int,int> > &fts2, vector< pair<int,int> > correspondences){
     Mat new_image;
     new_image.create(img1.rows *2, img1.cols, img1.type());
     for(int i = 0; i < img1.rows; i++){
        for(int j = 0; j < img1.cols; j++)
            new_image.at<int>(i,j) = img2.at<int>(i, j);
     }
     for(int i = img1.rows; i < img1.rows*2; i++){
        for(int j = 0; j < img1.cols; j++)
            new_image.at<int>(i, j) = img1.at<int>(i - img2.rows, j);
     }
     //namedWindow("correspondences",1);
     for(int i = 0; i < correspondences.size(); i++){
          pair<int,int> feature1 = fts1[correspondences[i].first];
          pair<int,int> feature2 = fts2[correspondences[i].second];
          //Point p1(feature1.second, feature1.first);
          Point p1(feature1.second, feature1.first);
          Point p2(feature2.second, feature2.first + img1.rows);
          Scalar color(25.0);
          circle(new_image, p1, 5, color, 2);
          circle(new_image, p2, 5, color, 2);
          line(new_image, p1, p2, color);
          string num = toString(i);
          imwrite("new_image"+num+".jpg", new_image);
          //imshow("correspondences", new_image);
          //waitKey(0);     
     }
     imwrite("new_image.jpg", new_image);
     return;
}

void debugging2(Mat img1, Mat img2,  vector<pair<int,int> > &fts1,  vector<pair<int,int> > &fts2, vector< pair<int,int> > correspondences){
     Mat new_image;
     new_image.create(img1.rows, img1.cols, img1.type());
     for(int i = 0; i < img1.rows; i++){
        for(int j = 0; j < img1.cols; j++)
            new_image.at<int>(i,j) = img2.at<int>(i, j);
     }
     //namedWindow("correspondences",1);
     for(int i = 0; i < correspondences.size(); i++){
          pair<int,int> feature1 = fts1[correspondences[i].first];
          pair<int,int> feature2 = fts2[correspondences[i].second];
          //Point p1(feature1.second, feature1.first);
          Point p1(feature1.second, feature1.first);
          Point p2(feature2.second, feature2.first);
          Scalar color1(25.0);
          Scalar color2(3.0);
          circle(new_image, p1, 1, color1, 1);
          circle(new_image, p2, 3, color2, 2);
          line(new_image, p1, p2, color1);
          string num = toString(i);
          imwrite("new_image"+num+".jpg", new_image);
          //imshow("correspondences", new_image);
          //waitKey(0);     
     }
     imwrite("new_image.jpg", new_image);
     return;
}


int main(int argc, char** argv){
    Mat img1, img2, imgray1, imgray2;
    if( argc != 3){
     cout <<"Pasar el nombre de las dos imagenes" << endl;
     return -1;
    }
    namedWindow("ventana",1);//
    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    cvtColor( img1, imgray1, CV_BGR2GRAY );
    cvtColor( img2, imgray2, CV_BGR2GRAY );
    vector<pair<int,int> > fts1 = GenFeature2(imgray1);
    vector<pair<int,int> > fts2 = GenFeature2(imgray2);
    cout<<"Cantidad características imagen 1: "<<fts1.size()<<" Cantidad características imagen 2: "<<fts2.size()<<endl;
    vector< pair<int,int> > correspondences = harrisFeatureMatcherMCC(imgray1, imgray2, fts1, fts2);
    cout <<"Cantidad de correspondencias " << correspondences.size() << endl;
    vector< pair<double,double> > FirstImageFeatures;
    vector< pair<double,double> > SecondImageFeatures;
    Mat Kinverse = GetInverseCalibrationMatrix();
    for(int i  = 0; i < correspondences.size(); i++){
        pair<int,int> myft = fts1[correspondences[i].first];
        Mat FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
        FtMatForm = Kinverse*FtMatForm;       
        pair<double,double> tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
        FirstImageFeatures.push_back(tmp);
        
        myft = fts2[correspondences[i].second];
        FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
        FtMatForm = Kinverse*FtMatForm;       
        tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
        SecondImageFeatures.push_back(tmp);
    }
    /* Entrada por archivo
    int nc;
    scanf("%d", &nc);
    vector<pair<double, double> > v1, v2;
    for(int i = 0; i < nc; i++){
        double a,b;
        scanf("%lf %lf", &a, &b);
        pair<double, double> p;
        p.first = a;
        p.second = b;
        v1.push_back(p);
    }
    for(int i = 0; i < nc; i++){
        double c,d;
        scanf("%lf %lf", &c, &d);
        pair<double, double> q;
        q.first = c;
        q.second = d;
        v2.push_back(q);
    }*/
    //Debugging
    vector<int> inliers_indexes;
    Mat RobustEssentialMatrix= Ransac(FirstImageFeatures, SecondImageFeatures, 0.98, 0.001, 0.5, 11, FirstImageFeatures.size()/2, inliers_indexes);  // TODO: Estimate experimentally the value of T */
    cout << "Final EssentialMatrix" << endl;
    cout << RobustEssentialMatrix << endl;
    Mat P = Mat::eye(3,4,CV_64F);
    GetRotationAndTraslation(RobustEssentialMatrix, FirstImageFeatures, SecondImageFeatures, inliers_indexes, P);
    cout << "Camera Matrix" << endl;
    cout << P << endl;
    /*
    cout << "inliers size" << inliers_indexes.size() << endl;
    vector< pair<int,int> > inlier_correspondences;
    for(int i = 0; i < inliers_indexes.size(); i++)
        inlier_correspondences.push_back(correspondences[inliers_indexes[i]]);
    debugging2(imgray1, imgray2, fts1, fts2, inlier_correspondences);*/
    return 0;
}
