#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
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

#include "MonoVisoParameters.h"

VisoMonoParam GLOBAL_PARAMETERS;

#include "Calibration.h"
#include "Normalizing.h"
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
          Point p1(feature1.second, feature1.first + img1.rows);
          Point p2(feature2.second, feature2.first);
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

void debugging2(Mat img1, Mat img2,  vector<pair<int,int> > &fts1,  vector<pair<int,int> > &fts2, vector< pair<int,int> > &correspondences){
     Mat new_image;
     new_image.create(img2.rows, img2.cols, img2.type());
     for(int i = 0; i < img2.rows; i++)
        for(int j = 0; j < img2.cols; j++)
            new_image.at<int>(i,j) = img2.at<int>(i, j);
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
          if (i % 20 == 0)
            imwrite("new_image"+num+".jpg", new_image);
          //imshow("correspondences", new_image);
          //waitKey(0);     
     }
     imwrite("new_image.jpg", new_image);
     return;
}

void Debug(vector<DMatch> &matches, Mat &I1, Mat &I2, vector<KeyPoint> &k1, vector<KeyPoint> &k2){
    Mat new_image = I2.clone();
    for(int i = 0; i < (int)matches.size(); i++){
        int idx1 = matches[i].queryIdx;
        int idx2 = matches[i].trainIdx;
        Point2f p1 = k1[idx1].pt;
        Point2f p2 = k2[idx2].pt;
        Scalar color1(25.0);
        Scalar color2(3.0);
        circle(new_image, p1, 1, color1, 1);
        circle(new_image, p2, 3, color2, 2);
        line(new_image, p1, p2, color1);
    }
    namedWindow("correspondences",1);
    imshow("correspondences", new_image);
    waitKey(0);
}

void Debug2(vector<DMatch> &matches, Mat &I1, Mat &I2, vector<KeyPoint> &k1, vector<KeyPoint> &k2, vector<int> &idx){
    Mat new_image = I2.clone();
    for(int i = 0; i < (int)idx.size(); i++){
        int idx1 = matches[idx[i]].queryIdx;
        int idx2 = matches[idx[i]].trainIdx;
        Point2f p1 = k1[idx1].pt;
        Point2f p2 = k2[idx2].pt;
        Scalar color1(25.0);
        Scalar color2(3.0);
        circle(new_image, p1, 1, color1, 1);
        circle(new_image, p2, 3, color2, 2);
        line(new_image, p1, p2, color1);
    }
    namedWindow("correspondences",1);
    imshow("correspondences", new_image);
    waitKey(0);
}

Mat PreviousImageGrayScale;
vector<KeyPoint> PreviousFeatures;
Mat PreviousFeatureDescriptors;
Mat Pose;
SurfFeatureDetector SurfDetector(300);
SurfDescriptorExtractor SurfDescriptor;
BFMatcher matcher(NORM_L2, true);


void init(Mat img){
    Pose = Mat::eye(4, 4, CV_64F);
    PreviousImageGrayScale = img;
    PreviousFeatures.clear();
    SurfDetector.detect(img, PreviousFeatures);
    SurfDescriptor.compute(img, PreviousFeatures, PreviousFeatureDescriptors);
}


bool compute(Mat CurrentImageGrayScale, Mat Kinverse, const int iteration){
    vector<KeyPoint> CurrentFeatures;
    SurfDetector.detect(CurrentImageGrayScale, CurrentFeatures);
    Mat CurrentFeatureDescriptors;
    SurfDescriptor.compute(CurrentImageGrayScale, CurrentFeatures, CurrentFeatureDescriptors);
    vector<DMatch> matches;
    matcher.match(PreviousFeatureDescriptors, CurrentFeatureDescriptors, matches);
    if (matches.size() > 200){
        nth_element(matches.begin(), matches.begin()+ 200, matches.end());
        matches.erase(matches.begin() + 201, matches.end());
    }
    //Debug(matches, PreviousImageGrayScale, CurrentImageGrayScale, PreviousFeatures, CurrentFeatures);
    vector< pair<double,double> > FirstImageFeatures;
    vector< pair<double,double> > SecondImageFeatures;
    for(int i  = 0; i < matches.size(); i++){
        Point2f myft = PreviousFeatures[matches[i].queryIdx].pt;
        Mat FtMatForm = (Mat_<double>(3,1) << (double)myft.x, (double)myft.y, 1.0);
        FtMatForm = Kinverse*FtMatForm;       
        pair<double,double> tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
        FirstImageFeatures.push_back(tmp);
        
        myft = CurrentFeatures[matches[i].trainIdx].pt;
        FtMatForm = (Mat_<double>(3,1) << (double)myft.x, (double)myft.y, 1.0);
        FtMatForm = Kinverse*FtMatForm;       
        tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
        SecondImageFeatures.push_back(tmp);
    }
    vector<int> inliers_indexes;
    Mat RobustEssentialMatrix= Ransac(FirstImageFeatures, SecondImageFeatures, 0.00001, 8, 2000, inliers_indexes);
    //cout << RobustEssentialMatrix << endl;
    
    //Debug2(matches, PreviousImageGrayScale, CurrentImageGrayScale, PreviousFeatures, CurrentFeatures, inliers_indexes);
    
    Mat P = Mat::eye(3,4,CV_64F);
    if (!GetRotationAndTraslation(RobustEssentialMatrix, FirstImageFeatures, SecondImageFeatures, inliers_indexes, P)){
        cerr << "Recovering Translation and Rotation: Failed" << endl;
        return false;
    }
    //cout << P << endl;
    Mat Transformation = Mat::zeros(4,4, CV_64F);
    Transformation.at<double>(3,3) = 1.0;
    for(int i = 0 ; i < 3; i++)
        for(int j = 0; j < 4; j++)
            Transformation.at<double>(i, j) = P.at<double>(i, j);
    Mat TransformationInverse = Transformation.inv();
    Pose = Pose * TransformationInverse;
    cerr << Pose.at<double>(0, 3) << " " << Pose.at<double>(1, 3) << " " << Pose.at<double>(2, 3) << endl;    
    
    PreviousImageGrayScale = CurrentImageGrayScale;
    PreviousFeatures = CurrentFeatures;
    PreviousFeatureDescriptors = CurrentFeatureDescriptors;
    
    //viejo
    
//    vector< pair<int,int> > correspondences = harrisFeatureMatcherMCC(PreviousImageGrayScale, CurrentImageGrayScale, PreviousFeatures, CurrentFeatures);
//    cout << "Iteracion" << iteration << "Cantidad de correspondencias " << correspondences.size() << endl;
//    vector< pair<double,double> > FirstImageFeatures;
//    vector< pair<double,double> > SecondImageFeatures;
//    for(int i  = 0; i < correspondences.size(); i++){
//        pair<int,int> myft = PreviousFeatures[correspondences[i].first];
//        Mat FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
//        FtMatForm = Kinverse*FtMatForm;       
//        pair<double,double> tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
//        FirstImageFeatures.push_back(tmp);
//        
//        myft = CurrentFeatures[correspondences[i].second];
//        FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
//        FtMatForm = Kinverse*FtMatForm;       
//        tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
//        SecondImageFeatures.push_back(tmp);
//    }
//    vector<int> inliers_indexes;
//    Mat RobustEssentialMatrix= Ransac(FirstImageFeatures, SecondImageFeatures, 0.98, 0.00001, 0.5, 8, FirstImageFeatures.size()/2, inliers_indexes);
//    cout << "Iteration" << iteration << "Final EssentialMatrix" << endl;
//    cout << RobustEssentialMatrix << endl;
//    
//    
//    vector<pair<int, int> > correspondences_inliers;
//    for(int i = 0; i < inliers_indexes.size(); i++)
//        correspondences_inliers.push_back(correspondences[inliers_indexes[i]]);
//    debugging2(PreviousImageGrayScale, CurrentImageGrayScale, PreviousFeatures, CurrentFeatures, correspondences_inliers);
//    
//    Mat P = Mat::eye(3,4,CV_64F);
//    if (!GetRotationAndTraslation(RobustEssentialMatrix, FirstImageFeatures, SecondImageFeatures, inliers_indexes, P))
//        return false;
//    cout << "Iteration" << iteration << "Camera Matrix" << endl;
//    cout << P << endl;
//    Mat Transformation = Mat::zeros(4,4, CV_64F);
//    Transformation.at<double>(3,3) = 1.0;
//    for(int i = 0 ; i < 3; i++)
//        for(int j = 0; j < 4; j++)
//            Transformation.at<double>(i, j) = P.at<double>(i, j);
//    Mat TransformationInverse = Transformation.inv();
//    Pose = Pose * TransformationInverse;
//    PreviousImageGrayScale = CurrentImageGrayScale;
//    PreviousFeatures = CurrentFeatures;
//    cerr << Pose.at<double>(0, 4) << Pose.at<double>(1, 4) << Pose.at<double>(2, 4) << endl;

}


int mainMain(int argc, char** argv){
    if (argc != 3){
        cout << "Pasar el path a las imagenes y el numero de estas" << endl;
        return 1;
    }
    String path = argv[1];
    String file = path + "/I1_000000.png";
    Mat Img, Imggray;
    Img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
    //cvtColor( Img, Imggray, CV_BGR2GRAY);
    init(Img);
    int total;
    sscanf(argv[2],"%d",&total);
    Mat Kinverse = GetInverseCalibrationMatrix();
    for(int i = 1; i <= total; i++){
        cout << "fotogram: " << i << endl;
        char cadena[60];
        sprintf(cadena, "%s/I1_%06d.png", argv[1], i);
        file = cadena;
        Img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
        //cvtColor( Img, Imggray, CV_BGR2GRAY);
        compute(Img, Kinverse, i);    // remember to handle the return value of this function
    }
    cout << "OK" << endl;
    return 0;
}

int main(){
    /* Entrada por archivo */
    int nc;
    Pose = Mat::eye(4, 4, CV_64F);
    while(scanf("%d", &nc) != EOF){
        vector<pair<int, int> > v1, v2;
        for(int i = 0; i < nc; i++){
            int a,b;
            scanf("%d %d", &a, &b);
            pair<int, int> p;
            p.first = a;
            p.second = b;
            v1.push_back(p);
        }
        for(int i = 0; i < nc; i++){
            int c,d;
            scanf("%d %d", &c, &d);
            pair<int, int> q;
            q.first = c;
            q.second = d;
            v2.push_back(q);
        }
        
    
//    vector<pair<int, int> > c;
//    for(int i = 0; i < nc; i++)
//        c.push_back(make_pair(i,i));
//        
//    String file = "Fotos/I1_000000.png";    
//    Mat Imgp = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//    file = "Fotos/I1_000001.png";   
//    Mat Imgc = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//    
//    debugging2(Imgp, Imgc, v1, v2, c);
        double fx = 645.24;
        double fy = 645.24;
        double cx = 635.96;
        double cy = 194.13;              
        Mat K = (Mat_<double>(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        Mat Kinverse = K.inv();
        //Mat K = GetCalibrationMatrix();
        vector< pair<double,double> > FirstImageFeatures;
        vector< pair<double,double> > SecondImageFeatures;
        for(int i  = 0; i < v1.size(); i++){
            pair<int,int> myft = v1[i];
            Mat FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
            //FtMatForm = Kinverse*FtMatForm;      
            pair<double,double> tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
            FirstImageFeatures.push_back(tmp);
            
            myft = v2[i];
            FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
            //FtMatForm = Kinverse*FtMatForm;       
            tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
            SecondImageFeatures.push_back(tmp);
        }
        vector<int> inliers_indexes;
        Mat RobustEssentialMatrix= Ransac(FirstImageFeatures, SecondImageFeatures, 0.00001, 8, 2000, inliers_indexes);
        /*
        RobustEssentialMatrix = K.t() * RobustEssentialMatrix * K;
        Mat w, u, vt;
        SVD::compute(RobustEssentialMatrix, w, u, vt, SVD::FULL_UV);
        w.at<double>(0, 2) = 0.0;
        Mat S = Mat::diag(w);
        RobustEssentialMatrix = u * S * vt;*/
     
//    
//    //Debugging
//    
//    /*
//   double TestE[3][3] = { {-0.0022604,   -5.5544313,   -0.5349702}, 
//                          {5.5467682,    0.0026827,   -0.1939585}, 
//                          {0.5235505,    0.1917261,   0.0001368 } };
//                          
//                          
//    Mat TE = Mat(3, 3, CV_64FC1, TestE);*/
//   

        /*
        Mat P = Mat::eye(3,4,CV_64F);
        GetRotationAndTraslation(RobustEssentialMatrix, FirstImageFeatures, SecondImageFeatures, inliers_indexes, P);
        cout << "Camera Matrix" << endl;
        cout << P << endl;
        Mat Transformation = Mat::zeros(4,4, CV_64F);
        Transformation.at<double>(3,3) = 1.0;
        for(int i = 0 ; i < 3; i++)
            for(int j = 0; j < 4; j++)
                Transformation.at<double>(i, j) = P.at<double>(i, j);
        Mat TransformationInverse = Transformation.inv();
        Pose = Pose * TransformationInverse;
        cout << "Pose" << endl;
        cout << Pose << endl;
        cerr << Pose.at<double>(0, 3) << " " << Pose.at<double>(1, 3) << " " << Pose.at<double>(2, 3) << endl;    */
    }
    return 0;
}


//int main(int argc, char** argv){
//    Mat img1, img2, imgray1, imgray2;
//    if( argc != 3){
//     cout <<"Pasar el nombre de las dos imagenes" << endl;
//     return -1;
//    }
//    init();
//    namedWindow("ventana",1);//
//    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
//    img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
//    cvtColor( img1, imgray1, CV_BGR2GRAY);
//    cvtColor( img2, imgray2, CV_BGR2GRAY);
//    vector<pair<int,int> > fts1 = GenFeature2(imgray1);
//    vector<pair<int,int> > fts2 = GenFeature2(imgray2);
//    cout<<"Cantidad características imagen 1: "<<fts1.size()<<" Cantidad características imagen 2: "<<fts2.size()<<endl;
//    vector< pair<int,int> > correspondences = harrisFeatureMatcherMCC(imgray1, imgray2, fts1, fts2);
//    cout <<"Cantidad de correspondencias " << correspondences.size() << endl;
//    vector< pair<double,double> > FirstImageFeatures;
//    vector< pair<double,double> > SecondImageFeatures;
//    Mat Kinverse = GetInverseCalibrationMatrix();
//    for(int i  = 0; i < correspondences.size(); i++){
//        pair<int,int> myft = fts1[correspondences[i].first];
//        Mat FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
//        FtMatForm = Kinverse*FtMatForm;       
//        pair<double,double> tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
//        FirstImageFeatures.push_back(tmp);
//        
//        myft = fts2[correspondences[i].second];
//        FtMatForm = (Mat_<double>(3,1) << (double)myft.first, (double)myft.second, 1.0);
//        FtMatForm = Kinverse*FtMatForm;       
//        tmp = make_pair(FtMatForm.at<double>(0,0), FtMatForm.at<double>(1,0));
//        SecondImageFeatures.push_back(tmp);
//    }
//    //Debugging
//    vector<int> inliers_indexes;
//    Mat RobustEssentialMatrix= Ransac(FirstImageFeatures, SecondImageFeatures, 0.98, 0.0025, 0.5, 11, FirstImageFeatures.size()/2, inliers_indexes);  // TODO: Estimate experimentally the value of T */
//    cout << "Final EssentialMatrix" << endl;
//    cout << RobustEssentialMatrix << endl;
//    Mat P = Mat::eye(3,4,CV_64F);
//    GetRotationAndTraslation(RobustEssentialMatrix, FirstImageFeatures, SecondImageFeatures, inliers_indexes, P);
//    cout << "Camera Matrix" << endl;
//    cout << P << endl;
//    Mat Pose
//    
//    /*
//    cout << "inliers size" << inliers_indexes.size() << endl;
//    vector< pair<int,int> > inlier_correspondences;
//    for(int i = 0; i < inliers_indexes.size(); i++)
//        inlier_correspondences.push_back(correspondences[inliers_indexes[i]]);
//    debugging2(imgray1, imgray2, fts1, fts2, inlier_correspondences);*/
//    return 0;
//}
