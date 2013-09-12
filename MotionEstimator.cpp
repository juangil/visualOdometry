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


using namespace std;
using namespace cv;


void MotionFromEightPointsAlgorithm(vector<pair<double, double> > v1, vector<pair<double, double> > v2){
    if (v1.size() != v2.size()){
        fprintf(stderr, "Los tamanos de los vectores de correspondencias son diferentes");
        return ;
    }
    int ncorrespondences = v1.size();
    Mat A = Mat::zeros(ncorrespondences, 9, CV_64FC1);
    Mat w, u, vt;
    for(int i = 0; i < ncorrespondences; i++){
        double u = v1[i].first;
        double v = v1[i].second;
        double up = v2[i].first;
        double vp = v2[i].second;
        A.at<double>(i, 0) = u*up;
        A.at<double>(i, 1) = u*vp; 
        A.at<double>(i, 2) = u;
        A.at<double>(i, 3) = up*v;
        A.at<double>(i, 4) = v*vp;
        A.at<double>(i, 5) = v;
        A.at<double>(i, 6) = up;
        A.at<double>(i, 7) = vp;
        A.at<double>(i, 8) = 1.0;
    }
    SVD::compute(A, w, u, vt, SVD::FULL_UV);
    // Solving the homogeneous linear system in a least squares sense
//    cout<<"A =="<<endl;
//    cout<< A <<endl;
//    cout<<"w =="<<endl;
//    cout<< w <<endl;
//    cout<<"u =="<<endl;
//    cout<< u <<endl;
//    cout<<"vt =="<<endl;
//    cout<< vt <<endl;
//    cout<<"===="<<endl;
    Mat sol = vt.row(8);
    cout<<sol.rows<<" "<<sol.cols<<endl;
    double tmpE[3][3] = { {sol.at<double>(0,0), sol.at<double>(0,1), sol.at<double>(0,2)},
                         {sol.at<double>(0,3), sol.at<double>(0,4), sol.at<double>(0,5)},
                         {sol.at<double>(0,6), sol.at<double>(0,7), sol.at<double>(0,8)} };
    Mat EssentialMatrix = Mat(3, 3, CV_64FC1, tmpE);
    //Recovering R,t
    SVD::compute(EssentialMatrix, w, u, vt, SVD::FULL_UV);
//    cout<<"E =="<<endl;
//    cout<< EssentialMatrix <<endl;
//    cout<<"w =="<<endl;
//    cout<< w <<endl;
//    cout<<"u =="<<endl;
//    cout<< u <<endl;
//    cout<<"vt =="<<endl;
//    cout<< vt <<endl;
//    cout<<"===="<<endl;
    // Enforcing  the Internal Constraint 
    w.at<double>(0, 2) = 0.0;
//    cout<<"w =="<<endl;
//    cout<< w <<endl;
    double tmpW[3][3] = { {0,1,0},
                         {-1,0,0},
                         {0,0,1} };
    Mat W = Mat(3, 3, CV_64FC1, tmpW);
    Mat S = Mat::diag(w);
    Mat Rot1 = u * W * vt;   //These matrixes should be rotation matrixes.
    Mat Rot2 = u * W.t() * vt; 
    Mat T = u * W * S * u.t(); // This matrix should be a skew matrix.
    //T = T * (1 / T.at<double>(2,1)); //This is for "normalizing" the traslation vector such that Tx = 1.0
    Mat T1 = (Mat_<double>(3,1) << T.at<double>(2,1), T.at<double>(0,2), T.at<double>(1,0));
    Mat T2 = -1*T1;
    cout<<"R1:"<<endl;
    cout<<Rot1<<endl;
    cout<<"R2:"<<endl;
    cout<<Rot2<<endl;
    cout<<"T1:"<<endl;
    cout<<T1<<endl;
    cout<<"T2:"<<endl;
    cout<<T2<<endl;
}


int main(){
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
    }
    MotionFromEightPointsAlgorithm( v1, v2);
    return 0;
}
