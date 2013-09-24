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


void FillingMatrix(Mat &dest, Mat &orig, Mat &orig2){
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            dest.at<double>(i,j) = orig.at<double>(i,j); 
    for(int i = 0; i < 3; i++)
        dest.at<double>(i, 3) = orig2.at<double>(i,0);
    return;
}

Mat TriangulatePoint(const Point2d &pt1, const Point2d &pt2, const Mat &P1, const Mat &P2)
{
    Mat A(4,4,CV_64F);
    Mat w, u, vt;
    
	A.at<double>(0,0) = pt1.x*P1.at<double>(2,0) - P1.at<double>(0,0);
	A.at<double>(0,1) = pt1.x*P1.at<double>(2,1) - P1.at<double>(0,1);
	A.at<double>(0,2) = pt1.x*P1.at<double>(2,2) - P1.at<double>(0,2);
	A.at<double>(0,3) = pt1.x*P1.at<double>(2,3) - P1.at<double>(0,3);

	A.at<double>(1,0) = pt1.y*P1.at<double>(2,0) - P1.at<double>(1,0);
	A.at<double>(1,1) = pt1.y*P1.at<double>(2,1) - P1.at<double>(1,1);
	A.at<double>(1,2) = pt1.y*P1.at<double>(2,2) - P1.at<double>(1,2);
	A.at<double>(1,3) = pt1.y*P1.at<double>(2,3) - P1.at<double>(1,3);

	A.at<double>(2,0) = pt2.x*P2.at<double>(2,0) - P2.at<double>(0,0);
	A.at<double>(2,1) = pt2.x*P2.at<double>(2,1) - P2.at<double>(0,1);
	A.at<double>(2,2) = pt2.x*P2.at<double>(2,2) - P2.at<double>(0,2);
	A.at<double>(2,3) = pt2.x*P2.at<double>(2,3) - P2.at<double>(0,3);

	A.at<double>(3,0) = pt2.y*P2.at<double>(2,0) - P2.at<double>(1,0);
	A.at<double>(3,1) = pt2.y*P2.at<double>(2,1) - P2.at<double>(1,1);
	A.at<double>(3,2) = pt2.y*P2.at<double>(2,2) - P2.at<double>(1,2);
	A.at<double>(3,3) = pt2.y*P2.at<double>(2,3) - P2.at<double>(1,3);

    SVD::compute(A, w, u, vt, SVD::FULL_UV);
	//SVD svd(A); another way to calculate SVD

    Mat X(4,1,CV_64F);

	X.at<double>(0,0) = vt.at<double>(3,0);
	X.at<double>(1,0) = vt.at<double>(3,1);
	X.at<double>(2,0) = vt.at<double>(3,2);
	X.at<double>(3,0) = vt.at<double>(3,3);

    return X;
}


bool InFrontOf(const cv::Mat &X, const cv::Mat &P)
{
    // back project
    cv::Mat X2 = P*X;
    double w = X2.at<double>(2,0);
    double W = X.at<double>(3,0);
    double ans = (w/W);
    return (ans > 0.0);
}


bool IsValidSolution(const vector<pair<double, double> > &v1,const vector<pair<double, double> > &v2, const Mat &P1,  Mat &P2){
    int infront = 0;
    for(int  i = 0; i < v1.size(); i++){
        Point2d pt1 = Point2d(v1[i].first, v1[i].second);
        Point2d pt2 = Point2d(v2[i].first, v2[i].second);
        Mat triangulated = TriangulatePoint(pt1, pt2, P1, P2);
        if (InFrontOf(triangulated, P1) && InFrontOf(triangulated, P2))
            infront++; 
    }
    cout<<"debug " << infront << endl;
    return infront == v1.size();
}





/* Normalization */


pair<double, double> GetCentroid(const vector<pair<double, double> > &v1){
    pair<double, double> ret;
    ret.first = 0;
    ret.second = 0;
    for(int i = 0; i < v1.size(); i++){
        ret.first += v1[i].first;
        ret.second += v1[i].second;
    }
    ret.first = ret.first / v1.size();
    ret.second = ret.second / v1.size();
    return ret;
}

double GetScalingFactor(const vector<pair<double, double> > &v1, pair<double, double> &c){
    double sum = 0.0;
    for(int i = 0; i < v1.size(); i++){
        double x = v1[i].first - c.first;
        double y = v1[i].second - c.second;
        sum += sqrt(x*x + y*y);
    }
    double k = ( sqrt(2) * v1.size()) / sum;
    return k;
}

void Test(const vector<pair<double, double> > &v1){
    cout<<"Testeando normalizacion"<<endl;
    double sum = 0;
    double sumx = 0;
    double sumy = 0;
    for(int i = 0; i < v1.size(); i++){
        double x = v1[i].first;
        double y = v1[i].second;
        sum += sqrt(x*x + y*y);
        sumx += x;
        sumy += y;
    }
    cout<<"centroid :"<< sumx / v1.size() <<" "<< sumy / v1.size() << endl;
    cout<<"mean distance to the origin :"<< sum / v1.size() <<endl;
}

Mat GetUnnormalizedEssentialMatrix(const Mat &E, const Mat &T1, const Mat &T2){
    Mat Ret = T1.t() * E * T2;
    return Ret;
}


Mat GetNormalizingTransformation(const vector<pair<double, double> > &v1){
    pair<double, double> c = GetCentroid(v1);
    double ScalingFactor = GetScalingFactor(v1, c);
    Mat T =  Mat::zeros(3,3,CV_64F);
    T.at<double>(0,0) = ScalingFactor;
    T.at<double>(1,1) = ScalingFactor;
    T.at<double>(0,2) = -c.first * ScalingFactor;
    T.at<double>(1,2) = -c.second * ScalingFactor;
    T.at<double>(2,2) = 1.0;
    return T;
}

//

void MotionFromEightPointsAlgorithm(vector<pair<double, double> > &v1, vector<pair<double, double> > &v2, const Mat &NT1, const Mat &NT2){
    // v1 and v2 represent the correspondences (normalized).
    // NT1 and NT2 are the normalizing transformations.
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
    //cout << "sol dims" <<endl;
    //cout<<sol.rows<<" "<<sol.cols<<endl;
    double tmpE[3][3] = { {sol.at<double>(0,0), sol.at<double>(0,1), sol.at<double>(0,2)},
                          {sol.at<double>(0,3), sol.at<double>(0,4), sol.at<double>(0,5)},
                          {sol.at<double>(0,6), sol.at<double>(0,7), sol.at<double>(0,8)} };
    Mat EssentialMatrix = Mat(3, 3, CV_64FC1, tmpE);
    
    // Testing normalization
    
    EssentialMatrix = GetUnnormalizedEssentialMatrix(EssentialMatrix, NT1, NT2);
    
    
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

    //finding the possible rotation and translation matrices
    double tmpW[3][3] = { {0,1,0},
                          {-1,0,0},
                          {0,0,1} };
    Mat W = Mat(3, 3, CV_64FC1, tmpW);
    Mat S = Mat::diag(w);
    Mat Rot1 = u * W * vt;   //These matrixes should be rotation matrixes.
    Mat Rot2 = u * W.t() * vt; 
    double det1 = determinant(Rot1);
    double det2 = determinant(Rot2);
    if(det1 < 0) Rot1 = -1*Rot1;
    if(det2 < 0) Rot2 = -1*Rot2;
    
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
    
    vector<Mat> vsols;
    
    Mat P = Mat::eye(3,4,CV_64F);
    Mat Pref = Mat::eye(3,4,CV_64F);

    //Testing all possible solutions.
    FillingMatrix(P, Rot1, T1);
    
    if (IsValidSolution(v1, v2, P, Pref))
        vsols.push_back(P);
   
    FillingMatrix(P, Rot1, T2);
    
    if (IsValidSolution(v1, v2, P, Pref))
        vsols.push_back(P);
        
    FillingMatrix(P, Rot2, T1);
    
    if (IsValidSolution(v1, v2, P, Pref))
        vsols.push_back(P);
        
    FillingMatrix(P, Rot2, T2);
    
    if (IsValidSolution(v1, v2, P, Pref))
        vsols.push_back(P);

    
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
    Mat T1 = GetNormalizingTransformation(v1);
    Mat T2 = GetNormalizingTransformation(v2);
    vector<pair<double, double> > v1normalized;
    for(int i = 0; i < v1.size(); i++){
        Mat vp = (Mat_<double>(3,1) << v1[i].first, v1[i].second, 1.0);
        Mat nvp = T1 * vp;
        pair<double, double> par;
        par.first = nvp.at<double>(0,0);
        par.second = nvp.at<double>(1,0);
        v1normalized.push_back(par);
    }
    vector<pair<double, double> > v2normalized;
    for(int i = 0; i < v2.size(); i++){
        Mat vp = (Mat_<double>(3,1) << v2[i].first, v2[i].second, 1.0);
        Mat nvp = T2 * vp;
        pair<double, double> par;
        par.first = nvp.at<double>(0,0);
        par.second = nvp.at<double>(1,0);
        v2normalized.push_back(par);
    }
    //Test(v1normalized);
    //Test(v2normalized);
    MotionFromEightPointsAlgorithm(v1normalized, v2normalized, T1, T2);
    return 0;
}
