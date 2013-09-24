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


void randSet(int *indexes, int s,int n) {
    int i=0;
    while (i<s) {
        int r = rand() % n;
        char ok=1;
        for (int j=0; j<i; j++) if (r==indexes[j]) ok=0;
        if (ok)
            indexes[i++] = r;
    }
}


int supportSize(const vector<pair<double, double> > &v1,const vector<pair<double, double> > &v2, const double t, const Mat &E){
    int support = 0;
    for(int i = 0; i < v1.size(); i++){
        Mat x = (Mat_<double>(3,1) << v1[i].first, v1[i].second, 1.0);
        Mat xp = (Mat_<double>(3,1) << v2[i].first, v2[i].second, 1.0);
        //Sampson Error
        double error = x.t() * E * xp;
        Mat tmp1 = (E * xp);
        Mat tmp2 = E.t() * x;
        error /= (pow(tmp1.at<double>(0,0), 2) + pow(tmp1.at<double>(1,0), 2) + pow(tmp2.at<double>(0,0), 2) + pow(tmp2.at<double>(1,0), 2));
        if (fabs(error) < t)
            support++;
    }
    return support;
}


Mat Ransac(const vector<pair<double, double> > &v1,const vector<pair<double, double> > &v2, double p, double t, double e, int s, int m){
    /*
        v1, v2 correspondences vectors
        p -> desired probabilty of selecting merely inliers in at least one Ransac step
        t -> Measure to test if a correspondence is an inlier or an outlier(Sampson Error)
        e -> Probability of selecting outliers given a model
        s -> minimal dataset to compute a model
        m -> minimum acceptable for the number of inliers
    */
    int indexes[s];
    double N = log(1-p)/log(1-pow(1 - e,n)); 
    int iter=0;
    int bestSupp = 0;
    Mat best;
    while ( (iter < N) && (bestSupp < m) ) {
        randSet(indexes, s, n);
        Mat EssentialMatrix = MotionFromEightPointsAlgorithm(v1, v2, indexes, s);
        int supp = supportSize(v1, v2, t, EssentialMatrix);
        if (supp > bestSupp) {
            best = EssentialMatrix;
            bestSupp=supp;
        }
        iter++;
        double w = bestSupp/((double)n); //current probability that a datapoint is inlier
        N = min(N, log(1-p)/log(1-pow(w,n))); //update to current maximum number of iterations
    }
      
}


int main(){
    
    return 0;
}
