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
    cout<<"cantidad características imagen 1: "<<fts1.size()<<" cantidad características imagen 2: "<<fts2.size()<<endl;
    vector< pair<int,int> > correspondences = harrisFeatureMatcherMCC(imgray1, imgray2, fts1, fts2);
    cout <<"cantidad de correspondencias " << correspondences.size() << endl;
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
    //Mat RobustEssentialMatrix= Ransac(v1, v2, 0.99, 0.001, 0.5, 12, v1.size());  // TODO: Estimate experimentally the value of T */
    return 0;
}
