
#ifndef IMG_LIB_H
#define IMG_LIB_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
using namespace std;


struct ImgError {
    int code;
    string message;

    ImgError(int code, const char *error) {
        this->code = code; message = string(error);
    }
};

void harrisFilter(Image<T> &dest, Image<T> &org, int w = 5, double k=0.01);


void harrisFilter(Mat dest, Mat org, int w = 5, double k=0.05) {
    int n = org.rows; int m = org.cols;
    Mat dx (n,m); Mat dy(n,m);
    sobel(org, dx, -1, 1, 0, w);
    sobel(org, dx, -1, 1, 0, w);      
    int wh = w/2;
    sobelFilter(dx,dy,org);    
    for(int row = 0; row < org.rows; ++row) {
        for(int col = 0; col < org.cols ; ++col) {
            double a=0, b=0, c=0;
            for(int i = -wh; i <= wh; ++i) {
                for(int j = -wh; j <= wh; ++j) {
                    double tx = dx.at(row+i, col+j), ty = dy.at(row+i,col+j);
                    a += tx*tx;
                    c += tx*ty;
                    b += ty*ty;
                }
            }
            double ab = a*b - c*c;
            double apb = a+b;
            dest.at(i,j) = (ab - k*(apb*apb));
        }
    }  
}

/*template<class T>
void cart2pol(Image<T> &r, Image<T> &theta, Image<T> &x, Image<T> &y) {
    if (!(r.rows == theta.rows && theta.rows == x.rows && x.rows == y.rows))
        throw new ImgError(12, "the images dont have the same dimensions on cart2pol");
    int rows = r.rows;
    int cols = r.cols;
    for(int i = 0; i < rows; i++){
        T* rx = x.getRow(i);
        T* ry = y.getRow(i);
        T* rr = r.getRow(i);
        T* rt = theta.getRow(i);
        for(int j = 0; j < cols; j++){
            rr[j] = sqrt( (rx[j] * rx[j]) + (ry[j] * ry[j]));
            rt[j] = atan2(ry[j], rx[j]);
        }
    }
}*/


#endif
