#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

const int MAX_INT = 1<<30;
const float MINFLOAT= numeric_limits<float>::min();

//template<class T>
struct patch{
     float A,B,C;
     
     patch(float a, float b, float c) : A(a), B(b), C(c) {}
     patch() {}
};
     

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

vector<pair<int,int> > GenFeature(Mat img,int blockSize = 2, int apertureSize = 3, float k = 0.04, int nonMaxRadius = 3, int upperLimitNormalization = 5000){
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
    int aim = total / 5; // Este parametro es fundamental para ajustar el numero de caracteristicas obtenidas al final.
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
            if ( NonMaxSupression(dst_norm, i, j, nonMaxRadius) &&  ( ((int)dst_norm.at<float>(i,j)) >= thresh) ) {
                features.push_back(make_pair(i,j));
                circle( dst_norm_scaled, Point( j, i ), 5,  Scalar(0), 2, 8, 0 ); //
            }
        }
    }
    return features;
}

/*Matching--------------------------------------------------------------------------------------------------------------------------*/

vector<patch> constantsFromNCC(vector< pair<int,int> > &featuresImg, Mat img, int w = 5){
    vector<patch> responseFts(featuresImg.size());
    
    for(int i = 0; i < featuresImg.size(); i++){
          float A = 0.0,B = 0.0,C = 0.0;
          for(int j = -w; j <= w; j++){
               for(int k = -w; k <= w; k++){
                    int nx = featuresImg[i].first + j;
                    int ny = featuresImg[i].second + k; 
                    cout<< img.at<float>(nx,ny)<<endl;             
                    A += img.at<float>(nx,ny);
                    B += img.at<float>(nx,ny)*img.at<float>(nx,ny);
               }
          }
          C = 1.0/sqrt(((w+w)*(w+w)*B) - A);
          responseFts[i] = patch(A,B,C);          
    }
    return responseFts;
}

void DeterminingFavorites(Mat img1, Mat img2, vector< pair<int,int> > &f1, vector< pair<int,int> > &f2, int *favorites, vector<patch> &rf1, vector<patch> &rf2, int delta = 15, int w = 5){
    for(int i = 0; i < f1.size(); i++){
        pair<int, int> current1 = f1[i];
        float mayor = MINFLOAT;
        int idMayor = -1;
        for(int j = 0; j < f2.size(); j++){
            pair<int, int> current2 = f2[j];
            int dx = abs(current1.first - current2.first);
            int dy = abs(current1.second - current2.second);
            if(dx > delta || dy > delta) continue;
            float D = 0.0;
            for(int k = -w; k <= w; k++){
               for(int l = -w; l <= w; l++){
                    int nx = current1.first + k;
                    int ny = current1.second + l;
                    int n2x = current2.first + k;
                    int n2y = current2.second + l;
                    D += img1.at<float>(nx,ny)*img2.at<float>(n2x,n2y);
               }
            }
            int tmp = (w+w)*(w+w);
            float correlation = ((tmp*D) - (rf1[i].A*rf2[j].A))*(rf1[i].C*rf2[j].C);
            //cout<<correlation<<" "<<mayor<<endl;
            if(correlation > mayor){
                mayor = correlation;
                idMayor = j;
            } 
        }
        favorites[i] = idMayor;
    }
    return;
}

vector<pair<int,int> > harrisFeatureMatcherMCC(Mat img1, Mat img2, vector< pair<int,int> > featuresImg1, vector< pair<int,int> > featuresImg2){
    int favoritesfromimg1[featuresImg1.size()]; // en la posicion i se guarda el favorito de la característica i de la imagen 1.
    int favoritesfromimg2[featuresImg2.size()]; // en la posicion i se guarda el favorito de la característica i de la imagen 2.
    vector< pair<int,int> > correspondences;
    vector<patch> responseFts1 (featuresImg1.size());
    vector<patch> responseFts2 (featuresImg2.size());
    responseFts1 = constantsFromNCC(featuresImg1, img1);
    responseFts2 = constantsFromNCC(featuresImg2, img2);
    //for(int i = 0; i < featuresImg1.size(); ++i) cout<<responseFts1[i].A<<" "<<responseFts1[i].B<<" "<<responseFts1[i].C<<endl;
    DeterminingFavorites(img1, img2, featuresImg1, featuresImg2, favoritesfromimg1, responseFts1, responseFts2);
    DeterminingFavorites(img2, img1, featuresImg2, featuresImg1, favoritesfromimg2, responseFts2, responseFts1);
    //for(int i = 0; i < featuresImg1.size(); ++i)cout<<"favorita de "<<i<<" es "<<favoritesfromimg1[i]<<endl;
    //for(int i = 0; i < featuresImg2.size(); ++i)cout<<"favorita de "<<i<<" es "<<favoritesfromimg2[i]<<endl;
    for(int i = 0; i < featuresImg1.size(); ++i){
        if(favoritesfromimg2[favoritesfromimg1[i]] == i) correspondences.push_back(make_pair(i, favoritesfromimg1[i]));
    }
    return correspondences;
}

/*end Matching--------------------------------------------------------------------------------------------------------------------------*/

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
     
     namedWindow("correspondences",1);
     int buenas = 0;
     for(int i = 0; i < correspondences.size(); i++){
          pair<int,int> feature1 = fts1[correspondences[i].first];
          pair<int,int> feature2 = fts2[correspondences[i].second];
          //Point p1(feature1.second, feature1.first);
          Point p1(feature1.second, feature1.first);
          Point p2(feature2.second, feature2.first + img1.rows); 
          if(feature1.second == feature2.second) buenas += 1;
          Scalar color(25.0);
          circle(new_image, p1, 5, color, 2);
          circle(new_image, p2, 5, color, 2);
          line(new_image, p1, p2, color);
          imshow("correspondences", new_image);
          waitKey(0);     
     }
     cout<<"buenas: "<<buenas<<" malas: "<<correspondences.size() - buenas<<endl;
     return;
}

int main(int argc, char** argv){
    Mat img1, img2, imgray1, imgray2;
    if( argc != 3){
     cout <<"Pasar el nombre de la imagen" << endl;
     return -1;
    }
    namedWindow("ventana",1);//
    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    cvtColor( img1, imgray1, CV_BGR2GRAY );
    cvtColor( img2, imgray2, CV_BGR2GRAY );
    vector<pair<int,int> > fts1 = GenFeature(imgray1);
    vector<pair<int,int> > fts2 = GenFeature(imgray2);
    cout<<"cantidad características imagen 1: "<<fts1.size()<<" cantidad características imagen 2: "<<fts2.size()<<endl;    
    //namedWindow("edges",1);
    //imshow("edges", db);
    //cout<<fts.size()<<endl;
    //waitKey(0);
    //for(int i = 0; i < fts.size(); ++i) cout<< fts[i] << end;
    vector< pair<int,int> > correspondences = harrisFeatureMatcherMCC(imgray1, imgray2, fts1, fts2);
    cout<<correspondences.size()<<endl;
    debugging(imgray1, imgray2, fts1, fts2, correspondences);
    //cout<<correspondences.size()<<endl;
    
    return 0;
}
