#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <set>
<<<<<<< HEAD
#include <string>
#include <sstream>
=======
#include <algorithm>
>>>>>>> ab8f12ce1250a62acf315c10a29e8b9d3a22c32f

using namespace std;
using namespace cv;

const int MAX_INT = 1<<30;

string toString(int a){
     stringstream ss;
     ss << a;
     return ss.str();
}
     

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

vector<pair<int,int> > GenFeature(Mat img,int blockSize = 2, int apertureSize = 3, double k = 0.04, int nonMaxRadius = 8, int upperLimitNormalization = 5000){
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
    int aim = 7*(total / 10); // Este parametro es fundamental para ajustar el numero de caracteristicas obtenidas al final.
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
   // db = dst_norm_scaled;
    return features;
}


struct MyKeyPoint{
    pair<int,int> coord;
    float cornerness;
    MyKeyPoint(){}
    MyKeyPoint(pair<int,int> c, float m){coord = c; cornerness = m;}
    ~MyKeyPoint() {}
    bool operator <(const MyKeyPoint &other) const {
        return cornerness > other.cornerness;
    }
};



vector<pair<int,int> > GenFeature2(Mat img,int blockSize = 2, int apertureSize = 3, double k = 0.04, int nonMaxRadius = 2, int amount = 3000, int upperLimitNormalization = 10000){
    // se asume que img esta en escala de grises
    Mat dst = Mat::zeros( img.size(), CV_32FC1 );
    cornerHarris( img, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    Mat dst_norm;
    normalize( dst, dst_norm, 0, upperLimitNormalization, NORM_MINMAX, CV_32FC1, Mat() );
    //cout << dst_norm << endl;
    set<MyKeyPoint> data_structure;
    MyKeyPoint worst;
    for( int i = 0; i < dst_norm.rows; i++ ){
        for( int j = 0; j < dst_norm.cols; j++ ){
            if ( NonMaxSupression(dst, i, j, nonMaxRadius)) {
                
                float c = dst_norm.at<float>(i, j);
                MyKeyPoint current(make_pair(i,j), c);
                /*
                if (data_structure.size() < amount){
                    data_structure.insert(current);
                }
                else{
                    set<MyKeyPoint>::iterator it = data_structure.end();
                    --it;
                    worst = *it;
                    if (current.cornerness > worst.cornerness){
                        data_structure.erase(worst);
                        data_structure.insert(current);
                    }
                }*/
                data_structure.insert(current);
            }
        }
    }
    vector<pair<int, int> > features;
    for(set<MyKeyPoint>::iterator it = data_structure.begin(); it != data_structure.end(); ++it){
        features.push_back((*it).coord);
        //cout<<(*it).cornerness<<endl;
    } 
    return features;
}

/*Matching*/

int SumofAbsoluteDifferences(Mat img1, Mat img2, pair<int,int> f1, pair<int,int> f2, int w = 10){
    if ( (img1.rows != img2.rows ) || (img1.cols != img2.cols)){
        printf("Las dimensiones de las imagenes no coinciden");
        return 0;
    }
    int limitx = img1.rows;
    int limity = img1.cols;
    int response = 0;
    for(int i = -w; i <= w; i++){
        for(int j = -w; j <= w; j++){
            int nx1 = f1.first + i;
            int ny1 = f1.second + j;
            int nx2 = f2.first + i;
            int ny2 = f2.second + j;
            bool c1 = (nx1 < 0 || ny1 < 0 || nx2 < 0 || ny2 < 0);
            bool c2 = (nx1 >= limitx || ny1 >= limity || nx2 >= limitx || ny2 >= limity);
            if (c1 || c2) continue;
            response += abs(img1.at<int>(nx1,ny1) - img2.at<int>(nx2,ny2));   
        }
    }
    return response;
}

     

void DeterminingFavorites(Mat img1, Mat img2, vector< pair<int,int> > &f1, vector< pair<int,int> > &f2, int *favorites, int delta = 15){
    for(int i = 0; i < f1.size(); i++){
        pair<int, int> current1 = f1[i];
        int menor = MAX_INT;
        int idMenor = -1;
        for(int j = 0; j < f2.size(); j++){
            pair<int, int> current2 = f2[j];
            int dx = abs(current1.first - current2.first);
            int dy = abs(current1.second - current2.second);
            if(dx > delta || dy > delta) continue;
            int similarity = SumofAbsoluteDifferences(img1, img2, current1, current2);
            if(similarity < menor){
                menor = similarity;
                idMenor = j;
            } 
        }
        favorites[i] = idMenor;
    }
    return;
}


vector<pair<int,int> > harrisFeatureMatcherMCC(Mat img1, Mat img2, vector< pair<int,int> > featuresImg1, vector< pair<int,int> > featuresImg2){
    int favoritesfromimg1[featuresImg1.size()]; // en la posicion i se guarda el favorito de la característica i de la imagen 1.
    int favoritesfromimg2[featuresImg2.size()]; // en la posicion i se guarda el favorito de la característica i de la imagen 2.
    vector< pair<int,int> > correspondences;
    DeterminingFavorites(img1, img2, featuresImg1, featuresImg2, favoritesfromimg1);
    DeterminingFavorites(img2, img1, featuresImg2, featuresImg1, favoritesfromimg2);
    for(int i = 0; i < featuresImg1.size(); ++i){
        if(favoritesfromimg2[favoritesfromimg1[i]] == i) correspondences.push_back(make_pair(i, favoritesfromimg1[i]));
    }
    return correspondences;
}

/*end Matching*/

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
          string num = toString(i);
          imwrite("new_image"+num+".jpg", new_image);
          if(i == 20) break;
          //imshow("correspondences", new_image);
          //waitKey(0);     
     }
     imwrite("new_image.jpg", new_image);
     //cout<<"buenas: "<<buenas<<" malas: "<<correspondences.size() - buenas<<endl;
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
    vector<pair<int,int> > fts1 = GenFeature2(imgray1);
    vector<pair<int,int> > fts2 = GenFeature2(imgray2);
    cout<<"cantidad características imagen 1: "<<fts1.size()<<" cantidad características imagen 2: "<<fts2.size()<<endl;    
    //namedWindow("edges",1);
    //imshow("edges", db);
    //cout<<fts.size()<<endl;
    //waitKey(0);
    //for(int i = 0; i < fts.size(); ++i) cout<< fts[i] << end;
    vector< pair<int,int> > correspondences = harrisFeatureMatcherMCC(imgray1, imgray2, fts1, fts2);
    cout <<"cantidad de correspondencias " << correspondences.size() << endl;
    debugging(imgray1, imgray2, fts1, fts2, correspondences);
    return 0;
}

