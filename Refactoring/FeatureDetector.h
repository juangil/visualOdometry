/*
    Feature generator parameters: the currently used function to find features is genfeatures2.
    
    The function GenFeatures2 receives an image, blockSize(size of the interest window), aperture size(size of the sobel operator),
    the k parameter for cornerness function, the radii for non-maximal supression and upperLimitNormalization(for leaving the cornerness
    in a normalized interval).
    
    The function NonMaxSupression receives and image m, de image coordinates x and y and the nonMaxRadius(to calculate the best feature in
    a neighborghood).
    
    
*/
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


vector<pair<int,int> > GenFeature2(Mat img,int blockSize = 2, int apertureSize = 3, double k = 0.04, int nonMaxRadius = 7, int amount = 2000, int upperLimitNormalization = 10000){
    // se asume que img esta en escala de grises
    Mat dst = Mat::zeros( img.size(), CV_32FC1 );
    cornerHarris( img, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    Mat dst_norm;
    normalize( dst, dst_norm, 0, upperLimitNormalization, NORM_MINMAX, CV_32FC1, Mat() );
    vector<MyKeyPoint> features;
    for( int i = 0; i < dst_norm.rows; i++ ){
        for( int j = 0; j < dst_norm.cols; j++ ){
            if ( NonMaxSupression(dst, i, j, nonMaxRadius)) {
                float c = dst_norm.at<float>(i, j);
                MyKeyPoint current(make_pair(i,j), c);
                features.push_back(current);
            }
        }
    }
    sort(features.begin(), features.end());
    vector<pair<int, int> > ret;
    int limit = features.size();
    if (amount < limit)
       limit = amount;
    for(int i = 0; i < limit; i++)
        ret.push_back(features[i].coord);
    return ret;
}



//vector<pair<int,int> > GenFeature2(Mat img,int blockSize = 2, int apertureSize = 3, double k = 0.04, int nonMaxRadius = 2, int amount = 3000, int upperLimitNormalization = 10000){
//    // Hay problemas con el set porque no guarda repetidos
//    // se asume que img esta en escala de grises
//    Mat dst = Mat::zeros( img.size(), CV_32FC1 );
//    cornerHarris( img, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
//    Mat dst_norm;
//    normalize( dst, dst_norm, 0, upperLimitNormalization, NORM_MINMAX, CV_32FC1, Mat() );
//    //cout << dst_norm << endl;
//    set<MyKeyPoint> data_structure;
//    MyKeyPoint worst;
//    for( int i = 0; i < dst_norm.rows; i++ ){
//        for( int j = 0; j < dst_norm.cols; j++ ){
//            if ( NonMaxSupression(dst, i, j, nonMaxRadius)) {
//                float c = dst_norm.at<float>(i, j);
//                MyKeyPoint current(make_pair(i,j), c);
//                /*
//                if (data_structure.size() < amount){
//                    data_structure.insert(current);
//                }
//                else{
//                    set<MyKeyPoint>::iterator it = data_structure.end();
//                    --it;
//                    worst = *it;
//                    if (current.cornerness > worst.cornerness){
//                        data_structure.erase(worst);
//                        data_structure.insert(current);
//                    }
//                }*/
//                data_structure.insert(current);
//            }
//        }
//    }
//    vector<pair<int, int> > features;
//    for(set<MyKeyPoint>::iterator it = data_structure.begin(); it != data_structure.end(); ++it){
//        features.push_back((*it).coord);
//        //cout<<(*it).cornerness<<endl;
//    } 
//    return features;
//}

