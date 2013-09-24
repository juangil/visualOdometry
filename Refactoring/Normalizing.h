
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
