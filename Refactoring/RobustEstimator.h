/* Ransac */

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

double GetSampsonError(const Mat &x, const Mat &xp, const Mat &E){
    Mat tmp0 = x.t() * E * xp;
    double error = tmp0.at<double>(0,0);
    Mat tmp1 = (E * xp);
    Mat tmp2 = E.t() * x;
    error /= (pow(tmp1.at<double>(0,0), 2) + pow(tmp1.at<double>(1,0), 2) + pow(tmp2.at<double>(0,0), 2) + pow(tmp2.at<double>(1,0), 2));
    return error;
}


int supportSize(const vector<pair<double, double> > &v1,const vector<pair<double, double> > &v2, const double t, const Mat &E){
    int support = 0;
    for(int i = 0; i < v1.size(); i++){
        Mat x = (Mat_<double>(3,1) << v1[i].first, v1[i].second, 1.0);
        Mat xp = (Mat_<double>(3,1) << v2[i].first, v2[i].second, 1.0);
        //Sampson Error
        double error = GetSampsonError(x, xp, E);
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
    
    //Normalizing Input
    
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
    
    // End Normalization
    
    int indexes[s];
    double N = log(1-p)/log(1-pow(1 - e,s)); 
    cout << "iteraciones iniciales" << N << endl;
    int iter=0;
    int bestSupp = 0;
    Mat best;
    while ( (iter < N) && (bestSupp < m) ) {
        randSet(indexes, s, v1normalized.size());
        Mat EssentialMatrix = EightPointsAlgorithm(v1normalized, v2normalized, indexes, s);
        int supp = supportSize(v1normalized, v2normalized, t, EssentialMatrix);
        if (supp > bestSupp) {
            best = EssentialMatrix;
            bestSupp=supp;
        }
        iter++;
        double w = bestSupp/((double)v1normalized.size()); //current probability that a datapoint is inlier
        N = min(N, log(1-p)/log(1-pow(w,s))); //update to current maximum number of iterations
        cout << "iteraciones " << N << endl;
    }
    cout << "Ransac Output for E" << endl;
    cout << "Final support size (amount of inliers):" << bestSupp << endl;
    // TODO: we must compute again E from the inliers (only)
    for(int i = 0; i < v1normalized.size(); i++){
        Mat x = (Mat_<double>(3,1) << v1normalized[i].first, v1normalized[i].second, 1.0);
        Mat xp = (Mat_<double>(3,1) << v2normalized[i].first, v2normalized[i].second, 1.0);
        cout << GetSampsonError(x, xp, best) << endl;
    }
    return best;    
}

