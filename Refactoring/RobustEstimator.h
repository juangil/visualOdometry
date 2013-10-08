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
    Mat tmp0 = xp.t() * E * x;
    double error = tmp0.at<double>(0,0);
    error = error * error;
    Mat tmp1 = (E .t() * xp);
    Mat tmp2 = E * x;
    double den = tmp1.at<double>(0,0) * tmp1.at<double>(0,0) + tmp1.at<double>(1,0) * tmp1.at<double>(1,0) + tmp2.at<double>(0,0) * tmp2.at<double>(0,0) + tmp2.at<double>(1,0) * tmp2.at<double>(1,0);
    error = error / den;
    //error /= (pow(tmp1.at<double>(0,0), 2) + pow(tmp1.at<double>(1,0), 2) + pow(tmp2.at<double>(0,0), 2) + pow(tmp2.at<double>(1,0), 2));
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

Mat Ransac(const vector<pair<double, double> > &v1,const vector<pair<double, double> > &v2, double t, int s, int m, vector<int> &idx){
    /*
        v1, v2 correspondences vectors
        t -> Measure to test if a correspondence is an inlier or an outlier(Sampson Error)
        s -> minimal dataset to compute a model
        m -> ransac iterations
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
    //double N = log(1-p)/log(1-pow(1 - e,s));  this calculation isn't used by now
    //cout << "iteraciones iniciales" << N << endl;
    int iter=0;
    int bestSupp = 0;
    Mat best;
    srand(0);
    while (iter < m){
    
        randSet(indexes, s, v1normalized.size());
        
        /*
        cout << "Start idx" << endl;
        for(int i = 0; i < s; i++)
            cout << indexes[i] << " ";
        cout << endl << "End idx" << endl;*/
        
        Mat EssentialMatrix = EightPointsAlgorithm(v1normalized, v2normalized, indexes, s);
        
        /*
        cout << "Essential Matrix" << endl;
        cout << EssentialMatrix << endl;
        namedWindow("correspondences");
        waitKey(0);*/
        
        int supp = supportSize(v1normalized, v2normalized, t, EssentialMatrix);
        if (supp > bestSupp) {
            cout << supp << endl;
            best = EssentialMatrix;
            bestSupp=supp;
        }
        iter++;
        //double w = bestSupp/((double)v1normalized.size()); //current probability that a datapoint is inlier
        //N = min(N, log(1-p)/log(1-pow(w,s))); //update to current maximum number of iterations, this calculation isn't used by now.
    }
    
    cout << "inliers" << endl;
    
    int inliers[bestSupp];
    int counter = 0;
    for(int i = 0; i < v1normalized.size(); i++){
        Mat x = (Mat_<double>(3,1) << v1normalized[i].first, v1normalized[i].second, 1.0);
        Mat xp = (Mat_<double>(3,1) << v2normalized[i].first, v2normalized[i].second, 1.0);
        double error = GetSampsonError(x, xp, best);
        if (fabs(error) < t){
            inliers[counter++] = i;
            idx.push_back(i);
            cout << i << " ";
        }
    }
    // computing again E from inliers
    Mat EssentialMatrix = EightPointsAlgorithm(v1normalized, v2normalized, inliers, bestSupp);
    
    
    cout << "Ransac Output for E" << endl;
    cout << EssentialMatrix << endl;
    cout << "Final support size (amount of inliers):" << bestSupp << endl;
    cout << "up:" << v1[0].first << " vp:" << v1[0].second << " uc: " << v2[0].first << "vc: " << v2[0].second << endl;
    namedWindow("correspondences");
    waitKey(0);
 
    // Denormalizing
    EssentialMatrix = T1.t() * EssentialMatrix * T2;
    //re-enforcing the internal constraint
    Mat w, u, vt;
    SVD::compute(EssentialMatrix, w, u, vt, SVD::FULL_UV);
    w.at<double>(0, 2) = 0.0;
    Mat S = Mat::diag(w);
    EssentialMatrix = u * S * vt;
    return EssentialMatrix;    
}

