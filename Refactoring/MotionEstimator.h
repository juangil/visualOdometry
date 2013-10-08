
Mat EightPointsAlgorithm(const vector<pair<double, double> > &v1, const vector<pair<double, double> > &v2, const int * indexes, const int s){
    // v1 and v2 represent the correspondences (normalized).
    // NT1 and NT2 are the normalizing transformations.
    if (v1.size() != v2.size()){
        fprintf(stderr, "Los tamanos de los vectores de correspondencias son diferentes");
        return (Mat_<double>(1,1) << 0.0);
    }
    
    int ncorrespondences = s;
    Mat A = Mat::zeros(ncorrespondences, 9, CV_64FC1);
    Mat w, u, vt;
    for(int i = 0; i < ncorrespondences; i++){
        int idx = indexes[i];
        double up = v1[idx].first;
        double vp = v1[idx].second;
        double u = v2[idx].first;
        double v = v2[idx].second;
        
        //printf("idx: %d ---- up:%.6lf , vp: %.6lf, u:%.6lf , v: %.6lf \n", idx, up, vp, u, v);
        
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
    
//    cout << "A matrix"  << endl;
//    cout << A << endl;
    
    SVD::compute(A, w, u, vt, SVD::FULL_UV);
    Mat sol = vt.row(8);
    double tmpE[3][3] = { {sol.at<double>(0,0), sol.at<double>(0,1), sol.at<double>(0,2)},    // This Method of inicializing matrixes seems to work wrong
                          {sol.at<double>(0,3), sol.at<double>(0,4), sol.at<double>(0,5)},
                          {sol.at<double>(0,6), sol.at<double>(0,7), sol.at<double>(0,8)} };
    Mat EssentialMatrix = Mat(3, 3, CV_64FC1, tmpE);
    // Enforcing  the Internal Constraint 
    SVD::compute(EssentialMatrix, w, u, vt, SVD::FULL_UV);
    w.at<double>(0, 2) = 0.0;
    Mat S = Mat::diag(w);
    return u * S * vt;
}

