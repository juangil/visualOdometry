
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
        double u = v1[idx].first;
        double v = v1[idx].second;
        double up = v2[idx].first;
        double vp = v2[idx].second;
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
    SVD::compute(A, w, u, vt, SVD::FULL_UV);
    // Solving the homogeneous linear system in a least squares sense
//    cout<<"A =="<<endl;
//    cout<< A <<endl;
//    cout<<"w =="<<endl;
//    cout<< w <<endl;
//    cout<<"u =="<<endl;
//    cout<< u <<endl;
//    cout<<"vt =="<<endl;
//    cout<< vt <<endl;
//    cout<<"===="<<endl;
    Mat sol = vt.row(8);
    cout << "sol dims" <<endl;
    cout<<sol.rows<<" "<<sol.cols<<endl;
    double tmpE[3][3] = { {sol.at<double>(0,0), sol.at<double>(0,1), sol.at<double>(0,2)},
                          {sol.at<double>(0,3), sol.at<double>(0,4), sol.at<double>(0,5)},
                          {sol.at<double>(0,6), sol.at<double>(0,7), sol.at<double>(0,8)} };
    Mat EssentialMatrix = Mat(3, 3, CV_64FC1, tmpE);
    
    // Testing normalization
    //EssentialMatrix = GetUnnormalizedEssentialMatrix(EssentialMatrix, NT1, NT2);
    
    
    //Recovering R,t
    SVD::compute(EssentialMatrix, w, u, vt, SVD::FULL_UV);
//    cout<<"E =="<<endl;
//    cout<< EssentialMatrix <<endl;
//    cout<<"w =="<<endl;
//    cout<< w <<endl;
//    cout<<"u =="<<endl;
//    cout<< u <<endl;
//    cout<<"vt =="<<endl;
//    cout<< vt <<endl;
//    cout<<"===="<<endl;
    // Enforcing  the Internal Constraint 
    w.at<double>(0, 2) = 0.0;
    Mat S = Mat::diag(w);
    return u * S * vt;
}

