



bool CompareTriangulatedPoints(const Mat &x1, const Mat &x2){
    double dist1 = fabs(x1.at<double>(0,0)) + fabs(x1.at<double>(1,0)) + fabs(x1.at<double>(2,0)); // Manhattan distance 
    double dist2 = fabs(x2.at<double>(0,0)) + fabs(x2.at<double>(1,0)) + fabs(x2.at<double>(2,0));
    return (dist1 < dist2); 
}

Mat FillingMatrix(Mat &orig, Mat &orig2){
    Mat dest = Mat::zeros(3,4, CV_64F);
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            dest.at<double>(i,j) = orig.at<double>(i,j); 
    for(int i = 0; i < 3; i++)
        dest.at<double>(i, 3) = orig2.at<double>(i,0);
    return dest;
}

Mat TriangulatePoint(const Point2d &pt1, const Point2d &pt2, const Mat &P1, const Mat &P2)
{
    Mat A(4,4,CV_64F);
    Mat w, u, vt;
    
	A.at<double>(0,0) = pt1.x*P1.at<double>(2,0) - P1.at<double>(0,0);
	A.at<double>(0,1) = pt1.x*P1.at<double>(2,1) - P1.at<double>(0,1);
	A.at<double>(0,2) = pt1.x*P1.at<double>(2,2) - P1.at<double>(0,2);
	A.at<double>(0,3) = pt1.x*P1.at<double>(2,3) - P1.at<double>(0,3);

	A.at<double>(1,0) = pt1.y*P1.at<double>(2,0) - P1.at<double>(1,0);
	A.at<double>(1,1) = pt1.y*P1.at<double>(2,1) - P1.at<double>(1,1);
	A.at<double>(1,2) = pt1.y*P1.at<double>(2,2) - P1.at<double>(1,2);
	A.at<double>(1,3) = pt1.y*P1.at<double>(2,3) - P1.at<double>(1,3);

	A.at<double>(2,0) = pt2.x*P2.at<double>(2,0) - P2.at<double>(0,0);
	A.at<double>(2,1) = pt2.x*P2.at<double>(2,1) - P2.at<double>(0,1);
	A.at<double>(2,2) = pt2.x*P2.at<double>(2,2) - P2.at<double>(0,2);
	A.at<double>(2,3) = pt2.x*P2.at<double>(2,3) - P2.at<double>(0,3);

	A.at<double>(3,0) = pt2.y*P2.at<double>(2,0) - P2.at<double>(1,0);
	A.at<double>(3,1) = pt2.y*P2.at<double>(2,1) - P2.at<double>(1,1);
	A.at<double>(3,2) = pt2.y*P2.at<double>(2,2) - P2.at<double>(1,2);
	A.at<double>(3,3) = pt2.y*P2.at<double>(2,3) - P2.at<double>(1,3);

    SVD::compute(A, w, u, vt, SVD::FULL_UV);
	//SVD svd(A); another way to calculate SVD

    Mat X(4,1,CV_64F);

	X.at<double>(0,0) = vt.at<double>(3,0);
	X.at<double>(1,0) = vt.at<double>(3,1);
	X.at<double>(2,0) = vt.at<double>(3,2);
	X.at<double>(3,0) = vt.at<double>(3,3);

    return X;
}


bool InFrontOf(const cv::Mat &X, const cv::Mat &P)
{
    // back project
    cv::Mat X2 = P*X;
    double w = X2.at<double>(2,0);
    double W = X.at<double>(3,0);
    double ans = (w/W);
    return (ans > 0.0);
}


void IsValidSolution(const vector<pair<double, double> > &v1,const vector<pair<double, double> > &v2, const vector<int> &idx, const Mat &P1,  Mat &P2,
    vector<Mat> &inliers){
    //inliers.clear();
    for(int  i = 0; i < idx.size(); i++){
        int index = idx[i];
        Point2d pt1 = Point2d(v1[index].first, v1[index].second);
        Point2d pt2 = Point2d(v2[index].first, v2[index].second);
        Mat triangulated = TriangulatePoint(pt1, pt2, P1, P2);
        if (InFrontOf(triangulated, P1) && InFrontOf(triangulated, P2))
            inliers.push_back(triangulated);
    }
    cout<<"Amount of point in front: " << inliers.size() << endl;
}


bool GetRotationAndTraslation(Mat E, const vector<pair<double, double> > &v1, const vector<pair<double, double> > &v2, const vector<int> &idx, Mat &FP){
    // E is valid essential Matrix (Rank 2)
    Mat w,u,vt;
    SVD::compute(E, w, u, vt, SVD::FULL_UV);
    cout << "valores singulares EssentialMatrix" << w << endl;
    double tmpW[3][3] = { {0,1,0},
                          {-1,0,0},
                          {0,0,1} };
    Mat W = Mat(3, 3, CV_64FC1, tmpW);
    Mat S = Mat::diag(w);
    Mat Rot1 = u * W * vt;   //These matrixes should be rotation matrixes.
    Mat Rot2 = u * W.t() * vt; 
    double det1 = determinant(Rot1);
    double det2 = determinant(Rot2);
    if(det1 < 0) Rot1 = -1*Rot1;
    if(det2 < 0) Rot2 = -1*Rot2;
    Mat T = u * W * S * u.t(); // This matrix should be a skew matrix.
    Mat T1 = (Mat_<double>(3,1) << T.at<double>(2,1), T.at<double>(0,2), T.at<double>(1,0));
    Mat T2 = -1*T1;
    
    Mat P1,P = Mat::eye(3,4,CV_64F);
    //Assumption:The camera matrix Pref is always at I|0
    Mat Pref = Mat::eye(3,4,CV_64F);
    
    vector<Mat> Ps;
    Ps.push_back(FillingMatrix(Rot1, T1));
    Ps.push_back(FillingMatrix(Rot1, T2));
    Ps.push_back(FillingMatrix(Rot2, T1));
    Ps.push_back(FillingMatrix(Rot2, T2));
    
    //Testing all possible solutions.
    
    Mat bestP;
    vector<Mat> Xbest;
    
    for(int i = 0; i < 4; i++){
        Mat P = Ps[i];
        vector<Mat> inliers;
        IsValidSolution(v1, v2, idx, P, Pref, inliers);
        if (inliers.size() > Xbest.size()){
            bestP = P;
            Xbest = inliers;
        }
    }
    
    //normalizing 3D points
    
    for(int i = 0; i < Xbest.size(); i++){
        Mat current = Xbest[i];
        double denominator = current.at<double>(3,0);
        current = current * (1/denominator);  
    }
    
    for(int i = 0; i < Xbest.size(); i++){
        Mat current = Xbest[i];
        if (current.at<double>(2,0) < 0){
            printf("Points behind the image plane");
            return false;
        }
    }
    
    if (Xbest.size() < 10){
        printf("at least 10 points are necessary to continue");
        return false;
    }
    
    sort(Xbest.begin(), Xbest.end(), CompareTriangulatedPoints); //The efficiency of this operation can be improved
    
    Mat Median = Xbest[Xbest.size() / 2];
    double DistMedian = fabs(Median.at<double>(0,0)) + fabs(Median.at<double>(1,0)) + fabs(Median.at<double>(2,0));

    if (DistMedian > GLOBAL_PARAMETERS.motionThreshold){
       printf("Little Motion\n");   
       return true;
    }

    Mat YandZ = Mat::zeros(2, Xbest.size(), CV_64F);

    for(int i = 0; i < Xbest.size(); i++){
       YandZ.at<double>(0, i) = Xbest[i].at<double>(1,0);
       YandZ.at<double>(1, i) = Xbest[i].at<double>(2,0);  
    }

    Mat Rot = (Mat_<double>(1,2) << cos(-GLOBAL_PARAMETERS.pitch), sin(-GLOBAL_PARAMETERS.pitch));
    
    cout << "Rot:" << Rot << endl;

    Mat RotY = Rot * YandZ;
    
    cout << "Roty:" << RotY << endl;

    double sigma = DistMedian/ (GLOBAL_PARAMETERS.motionThreshold * 0.5);   // own change: variable denominator
    double weight = 1.0/(2.0*sigma*sigma);
    double best_sum = 0;
    int best_idx = 0;

    for(int i = 0; i < Xbest.size(); i++){
        if (RotY.at<double>(0,i) < DistMedian / GLOBAL_PARAMETERS.motionThreshold)  //?
            continue;
        double sum = 0;
        for(int j = 0; j < Xbest.size(); j++){
            double dist = RotY.at<double>(0, j) - RotY.at<double>(0, i);
            sum += exp(-1 * dist * dist * weight);
        }
        if (sum > best_sum){
            best_sum = sum;
            best_idx = i;
        }
    }
    
    double ScaleFactor = GLOBAL_PARAMETERS.height / RotY.at<double>(0, best_idx);   // I think this denominator can be negative
    cout << "The Real Scale:" << ScaleFactor << endl;
    
    //Multiplying Traslation by the scale factor
    for(int i = 0; i < 3; i++)
        bestP.at<double>(i, 3) = bestP.at<double>(i, 3) * ScaleFactor;
    
    FP = bestP;
    return true; 
}
