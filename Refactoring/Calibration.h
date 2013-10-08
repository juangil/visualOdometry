

/*
    This module has the calibration parameters of the camera used on the visual odometry system.
    (cx, cy) is a principal point that is usually at the image center
     fx, fy are the focal lengths expressed in pixel units.
     
     The calibration Matrix is [fx 0 cx
                                0 fy cy
                                0 0  1 ]
                                
     The GetInverseCalibrationMatrix method returns the calibration matrix inverse.
*/


Mat GetCalibrationMatrix(){
    /* calibracion Dataset Karlsruhe */
    double fx = 645.24;
    double fy = 645.24;
    double cx = 635.96;
    double cy = 194.13;
    /*
    double fx = 923.5295;
    double fy = 922.2418;
    double cx = 507.2222;
    double cy = 383.5822;*/             
    Mat K = (Mat_<double>(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    return K;
}

Mat GetInverseCalibrationMatrix(){
    Mat K = GetCalibrationMatrix();
    Mat Kinv = K.inv();
    return Kinv;
}

void TestCalibration(){
    Mat Kinv = GetInverseCalibrationMatrix();
    cout << Kinv << endl;
    Mat K = GetCalibrationMatrix();
    cout << K << endl;
    cout << "Test" << endl;
    cout << Kinv * K << endl;
}
