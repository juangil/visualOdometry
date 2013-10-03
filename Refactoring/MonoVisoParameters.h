/* this module saves the monocular visual odometry parameters
    pitch : view angle of the camera with respect to the x-axis(negative in clockwise sense).
    height : Distance of the camera to the ground.
    motionThreshold : Parameter to know if a motion took place
*/

struct VisoMonoParam{
    double pitch;
    double height;
    double motionThreshold;
    VisoMonoParam(double pp = 0.0, double hh = 1.0, double mm = 100.0){pitch = pp; height = hh; motionThreshold = mm;}
    ~VisoMonoParam() {}
};
