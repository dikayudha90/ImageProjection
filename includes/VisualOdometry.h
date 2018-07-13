#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <viso_stereo.h>
#include <FeatureMatch.h>
#include <png++/png.hpp>

using namespace std;
//using namespace VisualOdometryStereo;

class PoseCalculator{
public:
    PoseCalculator();
    ~PoseCalculator();
    void CalculatePose();
    void CalculatePoseRANSAC();
private :
    void ShowMatched(Mat left_image, Mat right_image, const vector<Matcher::p_match>& matches, const vector<int32_t>& inlier_indices);
private:
    VisualOdometryStereo *viso;
    FeatureMatch *feature;
    //CameraReader *cameraReader;
    Matrix pose;
    Matcher::p_match single_good_match;
    vector<Point2f> previousRawLeftPoints;
    vector<Point2f> previousRawRightPoints;
    vector<DMatch> previousMatches;
    string dir;

};
