#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include "CameraReader.h"
//#include "AKAZE.h"
#include "cuda_profiler_api.h"

using namespace cuda;
//using namespace std;
//using namespace libAKAZECU;

class FeatureMatch{
public :
    FeatureMatch();
    ~FeatureMatch();
    void Matching();
    void RANSACFeatureMatch();
    void RANSACKNNMatch();
    void KAZERANSAC();
    vector<Point2f> getLeftImageFeatures(){
        return ransacLeftPoint;
    }

    vector<Point2f> getRightImageFeatures(){
        return ransacRightPoint;
    }

    vector<DMatch> getGoodMatch(){
        return GoodMatches;
    }

    CameraReader *camReader;
private:
    //AKAZEOptions options;
    vector<DMatch> GoodMatches;
    SURF_CUDA surf;

    int minHessian;
    Mat fundamentalMatrix;
    const double CONFIDENCE = 0.99;
    const double INLIER_RATIO = 0.2; // Assuming lots of noise in the data!
    const double INLIER_THRESHOLD = 2.0; // pixel distance

    vector<Point2f> ransacLeftPoint, ransacRightPoint;
};
