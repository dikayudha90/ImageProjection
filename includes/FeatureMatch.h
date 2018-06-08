#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include "CameraReader.h"

using namespace cuda;

class FeatureMatch{
public :
    FeatureMatch();
    ~FeatureMatch();
    void Matching();
private:
    SURF_CUDA surf;
    CameraReader *camReader;
    int minHessian;
};
