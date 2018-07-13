#include "FeatureMatch.h"

class Projection{
public:
    Projection();
    ~Projection();
    void Project();

private:
    //Mat LinearTriangulation(vector<Point2f> leftPoints, vector<Point2f> rightPoints, Mat K, Mat R, Mat C);
    Mat LinearTriangulation();
    Mat computeProjectionMatrix(Mat camMat, Mat rotVec, Mat transVec);

    Mat_<float> k1, k2, rotationMatrix, translationMatrix;
};
