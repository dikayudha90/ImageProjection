#include "Projection.h"

FeatureMatch *featureMatch;

Projection::Projection(){
	featureMatch = new FeatureMatch();
	k1 = Mat_<float>(3,3);
	k2 = Mat_<float>(3,3);

	rotationMatrix = Mat_<float>(3,3);
	translationMatrix = Mat_<float>(3,1);

	k1 << 810.7371, 0, 305.9830, 0, 811.3126, 287.8322, 0, 0, 1;
	k2 << 800.7060, 0, 281.9159, 0, 799.8111, 261.1201, 0, 0, 1;

	rotationMatrix << 0.9995, -0.0075, -0.0315, 0.0085, 0.9994, 0.0323, 0.0312, -0.0326, 0.9990;
	translationMatrix << 59.9551, -0.8243, -8.7038;
}

Projection::~Projection(){
	delete featureMatch;
}

Mat Projection::computeProjectionMatrix(Mat camMat, Mat rotVec, Mat transVec){
	Mat rotMat(3, 3,CV_64F), RTMat(3, 4, CV_64F);
	//Rodrigues(rotVec.at(0), rotMat);
	//hconcat(rotMat, transVec.at(0), RTMat);

	//return (camMat * RTMat);
}

Mat Projection::LinearTriangulation(){
	//Mat projMat1, projMat2, trianCoords4D;

	//featureMatch->KAZERANSAC();
	//vector<Point2f> leftImageFeatures = featureMatch->getLeftImageFeatures();
	//vector<Point2f> rightImageFeatures = featureMatch->getRightImageFeatures();

	//vector<Vec2d> pointsLeft, pointsRight;
//	for(unsigned int i = 0; i < leftImageFeatures.size(); ++i){
//		pointsLeft[i][0] = leftImageFeatures[i].x;
//		pointsLeft[i][1] = leftImageFeatures[i].y;

//		pointsRight[i][0] = rightImageFeatures[i].x;
//		pointsRight[i][1] = rightImageFeatures[i].y;
//	}

//	Mat identityMat = Mat::eye(3,3, CV_32F);
//	Mat zeroVector = Mat::zeros(3, 1, CV_32F);

//	projMat1 = computeProjectionMatrix(k1, identityMat, zeroVector);
//	projMat2 = computeProjectionMatrix(k2, rotationMatrix, translationMatrix);

//	triangulatePoints(projMat1, projMat2, pointsLeft, pointsRight, trianCoords4D);

//	Vec4d triangCoords1 = trianCoords4D.col(0);
//	Vec4d triangCoords2 = trianCoords4D.col(1);

//	Vec3d coords13D, coords23D;
//	for(unsigned int i = 0; i < 3; ++i){
//		coords13D[i] = triangCoords1[i]/triangCoords1[3];
//		coords23D[i] = triangCoords2[i]/triangCoords1[3];
//	}

	//cout << "..." << endl;
}

/*Mat Projection::LinearTriangulation(vector<Point2f> leftPoints, vector<Point2f> rightPoints, Mat K, Mat R, Mat C){
	Mat minC = -C;
	Mat P1_R = K;
	Mat P1_t = Mat::zeros(3, 1, CV_32FC1);
	Mat P2_R = K*R;
	Mat P2_t = K*R*minC;

	Mat P1;
	P1.push_back(P1_R);
	P1.push_back(P1_t);

	Mat P2;
	P2.push_back(P2_R);
	P2.push_back(P2_t);

	for(unsigned int i = 0; i < leftPoints.size(); ++i){
		Point3f point;
		point.x = leftPoints[0].x;
		point.y = leftPoints[0].y;
		point.z = 0;
		Mat leftPoint(point);
	}
}*/

void Projection::Project(){
	Mat projMat1, projMat2, trianCoords4D;

	featureMatch->KAZERANSAC();
	vector<Point2f> leftImageFeatures = featureMatch->getLeftImageFeatures();
	vector<Point2f> rightImageFeatures = featureMatch->getRightImageFeatures();

	vector<Vec2f> pointsLeft, pointsRight;
	for(unsigned int i = 0; i < 10; ++i){
		pointsLeft.push_back(leftImageFeatures[i]);

		pointsRight.push_back(rightImageFeatures[i]);
	}

	Mat identityMat = Mat::eye(3,3, CV_32F);
	Mat zeroVector = Mat::zeros(3, 1, CV_32F);
	//vector<Mat> identityMatVec, zeroMatVect;

	//identityMat.push_back(identityMat);
	//zeroMatVect.push_back(zeroVector);

	computeProjectionMatrix(k1, identityMat, zeroVector);
	//identityMatVec.clear();
	//zeroMatVect.clear();
	//projMat2 = computeProjectionMatrix(k2, rotationMatrix, translationMatrix);

//	triangulatePoints(projMat1, projMat2, pointsLeft, pointsRight, trianCoords4D);

//	Vec4d triangCoords1 = trianCoords4D.col(0);
//	Vec4d triangCoords2 = trianCoords4D.col(1);

//	Vec3d coords13D, coords23D;
//	for(unsigned int i = 0; i < 3; ++i){
//		coords13D[i] = triangCoords1[i]/triangCoords1[3];
//		coords23D[i] = triangCoords2[i]/triangCoords1[3];
//	}

//	cout << "..." << endl;
	//LinearTriangulation();
//	featureMatch->RANSACKNNMatch();
//	vector<Point2f> leftImageFeatures = featureMatch->getLeftImageFeatures();
//	vector<Point2f> rightImageFeatures = featureMatch->getRightImageFeatures();
}
