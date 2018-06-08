#include "FeatureMatch.h"

FeatureMatch::FeatureMatch(){
	camReader = new CameraReader(5);
	minHessian = 400;
	surf.hessianThreshold = minHessian;
}

FeatureMatch::~FeatureMatch(){
	delete camReader;
}

void FeatureMatch::Matching(){
	camReader->Update();
	//imshow("1", camReader->getCamLeft());
	//imshow("2", camReader->getCamRight());
	Mat imgLeftGray;
	Mat imgRightGray;
	cv::cvtColor(camReader->getCamLeft(), imgLeftGray, CV_BGR2GRAY);
	cv::cvtColor(camReader->getCamRight(), imgRightGray, CV_BGR2GRAY);

	GpuMat imgLeftGPU(imgLeftGray);
	GpuMat imgRightGPU(imgRightGray);

	GpuMat keyPointsLeftGPU, keyPointsRightGPU;
	GpuMat descriptorLeftGPU, descriptorRightGPU;

	surf(imgLeftGPU, cuda::GpuMat(), keyPointsLeftGPU, descriptorLeftGPU);
	surf(imgRightGPU, cuda::GpuMat(), keyPointsRightGPU, descriptorRightGPU);

	Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher();
	vector<vector<DMatch> > matches;
	matcher->knnMatch(descriptorLeftGPU, descriptorRightGPU, matches, 2);

	vector<KeyPoint> keyPointsLeft, keyPointsRight;

	surf.downloadKeypoints(keyPointsLeftGPU, keyPointsLeft);
	surf.downloadKeypoints(keyPointsRightGPU, keyPointsRight);

	vector<DMatch> good_matches;
	for(int k = 0; k < std::min(keyPointsLeft.size() - 1, matches.size()); ++k){
		if((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0)){
			good_matches.push_back((matches[k][0]));
		}
	}

	Mat img_matches;
	drawMatches(camReader->getCamLeft(), keyPointsLeft, camReader->getCamRight(), keyPointsRight, good_matches,
				img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);

	imshow("match!", img_matches);
	if(waitKey(1) >= 0) return;

}
