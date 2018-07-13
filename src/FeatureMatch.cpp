#include "FeatureMatch.h"

FeatureMatch::FeatureMatch(){
	camReader = new CameraReader(5);
	minHessian = 10;
	surf.hessianThreshold = minHessian;
}

FeatureMatch::~FeatureMatch(){
	delete camReader;
}

void FeatureMatch::Matching(){
	camReader->Update();

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

void FeatureMatch::RANSACKNNMatch(){
	int best_inliers;
	camReader->Update();

	Mat imgLeftGray;
	Mat imgRightGray;
	vector<char> inlier_mask;
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

	vector<Point2f> left, right;

	for(int i = 0; i < good_matches.size(); ++i){
		left.push_back(keyPointsLeft[good_matches[i].queryIdx].pt);
		right.push_back(keyPointsRight[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(left, right, RANSAC, 3, inlier_mask);

	float best_H[9];
	int k = 0;
	if(H.rows > 2 && H.cols > 2){
		for(int y = 0; y < 3; y++) {
			for(int x = 0; x < 3; x++) {
				best_H[k] = H.at<double>(y,x);
				cout << best_H[k] << " ";
				k++;
			}
			cout << endl;
		}
	}

	cout << endl;
	for(int i = 0; i < left.size(); ++i){
		float x = best_H[0]*left[i].x + best_H[1]*left[i].y + best_H[2];
		float y = best_H[3]*left[i].x + best_H[4]*left[i].y + best_H[5];
		float z = best_H[6]*left[i].x + best_H[7]*left[i].y + best_H[8];

		x /= z;
		y /= z;

		float dist_sq = (right[i].x - x)*(right[i].x- x) + (right[i].y - y)*(right[i].y - y);

		if(dist_sq < INLIER_THRESHOLD) {
			best_inliers++;
		}
	}

	int h = imgLeftGray.rows + imgRightGray.rows;
	int w = max(imgLeftGray.cols, imgRightGray.cols);

	Mat result(h, w, CV_8UC3);

	for(int y=0; y < imgLeftGray.rows; y++) {
		for(int x=0; x < imgLeftGray.cols; x++) {
			result.at<Vec3b>(y,x)[0] = imgLeftGray.at<uchar>(y,x);
			result.at<Vec3b>(y,x)[1] = imgLeftGray.at<uchar>(y,x);
			result.at<Vec3b>(y,x)[2] = imgLeftGray.at<uchar>(y,x);
		}
	}

	for(int y=0; y < imgRightGray.rows; y++) {
		for(int x=0; x < imgRightGray.cols; x++) {
			result.at<Vec3b>(y+imgRightGray.rows,x)[0] = imgRightGray.at<uchar>(y,x);
			result.at<Vec3b>(y+imgRightGray.rows,x)[1] = imgRightGray.at<uchar>(y,x);
			result.at<Vec3b>(y+imgRightGray.rows,x)[2] = imgRightGray.at<uchar>(y,x);
		}
	}

	for(unsigned int i=0; i < inlier_mask.size(); i++) {
		if(inlier_mask[i]) {
			line(result, Point(left[i].x, left[i].y), Point(right[i].x, imgLeftGray.rows + right[i].y), CV_RGB(255,0,0));
		}
	}

	imshow("match!", result);
	if(waitKey(1) >= 0) return;
}



void FeatureMatch::KAZERANSAC(){
	const float inlier_threshold = 20.0f; // Distance threshold to identify inliers
	const float nn_match_ratio = 100.5f;
	Mat inlier_mask;

	camReader->Update();

	Mat img1 = camReader->getCamLeftGray();//imread("../data/graf1.png", IMREAD_GRAYSCALE);
	Mat img2 = camReader->getCamRightGray();//imread("../data/graf3.png", IMREAD_GRAYSCALE);

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);
	/*start from here */
	vector<KeyPoint> matched1, matched2, inliers1, inliers2;

	for(size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if(dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}

	vector<Point2f> left, right;
	vector<DMatch> good_matches;
	for(int i = 0; i < matched1.size(); ++i){
		left.push_back(matched1[i].pt);
		right.push_back(matched2[i].pt);
	}

	try{
		Mat homography = findHomography(left, right, RANSAC, 3, inlier_mask);
		for(unsigned i = 0; i < matched1.size(); i++) {
			Mat col = Mat::ones(3, 1, CV_64F);
			col.at<double>(0) = matched1[i].pt.x;
			col.at<double>(1) = matched1[i].pt.y;

			col = homography * col;
			col /= col.at<double>(2);
			double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
								pow(col.at<double>(1) - matched2[i].pt.y, 2));

			if(dist < inlier_threshold) {
				int new_i = static_cast<int>(inliers1.size());
				//cout << new_i << endl;
				inliers1.push_back(matched1[i]);
				inliers2.push_back(matched2[i]);
				good_matches.push_back(DMatch(new_i, new_i, 0));
				ransacLeftPoint.push_back(matched1[i].pt);
				ransacRightPoint.push_back(matched2[i].pt);				

				//cout << inliers1[i].size << " " << inliers2[i].size << endl;
			}
			GoodMatches = good_matches;
		}

	} catch(const std::exception& e) {

	}

	Mat res;
	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
	imshow("res.png", res);
	if(waitKey(1) >= 0) return;
}
