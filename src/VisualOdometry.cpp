#include "VisualOdometry.h"

PoseCalculator::PoseCalculator(){
	VisualOdometryStereo::parameters param;
	param.calib.f = 811;
	param.calib.cu = 305.98;
	param.calib.cv = 287.83;
	param.base = 0.06;
	param.inlier_threshold = 1.5;
	param.ransac_iters = 100;
	param.bucket.max_features = 50;

	param.match.nms_n = 3;
	param.match.nms_tau = 50;
	param.match.match_binsize = 50;
	param.match.match_radius = 200;
	param.match.match_disp_tolerance = 2;
	param.match.outlier_disp_tolerance = 100;
	param.match.outlier_flow_tolerance = 100;
	param.match.multi_stage = 1;
	param.match.half_resolution = 1;
	param.match.refinement = 1;

	feature = new FeatureMatch();
	//cameraReader = new CameraReader(5);
	viso = new VisualOdometryStereo(param);
	pose = Matrix::eye(4);

	for(int i = 0; i < 1000; ++i){
		Point2f start(0,0);
		DMatch startMatch(0,0,0);
		previousRawLeftPoints.push_back(start);
		previousRawRightPoints.push_back(start);
		previousMatches.push_back(startMatch);
	}

	dir = "2010_03_09_drive_0019";

}

PoseCalculator::~PoseCalculator(){
	delete feature;
	delete viso;
}

void PoseCalculator::CalculatePose(){
	for(; ; ){
		try{
			feature->camReader->Update();

			int width = feature->camReader->getCamLeftGray().cols;
			int height = feature->camReader->getCamLeftGray().rows;

			uint8_t *left_img_data = feature->camReader->getCamLeftGray().data;
			uint8_t *right_img_data = feature->camReader->getCamRightGray().data;

			Mat leftImage = Mat(height, width, CV_8U, left_img_data);
			Mat rightImage = Mat(height, width, CV_8U, right_img_data);

			int32_t dims[] = {width, height, width};
			if(viso->process(feature->camReader->getCamLeftGray().data, feature->camReader->getCamRightGray().data, dims)){
				pose = pose * Matrix::inv(viso->getMotion());

				double num_matches = viso->getNumberOfMatches();
				double num_inliers = viso->getNumberOfInliers();

				cout << ", Matches: " << num_matches;
				cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
				cout << pose << endl << endl;

				float sy = sqrt(pose.val[0][0] * pose.val[0][0] +  pose.val[1][0] * pose.val[1][0]);
				float roll = atan2(pose.val[2][1] , pose.val[2][2]) * 180/3.14;
				float pitch = atan2(-pose.val[2][0], sy)  * 180/3.14;
				float yaw = atan2(pose.val[1][0], pose.val[0][0])  * 180/3.14;
				//cout << roll << "\t" << pitch << "\t" << yaw << endl;
				ShowMatched(leftImage, rightImage, viso->getMatches(), viso->getInlierIndices());
			} else {
				//cout << " ... failed" << endl;
			}

		} catch(...){

		}
	}

}

void PoseCalculator::CalculatePoseRANSAC(){
	feature->KAZERANSAC();
	vector<Matcher::p_match> good_match;
	vector<Point2f> rawLeftPoints = feature->getLeftImageFeatures();
	vector<Point2f> rawRightPoints = feature->getRightImageFeatures();
	vector<DMatch> matches = feature->getGoodMatch();
	for(int i = 0; i < rawLeftPoints.size(); ++i){

		single_good_match.u1c = rawLeftPoints[i].x;
		single_good_match.v1c = rawLeftPoints[i].y;
		single_good_match.u2c = rawRightPoints[i].x;
		single_good_match.v2c = rawRightPoints[i].y;
		single_good_match.i1c = static_cast<int32_t>(matches[i].queryIdx);
		single_good_match.i2c = static_cast<int32_t>(matches[i].trainIdx);

		single_good_match.u1p = previousRawLeftPoints[i].x;
		single_good_match.v1p = previousRawLeftPoints[i].y;
		single_good_match.u2p = previousRawRightPoints[i].x;
		single_good_match.v2p = previousRawRightPoints[i].y;
		single_good_match.i1p = static_cast<int32_t>(previousMatches[i].queryIdx);
		single_good_match.i2p = static_cast<int32_t>(previousMatches[i].trainIdx);

		good_match.push_back(single_good_match);
		//cout << good_match[i].u1c << " " << good_match[i].u2c << " " << good_match[i].u1p << " " << good_match[i].u2p << endl;
	}

	if(viso->process(good_match)){
		pose = pose * Matrix::inv(viso->getMotion());

		double num_matches = viso->getNumberOfMatches();
		double num_inliers = viso->getNumberOfInliers();

		cout << ", Matches: " << num_matches;
		cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
		cout << pose << endl << endl;
		//cout << viso->getInlierIndices().size() << endl;
		//ShowMatched(feature->camReader->getCamLeftGray(), feature->camReader->getCamRightGray(), viso->getMatches(), viso->getInlierIndices());
	} else {
		//cout << " ... failed" << endl;
	}

	copy(matches.begin(), matches.end(), back_inserter(previousMatches));
	copy(rawLeftPoints.begin(), rawLeftPoints.end(), back_inserter(previousRawLeftPoints));
	copy(rawRightPoints.begin(), rawRightPoints.end(), back_inserter(previousRawRightPoints));

}

void PoseCalculator::ShowMatched(Mat left_image, Mat right_image, const vector<Matcher::p_match>& matches, const vector<int32_t>& inlier_indices){
	vector<DMatch> cv_matched;
	vector<KeyPoint> kpLeft, kpRight;
	for(int i = 0; i < inlier_indices.size(); ++i){
		Matcher::p_match match = matches[inlier_indices[i]];
		DMatch cv_matched_temp(i, i, 0);

		cv_matched.push_back(cv_matched_temp);

		KeyPoint keyLeftTemp, keyRightTemp;

		keyLeftTemp.pt.x = match.u1c;
		keyLeftTemp.pt.y = match.v1c;

		kpLeft.push_back(keyLeftTemp);

		keyRightTemp.pt.x = match.u2c;
		keyRightTemp.pt.y = match.v2c;

		//cout << match.v1c - match.v2c << endl;

		kpRight.push_back(keyRightTemp);
	}

	Mat result1;
	Mat result2;
	Mat result;
	//drawKeypoints(left_image, kpLeft,result1);
	//drawKeypoints(right_image, kpRight,result2);
	drawMatches(left_image, kpLeft, right_image, kpRight, cv_matched, result);
	imshow("show", result);
	//imshow("show", result1);
	//imshow("show", result2);
	if(waitKey(1) >= 0) return;
}
