#include "CameraReader.h"

CameraReader::CameraReader(int fps){
	camL = 0;
	camR = 1;

	driver = CV_CAP_VFW;

	capL = new VideoCapture(camL + driver);
	capR = new VideoCapture(camR + driver);

	//check if we succeeded
	if (! (capL->isOpened() && capR->isOpened())) {
		cerr << "Unable to open the cameras" << endl;
		capL->release(); capR->release();
		return;
	}
	//SET SAME SETTINGS ON BOTH CAM
	Size frameSize(640, 480);

	capL->set(CV_CAP_PROP_FRAME_WIDTH, frameSize.width);
	capL->set(CV_CAP_PROP_FRAME_HEIGHT, frameSize.height);
	capL->set(CV_CAP_PROP_FPS, fps); //desired  FPS

	capR->set(CV_CAP_PROP_FRAME_WIDTH, frameSize.width);
	capR->set(CV_CAP_PROP_FRAME_HEIGHT, frameSize.height);
	capR->set(CV_CAP_PROP_FPS, fps); //desired  FPS

	// GRAB ONCE TO GET FRAME INFO (we'll lost this frames)

	if (!(capL->read(imgL) && capR->read(imgR))) {
		std::cerr << "Unable to grab from some CAM";
		capL->release(); capR->release();
		return;
	}
	if ((imgL.size() != imgR.size()) || (imgL.type() != imgR.type()))  {
		cerr << "The cameras uses different framesize or type" << endl;
		capL->release(); capR->release();
		return;
	}
}

CameraReader::~CameraReader(){
	capL->release();
	capR->release();
}

Mat CameraReader::getCamLeft(){
	return imgL;
}

Mat CameraReader::getCamRight(){
	return imgR;
}

Mat CameraReader::getCamLeftGray(){
	Mat grayImgL;
	cvtColor(imgL, grayImgL, CV_RGB2GRAY);
	return grayImgL;
}

Mat CameraReader::getCamRightGray(){
	Mat grayImgR;
	cvtColor(imgR, grayImgR, CV_RGB2GRAY);
	return grayImgR;
}

void CameraReader::Update(){

	if (! (capL->grab() && capR->grab()) ) {
		std::cerr << "Unable to grab from one or both cameras";
		return;
	}

	capL->retrieve(imgL);
	capR->retrieve(imgR);

	if (imgL.empty() || imgR.empty()) {
		std::cerr << "Empty frame received from some CAM !";
		return;
	}

	//cv::imshow("frame", imgL);
	//cv::imshow("frame2", imgR);

	//if (cv::waitKey(1) >= 0) return;

}
