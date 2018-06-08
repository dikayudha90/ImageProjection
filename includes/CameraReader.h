#include <stdio.h>

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;

class CameraReader{
	public:
		CameraReader(int fps);
		~CameraReader();
                void Update();
                Mat getCamLeft();
                Mat getCamRight();
	private:
                // SELECT CAM ID FOR YOUR CAMS
                int camL;
                int camR;
                // CHOOSE YOUR BEST DRIVER
                int driver;// = CV_CAP_VFW; //use DSHOW or VFW or MSMF
                // OPEN THE DEVICES
                VideoCapture *capL;//(camL + driver);
                VideoCapture *capR;//(camR+ driver);
                Mat imgL, imgR;
};
