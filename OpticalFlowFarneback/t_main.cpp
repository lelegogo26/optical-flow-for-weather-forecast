#include <cv.h>
#include <highgui.h>
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <cxcore.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <fstream>  
#include <windows.h>
#include <opencv2/imgproc/imgproc.hpp>  

using namespace cv;  
using namespace std; 

void main()
{
	// ∂¡»ÎÕº∆¨  
 	Mat ImgPre = imread("E:\\OpenCV\\workplace\\TestOpenCV\\Pre.png");
 	Mat ImgCur = imread("E:\\OpenCV\\workplace\\TestOpenCV\\Cur.png");

// 	imshow("PRE", ImgPre);
// 	imshow("Cur", ImgCur);
	Mat oflow;
	calcOpticalFlowFarneback(ImgPre, ImgCur, oflow, 0.5, 3, 20, 1, 5, 1.1, OPTFLOW_USE_INITIAL_FLOW);
}