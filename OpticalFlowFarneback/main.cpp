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
#include <opencv2/video/tracking.hpp> 	 // 必须添加


using namespace cv;  
using namespace std; 

#define UNKNOWN_FLOW_THRESH 1e9  
#define PI 3.1415926

string int2str( int val )
{
	ostringstream out;
	out<<val;
	return out.str();
}

void saveMat(Mat &Motion, string s)  //保存运动矢量场
{
	s += ".txt";
	FILE *pOut = fopen(s.c_str(), "w+");
	for (int i = 0; i<Motion.rows; i++){
		for (int j = 0; j<Motion.cols; j++){
			fprintf(pOut, "%lf", Motion.at<float>(i, j));
			if (j == Motion.cols - 1) fprintf(pOut, "\n");
			else fprintf(pOut, " ");
		}
	}
}

void makecolorwheel(vector<Scalar> &colorwheel)  
{  
	int RY = 15;  
	int YG = 6;  
	int GC = 4;  
	int CB = 11;  
	int BM = 13;  
	int MR = 6;  

	int i;  

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  

void motionToColor(Mat flow, Mat &color)  
{  
	if (color.empty())  
		color.create(flow.rows, flow.cols, CV_8UC3);  

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())  
		makecolorwheel(colorwheel);  

	// determine motion range:  
	float maxrad = -1;  

	// Find max flow to normalize fx and fy  
	for (int i= 0; i < flow.rows; ++i)   
	{  
		for (int j = 0; j < flow.cols; ++j)   
		{  
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
			float fx = flow_at_point[0];  
			float fy = flow_at_point[1];  
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
				continue;  
			float rad = sqrt(fx * fx + fy * fy);  
			maxrad = maxrad > rad ? maxrad : rad;  
		}  
	}  

	for (int i= 0; i < flow.rows; ++i)   
	{  
		for (int j = 0; j < flow.cols; ++j)   
		{  
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  

			float fx = flow_at_point[0] / maxrad;  
			float fy = flow_at_point[1] / maxrad;  
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
			{  
				data[0] = data[1] = data[2] = 0;  
				continue;  
			}  
			float rad = sqrt(fx * fx + fy * fy);  

			float angle = atan2(-fy, -fx) / CV_PI;  
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
			int k0 = (int)fk;  
			int k1 = (k0 + 1) % colorwheel.size();  
			float f = fk - k0;  
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)   
			{  
				float col0 = colorwheel[k0][b] / 255.0;  
				float col1 = colorwheel[k1][b] / 255.0;  
				float col = (1 - f) * col0 + f * col1;  
				if (rad <= 1)  
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else  
					col *= .75; // out of range  
				data[2 - b] = (int)(255.0 * col);  
			}  
		}  
	}  
}  


inline bool isFlowCorrect(Point2f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.y) < 1e9;
}



static Vec3b computeColor(float fx, float fy)
{
	static bool first = true;

	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow
	//  than between yellow and green)
	const int RY = 15;
	const int YG = 6;
	const int GC = 4;
	const int CB = 11;
	const int BM = 13;
	const int MR = 6;
	const int NCOLS = RY + YG + GC + CB + BM + MR;
	static Vec3i colorWheel[NCOLS];

	if (first){
		int k = 0;

		for (int i = 0; i < RY; ++i, ++k)
			colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

		for (int i = 0; i < YG; ++i, ++k)
			colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

		for (int i = 0; i < GC; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

		for (int i = 0; i < CB; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

		for (int i = 0; i < BM; ++i, ++k)
			colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

		for (int i = 0; i < MR; ++i, ++k)
			colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

		first = false;
	}

	const float rad = sqrt(fx * fx + fy * fy);
	const float a = atan2(-fy, -fx) / (float)CV_PI;

	const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
	const int k0 = static_cast<int>(fk);
	const int k1 = (k0 + 1) % NCOLS;
	const float f = fk - k0;

	Vec3b pix;

	for (int b = 0; b < 3; b++)
	{
		const float col0 = colorWheel[k0][b] / 255.f;
		const float col1 = colorWheel[k1][b] / 255.f;

		float col = (1 - f) * col0 + f * col1;

		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range

		pix[2 - b] = static_cast<uchar>(255.f * col);
	}

	return pix;
}

//void drawOpticalFlow(Mat & flow, Mat& flowImage, int stride, float scale, Scalar& color)
//{
//	dst.create(flow.size(), CV_8UC3);
//	dst.setTo(Scalar::all(0));
//
//	// determine motion range:
//	float maxrad = maxmotion;
//
//	if (maxmotion <= 0)
//	{
//		maxrad = 1;
//		for (int y = 0; y < flow.rows; ++y)
//		{
//			for (int x = 0; x < flow.cols; ++x)
//			{
//				Point2f u = flow(y, x);
//
//				if (!isFlowCorrect(u))
//					continue;
//
//				maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
//			}
//		}
//	}
//
//	for (int y = 0; y < flow.rows; ++y)
//	{
//		for (int x = 0; x < flow.cols; ++x)
//		{
//			Point2f u = flow(y, x);
//
//			if (isFlowCorrect(u))
//				dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
//		}
//	}
//}

void OpticalFlowQualityControl(Mat &u, Mat &v)   // 运动矢量场的质量控制
{
	Mat n = Mat::zeros(u.rows, u.cols, CV_32F);;
	for (int i=0; i<u.rows; i++)
	{
		for (int j=0; j<u.cols; j++)
		{
			n.at<float>(i,j) = fastAtan2(v.at<float>(i,j), u.at<float>(i,j));   // 反正切
		}		

	}
	for (int i=1; i<u.rows-1; i++)
	{
		for (int j=1; j<u.cols-1; j++)
		{
			if (abs(n.at<float>(i,j))-abs(n.at<float>(i-1,j-1)+n.at<float>(i-1,j)+n.at<float>(i-1,j+1)+n.at<float>(i,j-1)+
				n.at<float>(i,j+1)+n.at<float>(i+1,j-1)+n.at<float>(i+1,j)+n.at<float>(i+1,j+1))/8 > PI/180*25)
			{
				u.at<float>(i,j) = (u.at<float>(i-1,j-1)+u.at<float>(i-1,j)+u.at<float>(i-1,j+1)+u.at<float>(i,j-1)+
					u.at<float>(i,j+1)+u.at<float>(i+1,j-1)+u.at<float>(i+1,j)+u.at<float>(i+1,j+1))/8;  //中值
				v.at<float>(i,j) = (v.at<float>(i-1,j-1)+v.at<float>(i-1,j)+v.at<float>(i-1,j+1)+v.at<float>(i,j-1)+
					v.at<float>(i,j+1)+v.at<float>(i+1,j-1)+v.at<float>(i+1,j)+v.at<float>(i+1,j+1))/8;
			}
			if((abs(u.at<float>(i,j))-(u.at<float>(i-1,j-1)+u.at<float>(i-1,j)+u.at<float>(i-1,j+1)+u.at<float>(i,j-1)+
				u.at<float>(i,j+1)+u.at<float>(i+1,j-1)+u.at<float>(i+1,j)+u.at<float>(i+1,j+1))/8)*(200000*2/u.rows/360) > 5)
			{
				u.at<float>(i,j) = (u.at<float>(i-1,j-1)+u.at<float>(i-1,j)+u.at<float>(i-1,j+1)+u.at<float>(i,j-1)+
					u.at<float>(i,j+1)+u.at<float>(i+1,j-1)+u.at<float>(i+1,j)+u.at<float>(i+1,j+1))/8; 
			}
			if((abs(v.at<float>(i,j))-(v.at<float>(i-1,j-1)+v.at<float>(i-1,j)+v.at<float>(i-1,j+1)+v.at<float>(i,j-1)+
				v.at<float>(i,j+1)+v.at<float>(i+1,j-1)+v.at<float>(i+1,j)+v.at<float>(i+1,j+1))/8)*(200000*2/v.rows/360) > 5)
			{
				v.at<float>(i,j) = (v.at<float>(i-1,j-1)+v.at<float>(i-1,j)+v.at<float>(i-1,j+1)+v.at<float>(i,j-1)+
					v.at<float>(i,j+1)+v.at<float>(i+1,j-1)+v.at<float>(i+1,j)+v.at<float>(i+1,j+1))/8; 
			}
			if(abs(u.at<float>(i,j))*(200000*2/u.rows/360) > 10)
			{
				u.at<float>(i,j) = (u.at<float>(i-1,j-1)+u.at<float>(i-1,j)+u.at<float>(i-1,j+1)+u.at<float>(i,j-1)+
					u.at<float>(i,j+1)+u.at<float>(i+1,j-1)+u.at<float>(i+1,j)+u.at<float>(i+1,j+1))/8; 
			}
			if(abs(v.at<float>(i,j))*(200000*2/v.rows/360) > 10)
			{
				v.at<float>(i,j) = (v.at<float>(i-1,j-1)+v.at<float>(i-1,j)+v.at<float>(i-1,j+1)+v.at<float>(i,j-1)+
					v.at<float>(i,j+1)+v.at<float>(i+1,j-1)+v.at<float>(i+1,j)+v.at<float>(i+1,j+1))/8; 
			}
			if(abs(sqrt(pow(u.at<float>(i,j),2)+pow(v.at<float>(i,j),2))-(sqrt(pow(u.at<float>(i-1,j-1),2)+
				pow(v.at<float>(i-1,j-1),2))+sqrt(pow(u.at<float>(i-1,j),2)+pow(v.at<float>(i-1,j),2))+
				sqrt(pow(u.at<float>(i-1,j+1),2)+pow(v.at<float>(i-1,j+1),2))+sqrt(pow(u.at<float>(i,j-1),2)+
				pow(v.at<float>(i,j-1),2))+sqrt(pow(u.at<float>(i,j+1),2)+pow(v.at<float>(i,j+1),2))+
				sqrt(pow(u.at<float>(i+1,j-1),2)+pow(v.at<float>(i+1,j-1),2))+sqrt(pow(u.at<float>(i+1,j),2)+
				pow(v.at<float>(i+1,j),2))+sqrt(pow(u.at<float>(i+1,j+1),2)+pow(v.at<float>(i+1,j+1),2))/8))*(200000*2/v.rows/360)>15)
			{
				u.at<float>(i,j) = (u.at<float>(i-1,j-1)+u.at<float>(i-1,j)+u.at<float>(i-1,j+1)+u.at<float>(i,j-1)+
					u.at<float>(i,j+1)+u.at<float>(i+1,j-1)+u.at<float>(i+1,j)+u.at<float>(i+1,j+1))/8;  
				v.at<float>(i,j) = (v.at<float>(i-1,j-1)+v.at<float>(i-1,j)+v.at<float>(i-1,j+1)+v.at<float>(i,j-1)+
					v.at<float>(i,j+1)+v.at<float>(i+1,j-1)+v.at<float>(i+1,j)+v.at<float>(i+1,j+1))/8;		
			}
		}
	}
}

float sumMat(Mat inputImg)  // 矩阵元素累加
{
	float sum = 0.0;
	int rowNumber = inputImg.rows;
	int colNumber = inputImg.cols * inputImg.channels();
	for (int i = 0; i < rowNumber;i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			sum = inputImg.at<float>(i,j) + sum;
		}
	}
	return sum;
}

void UpdateMotionVectorField(Mat &alpha, Mat &beta, Mat u, Mat v) 
{
	Mat e = Mat::zeros(9, 1, CV_32F);
	Mat d = Mat::zeros(9, 1, CV_32F);
	int countE, countD;
	for (int i=1; i<u.rows-1; i++)
	{
		for (int j=1; j<u.cols-1; j++)
		{
			e = (u.at<float>(i-1,j-1),u.at<float>(i-1,j),u.at<float>(i-1,j+1),u.at<float>(i,j),
				u.at<float>(i,j-1),u.at<float>(i,j+1),u.at<float>(i+1,j-1),u.at<float>(i+1,j),u.at<float>(i+1,j+1));
			d = (v.at<float>(i-1,j-1),v.at<float>(i-1,j),v.at<float>(i-1,j+1),v.at<float>(i,j),
				v.at<float>(i,j-1),v.at<float>(i,j+1),v.at<float>(i+1,j-1),v.at<float>(i+1,j),v.at<float>(i+1,j+1));
			countE = countNonZero(e);
			countD = countNonZero(d);
			if (countE != 0) // 统计数组不等于0的个数
				alpha.at<float>(i, j) = sumMat(e)/countE;
			if (countD != 0)     
				beta.at<float>(i, j) = sumMat(d)/countD;
		}
	}
}
// 外推预报
void ExtrapolationForecast(Mat &NewImage, Mat &BlurThresholdGrayPre, Mat &alpha, Mat &beta)  
{
	NewImage = Mat::zeros(BlurThresholdGrayPre.rows, BlurThresholdGrayPre.cols, CV_32F);

	int i1, j1;
	float k, m;
	for (int i=1; i<NewImage.rows-1; i++)
	{
		for (int j=1; j<NewImage.cols-1; j++)
		{
			if (alpha.at<float>(i,j)>= 0.0 && beta.at<float>(i,j)>= 0.0)
			{
				i1 = i-floor(alpha.at<float>(i,j))-1;
				j1 = j-floor(beta.at<float>(i,j))-1;
				k = 1-(alpha.at<float>(i,j)-floor(alpha.at<float>(i,j)));
				m = 1-(beta.at<float>(i,j)-floor(beta.at<float>(i,j)));
				if (i1 > BlurThresholdGrayPre.rows || j1 > BlurThresholdGrayPre.cols || i1 <= -1 || j1 <= -1)
					NewImage.at<float>(i,j) = 255;
				else
					NewImage.at<float>(i,j) = (1.0-k)*(1.0-m)*BlurThresholdGrayPre.at<float>(i1,j1)+(1.0-k)*m*BlurThresholdGrayPre.at<float>(i1,j1+1)+
					k*(1.0-m)*BlurThresholdGrayPre.at<float>(i1+1,j1)+k*m*BlurThresholdGrayPre.at<float>(i1+1,j1+1);
			}
			else if (alpha.at<float>(i,j)< 0.0 && beta.at<float>(i,j)> 0.0)
			{
				i1 = i-ceil(alpha.at<float>(i,j));
				j1 = j-floor(beta.at<float>(i,j))-1;
				k = abs(alpha.at<float>(i,j)-ceil(alpha.at<float>(i,j)));
				m = 1-(beta.at<float>(i,j)-floor(beta.at<float>(i,j)));
				if (i1 > BlurThresholdGrayPre.rows || j1 > BlurThresholdGrayPre.cols || i1 <= -1 || j1 <= -1)
					NewImage.at<float>(i,j) = 255;
				else
					NewImage.at<float>(i,j) = (1.0-k)*(1.0-m)*BlurThresholdGrayPre.at<float>(i1,j1)+(1.0-k)*m*BlurThresholdGrayPre.at<float>(i1,j1+1)+
					k*(1.0-m)*BlurThresholdGrayPre.at<float>(i1+1,j1)+k*m*BlurThresholdGrayPre.at<float>(i1+1,j1+1);
			}
			else if (alpha.at<float>(i,j)< 0.0 && beta.at<float>(i,j)<= 0.0)
			{
				i1 = i-ceil(alpha.at<float>(i,j));
				j1 = j-ceil(beta.at<float>(i,j));
				k = abs(alpha.at<float>(i,j)-ceil(alpha.at<float>(i,j)));
				m = abs(beta.at<float>(i,j)-ceil(beta.at<float>(i,j)));
				if (i1 > BlurThresholdGrayPre.rows || j1 > BlurThresholdGrayPre.cols || i1 <= -1 || j1 <= -1)
					NewImage.at<float>(i,j) = 255;
				else
					NewImage.at<float>(i,j) = (1.0-k)*(1.0-m)*BlurThresholdGrayPre.at<float>(i1,j1)+(1.0-k)*m*BlurThresholdGrayPre.at<float>(i1,j1+1)+
					k*(1.0-m)*BlurThresholdGrayPre.at<float>(i1+1,j1)+k*m*BlurThresholdGrayPre.at<float>(i1+1,j1+1);
			}
			else if (alpha.at<float>(i,j)>= 0.0 && beta.at<float>(i,j)< 0.0)
			{
				i1 = i-floor(alpha.at<float>(i,j))-1;
				j1 = j-ceil(beta.at<float>(i,j));
				k = 1-(alpha.at<float>(i,j)-floor(alpha.at<float>(i,j)));
				m = abs(beta.at<float>(i,j)-ceil(beta.at<float>(i,j)));
				if (i1 > BlurThresholdGrayPre.rows || j1 > BlurThresholdGrayPre.cols || i1 <= -1 || j1 <= -1)
					NewImage.at<float>(i,j) = 255;
				else
					NewImage.at<float>(i,j) = (1.0-k)*(1.0-m)*BlurThresholdGrayPre.at<float>(i1,j1)+(1.0-k)*m*BlurThresholdGrayPre.at<float>(i1,j1+1)+
					k*(1.0-m)*BlurThresholdGrayPre.at<float>(i1+1,j1)+k*m*BlurThresholdGrayPre.at<float>(i1+1,j1+1);
			}
		}
	}
}


void Gray2RGB(Mat &NewColorImage, Mat NewImage1)
{
	for (int i=0; i<NewImage1.rows; i++)
	{
		uchar* data = NewImage1.ptr<uchar>(i);
		for (int j=0; j<NewImage1.cols*NewImage1.channels() ; j++)
		{

			if (data[j] >=0 && data[j] <26)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 0;
				NewColorImage.at<Vec3b>(i, j)[1] = 0;
				NewColorImage.at<Vec3b>(i, j)[0] = 3;
			}
			else if (data[j] >=26 && data[j] <56)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 127;
				NewColorImage.at<Vec3b>(i, j)[1] = 194;
				NewColorImage.at<Vec3b>(i, j)[0] = 229;
			}
			else if (data[j] >=56 && data[j] <66)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 0;
				NewColorImage.at<Vec3b>(i, j)[1] = 174;
				NewColorImage.at<Vec3b>(i, j)[0] = 165;
			}
			else if (data[j] >=66 && data[j] <76)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 198;
				NewColorImage.at<Vec3b>(i, j)[1] = 195;
				NewColorImage.at<Vec3b>(i, j)[0] = 255;
			}
			else if (data[j] >=76 && data[j] <86)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 123;
				NewColorImage.at<Vec3b>(i, j)[1] = 113;
				NewColorImage.at<Vec3b>(i, j)[0] = 239;
			}
			else if (data[j] >=86 && data[j] <96)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 24;
				NewColorImage.at<Vec3b>(i, j)[1] = 36;
				NewColorImage.at<Vec3b>(i, j)[0] = 214;
			}
			else if (data[j] >=96 && data[j] <106)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 165;
				NewColorImage.at<Vec3b>(i, j)[1] = 255;
				NewColorImage.at<Vec3b>(i, j)[0] = 173;
			}
			else if (data[j] >=106 && data[j] <116)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 0;
				NewColorImage.at<Vec3b>(i, j)[1] = 235;
				NewColorImage.at<Vec3b>(i, j)[0] = 0;
			}
			else if (data[j] >=116 && data[j] <126)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 16;
				NewColorImage.at<Vec3b>(i, j)[1] = 146;
				NewColorImage.at<Vec3b>(i, j)[0] = 24;
			}
			else if (data[j] >=126 && data[j] <136)
			{

				NewColorImage.at<Vec3b>(i, j)[2] = 255;
				NewColorImage.at<Vec3b>(i, j)[1] = 247;
				NewColorImage.at<Vec3b>(i, j)[0] = 99;
			}
			else if (data[j] >=136 && data[j] <146)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 206;
				NewColorImage.at<Vec3b>(i, j)[1] = 203;
				NewColorImage.at<Vec3b>(i, j)[0] = 0;
			}
			else if (data[j] >=146 && data[j] <156)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 140;
				NewColorImage.at<Vec3b>(i, j)[1] = 142;
				NewColorImage.at<Vec3b>(i, j)[0] = 0;
			}
			else if (data[j] >=156 && data[j] <166)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 255;
				NewColorImage.at<Vec3b>(i, j)[1] = 174;
				NewColorImage.at<Vec3b>(i, j)[0] = 173;
			}
			else if (data[j] >=166 && data[j] <176)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 255;
				NewColorImage.at<Vec3b>(i, j)[1] = 101;
				NewColorImage.at<Vec3b>(i, j)[0] = 90;
			}
			else if (data[j] >=176 && data[j] <186)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 239;
				NewColorImage.at<Vec3b>(i, j)[1] = 0;
				NewColorImage.at<Vec3b>(i, j)[0] = 49;
			}
			else if (data[j] >=186 && data[j] <196)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 214;
				NewColorImage.at<Vec3b>(i, j)[1] = 142;
				NewColorImage.at<Vec3b>(i, j)[0] = 255;
			}
			else if (data[j] >=196 && data[j] <=255)
			{
				NewColorImage.at<Vec3b>(i, j)[2] = 173;
				NewColorImage.at<Vec3b>(i, j)[1] = 36;
				NewColorImage.at<Vec3b>(i, j)[0] = 255;
			}
		}
	}
}

void main()
{
	long start_time = GetTickCount();
	// 读入图片  						   E:\OpenCV\workplace\OpticalFlowFarneback
 	Mat ImgPre = imread("E:\\OpenCV\\workplace\\OpticalFlowFarneback\\20170909_151010.00.010.000_R3.png", CV_8U);	 // 直接读入是三通道
 	Mat ImgCur = imread("E:\\OpenCV\\workplace\\OpticalFlowFarneback\\20170909_151616.00.010.000_R3.png", CV_8U);

	ImgPre.convertTo(ImgPre,CV_32F);
	ImgCur.convertTo(ImgCur,CV_32F);	
	Mat  BlurThresholdGrayPre, BlurThresholdGrayCur;
	medianBlur(ImgPre, BlurThresholdGrayPre, 3);
	medianBlur(ImgCur, BlurThresholdGrayCur, 3);

	
	Mat u = Mat::zeros(ImgPre.rows, ImgPre.cols, CV_32F);
    Mat v = Mat::zeros(ImgPre.rows, ImgPre.cols, CV_32F);
	Mat oflow;
	calcOpticalFlowFarneback(BlurThresholdGrayPre, BlurThresholdGrayCur, oflow, 0.5, 3, 20, 3, 7, 0.6, 0); //OPTFLOW_FARNEBACK_GAUSSIAN

	float maxrad = 1;  

 	for (int i= 0; i < oflow.rows; i++)   
 	{  
 		for (int j = 0; j < oflow.cols; j++)   
 		{  
 			Vec2f flow_at_point = oflow.at<Vec2f>(i, j);  //flow:x,y
 			u.at<float>(i, j) = flow_at_point[0]/ maxrad;  
 			v.at<float>(i, j) = flow_at_point[1]/ maxrad;  
 		}  
 	}  

	OpticalFlowQualityControl(u, v); // 光流场的质量控制		
	Mat alpha = Mat::zeros(u.rows, u.cols, CV_32F);
	Mat beta = Mat::zeros(u.rows, u.cols, CV_32F);
	Mat NewImage = Mat::zeros(u.rows, u.cols, CV_32F);
	Mat NewImage1 = Mat::zeros(u.rows, u.cols, CV_32F);
	Mat NewColorImage = Mat::zeros(u.rows, u.cols, CV_8UC3);
	alpha = v;
	beta = u;
	// 外推预报
	for (int ForecastTime = 0; ForecastTime<=5 ;ForecastTime++)
	{
		ExtrapolationForecast(NewImage, BlurThresholdGrayPre, alpha, beta);  //外推预报
		if (ForecastTime>0)
		{
		// 保存图片
			string ImageAddres = int2str(ForecastTime)+".png";
			string ImageAddres1 = "Color"+int2str(ForecastTime)+".png";
			NewImage.convertTo(NewImage1,CV_8U);
			imwrite(ImageAddres, NewImage1);
			Gray2RGB(NewColorImage,NewImage1);
			imwrite(ImageAddres1, NewColorImage);  
		}
		if (ForecastTime<5)
		{
			Mat alpha = Mat::zeros(u.rows, u.cols, CV_32F);
			Mat beta = Mat::zeros(u.rows, u.cols, CV_32F);
			UpdateMotionVectorField(alpha, beta, u, v);   //更新光流场
			NewImage.copyTo(BlurThresholdGrayPre);
			u = alpha;
			v = beta;
		}
	}
	long end_time = GetTickCount(); //获取此程序段开始执行时间
	cout << "程序段运行时间：" << (end_time - start_time) << "ms!" << endl; //差值即执行时间
}