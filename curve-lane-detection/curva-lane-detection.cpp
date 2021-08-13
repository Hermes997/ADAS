#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>


#include <sstream>
#include <fstream>
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;

void on_luminance_change1(int pos, void* userdata);
void on_luminance_change2(int pos, void* userdata);
void on_saturation_change1(int pos, void* userdata);
void on_saturation_change2(int pos, void* userdata);
void on_canny_change1(int pos, void* userdata);
void on_canny_change2(int pos, void* userdata);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

int temp1 = 100;
int temp2 = 255;
int temp3 = 100;
int temp4 = 255;
int temp5 = 100;
int temp6 = 200;


int main(void){

	

	VideoCapture cap1("driving8.mp4");

	float data1[] = { 13613.69404427889*0.375, 0, 182.6215289088512 * 0.375,0, 14639.52311652638 * 0.375, 333.1058777908088 * 0.375,0, 0, 1};
	float data2[] = { -52.39363722291178, 911.8463386908078, 0.02802985675847747, 1.404366773057897, -12006.84148733081 };

	Mat cameraMatrix(3,3,CV_32FC1,data1);
	Mat distanceCoefficients(5, 1, CV_32FC1, data2);

	cout << cameraMatrix << endl << endl;
	cout << distanceCoefficients << endl;
	


	if (!cap1.isOpened()) {
		printf("Can't open the camera");
		return -1;
	}

	Mat img1, img1_gray, img1_hls, img1_canny, camimg, img1_a, img1_b, warpimg1, warpimg1_y, warpimg1_c, warpimg1_f, warpimg1_s, warpimg1_k;

	vector<Mat> img1_hls_splited(3);
	vector<Point2f> corners(4), warpcorners(4), corners2(4), warpcorners2(4);
	Mat trans, trans2;
	Size interest(900*0.375, 600*0.375);
	Rect interest_s(270 * 0.375, 480 * 0.375, 660 * 0.375, 130 * 0.375);
	int c1 = 0;

	while (1) {

		cap1 >> camimg;
		if (camimg.empty())
		{
			printf("empty image");
			return 0;
		}

		undistort(camimg, img1_a, cameraMatrix, distanceCoefficients);

		//line(img1_a, Point(318 * 0.375, 647 * 0.375), Point(506 * 0.375, 470 * 0.375), Scalar(255, 0, 0), 1, 8, 0);
		//line(img1_a, Point(870 * 0.375, 647 * 0.375), Point(697 * 0.375, 470 * 0.375), Scalar(255, 0, 0), 1, 8, 0);
		


		
		img1 = img1_a(interest_s);

		corners[0] = (Point((int)interest_s.width * 0.285, 0));
		corners[1] = (Point((int)interest_s.width * 0.722, 0));
		corners[2] = (Point(0, (int)interest_s.height));
		corners[3] = (Point((int)interest_s.width, (int)interest_s.height));

		warpcorners[0] = (Point(0, 0));
		warpcorners[1] = (Point(interest.width, 0));
		warpcorners[2] = (Point(0, interest.height));
		warpcorners[3] = (Point(interest.width, interest.height));

		warpimg1 = (interest, img1.type());

		trans = getPerspectiveTransform(corners, warpcorners);

		warpPerspective(img1, warpimg1, trans, interest);
		warpimg1_y = warpimg1;
		cvtColor(warpimg1, img1_gray, COLOR_BGR2GRAY);
		cvtColor(warpimg1, img1_hls, CV_BGR2HLS);


		/*split(img1_hls, img1_hls_splited);
	



		//imshow("img1_hls", img1_hls);
		//imshow("img1_h", img1_hls_splited[0]);

		
		namedWindow("img1_l");
		namedWindow("img1_s");
		namedWindow("img1_canny");

		createTrackbar("lumi1", "img1_l", 0, 255, on_luminance_change1, (void*)&img1_hls_splited[1]);
		createTrackbar("lumi2", "img1_l", 0, 255, on_luminance_change2, (void*)&img1_hls_splited[1]);
		createTrackbar("satu1", "img1_s", 0, 255, on_saturation_change1, (void*)&img1_hls_splited[0]);
		createTrackbar("satu2", "img1_s", 0, 255, on_saturation_change2, (void*)&img1_hls_splited[0]);
		createTrackbar("can1", "img1_canny", 0, 255, on_canny_change1, (void*)&img1_gray);
		createTrackbar("can2", "img1_canny", 0, 255, on_canny_change2, (void*)&img1_gray);

		
		setTrackbarPos("lumi1", "img1_l", temp1);
		setTrackbarPos("lumi2", "img1_l", temp2);
		setTrackbarPos("satu1", "img1_s", temp3);
		setTrackbarPos("satu2", "img1_s", temp4);
		setTrackbarPos("can1", "img1_canny", temp5);
		setTrackbarPos("can2", "img1_canny", temp6);

		threshold(img1_hls_splited[1], img1_hls_splited[1], getTrackbarPos("lumi1", "img1_l"), getTrackbarPos("lumi2", "img1_l"), THRESH_BINARY);
		threshold(img1_hls_splited[0], img1_hls_splited[0], getTrackbarPos("satu1", "img1_s"), getTrackbarPos("satu2", "img1_s"), THRESH_BINARY);
		Canny(img1_gray, img1_canny, getTrackbarPos("can1", "img1_canny"), getTrackbarPos("can2", "img1_canny"));

		
		imshow("img1_l", img1_hls_splited[1]);
		imshow("img1_s", img1_hls_splited[0]);
		imshow("img1_canny", img1_canny);*/

		setMouseCallback("warpimg1", CallBackFunc, (void*)&warpimg1);
		setMouseCallback("img1", CallBackFunc, (void*)&img1);

		
		cvtColor(warpimg1, warpimg1_c, CV_BGR2HLS);
		

		
		
		for (int i = 0; i <= warpimg1_c.cols - 1; i++){
			for (int j = 0; j <= warpimg1_c.rows - 1; j++) {
				if (!(((int)warpimg1_c.at<Vec3b>(j, i)[0] > 10 && (int)warpimg1_c.at<Vec3b>(j, i)[0] < 30 && (int)warpimg1_c.at<Vec3b>(j, i)[1] > 80 && (int)warpimg1_c.at<Vec3b>(j, i)[1] < 130)
					|| ((int)warpimg1_c.at<Vec3b>(j, i)[1] > 88 && (int)warpimg1_c.at<Vec3b>(j, i)[1] < 250))) {
					warpimg1.at<Vec3b>(j, i) = 0;
					
				}
			}
		}
		
		for (int i = warpimg1.cols * 0.25 ; i <= warpimg1.cols * 0.75 ; i++) {
			for (int j = 0; j <= warpimg1.rows - 1; j++) {
				warpimg1.at<Vec3b>(j, i) = 0;
			}
		}

		
		
		cvtColor(warpimg1, warpimg1, CV_HLS2BGR);
		cvtColor(warpimg1, warpimg1, CV_BGR2GRAY);
		threshold(warpimg1, warpimg1, 80, 255, THRESH_BINARY);
		cvtColor(warpimg1, warpimg1_f, CV_GRAY2BGR);

		int i, j, k;

		for (j = 30; j <= warpimg1.rows - 1; j=j+30) {
			for (i = (int)(warpimg1.cols * 0.25); i >= (int)(warpimg1.cols * 0.1); i--) {
				if ((int)(warpimg1.at<uchar>(j, i)) >= 10) {
					rectangle(warpimg1_f, Rect(i - 15, j - 15, 30, 30), Scalar(0, 0, 255), 1, 8, 0);
					break;
				}
			}

			for (k = (int)(warpimg1.cols * 0.75); k <= (int)(warpimg1.cols * 0.9); k++) {
				if ((int)(warpimg1.at<uchar>(j, k)) >= 255) {
					rectangle(warpimg1_f, Rect(k - 15, j - 15, 30, 30), Scalar(0, 0, 255), 1, 8, 0);
					break;
				}
			}
			if (i != (int)(warpimg1.cols * 0.1) + 1 && k != (int)(warpimg1.cols * 0.9) + 1) {
				rectangle(warpimg1_f, Rect(Point(i, j), Point(k, j + 30)), Scalar(0, 50, 0), CV_FILLED, 8, 0);
			}
			

		}

		/*corners2[0] = (Point((int)interest_s.width * 0.285, 0));
		corners2[1] = (Point((int)interest_s.width * 0.722, 0));
		corners2[2] = (Point(0, (int)interest_s.height));
		corners2[3] = (Point((int)interest_s.width, (int)interest_s.height));

		warpcorners2[0] = (Point(0, 0));
		warpcorners2[1] = (Point(interest.width, 0));
		warpcorners2[2] = (Point(0, interest.height));
		warpcorners2[3] = (Point(interest.width, interest.height));*/

		warpPerspective(warpimg1_f, warpimg1_s, trans.inv(), Size(img1.cols, img1.rows));
		
		warpimg1_k = img1 + warpimg1_s;
		imshow("warpimg1", warpimg1);
		imshow("warpimg1_f", warpimg1_f);
		imshow("img1", img1);
		imshow("warpimg1_s", warpimg1_s);
		imshow("warpimg1_k", warpimg1_k);
		imshow("img1_a", img1_a);
		//imshow("warpimg1_y", warpimg1_y);



		if (cv::waitKey(10) == 1)
			break;

	}

	return 0;
}

void on_luminance_change1(int pos, void* userdata) {
	Mat src = *(Mat*)userdata;

	temp1 = pos;

}


void on_luminance_change2(int pos, void* userdata) {
	Mat src = *(Mat*)userdata;

	temp2 = pos;

}

void on_saturation_change1(int pos, void* userdata) {
	Mat src = *(Mat*)userdata;

	temp3 = pos;

}

void on_saturation_change2(int pos, void* userdata) {
	Mat src = *(Mat*)userdata;

	temp4 = pos;

}

void on_canny_change1(int pos, void* userdata) {
	Mat src = *(Mat*)userdata;

	temp5 = pos;

}

void on_canny_change2(int pos, void* userdata) {
	Mat src = *(Mat*)userdata;

	temp6 = pos;

}

void CallBackFunc(int event, int y, int x, int flags, void* userdata) {
	
	if (event == EVENT_LBUTTONDOWN) {
		Mat src = *(Mat*)userdata;
		Mat hls;
		cvtColor(src, hls, CV_BGR2HLS);
		cout << "ÁÂÇ¥ :  x = " << x << ", y = " << y << endl << endl;
		cout << "»ö±ò :  b = " << (int)src.at<Vec3b>(x ,y)[0] << ", g = " << (int)src.at<Vec3b>(x, y)[1] << ", r = " << (int)src.at<Vec3b>(x, y)[2] << endl << endl;
		cout << "»ö±ò :  h = " << (int)hls.at<Vec3b>(x, y)[0] << ", l = " << (int)hls.at<Vec3b>(x, y)[1] << ", s = " << (int)hls.at<Vec3b>(x, y)[2] << endl << endl << endl << endl;
	}
}