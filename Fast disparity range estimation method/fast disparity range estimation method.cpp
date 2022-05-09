//---------------------------------------------------------------------------------
/*# fast disparity range estimation method

This implementation is inspired by the paper£º Edge - Preserving Stereo Matching Using Minimum Spanning Tree
Link to paper : https://ieeexplore.ieee.org/document/8930525
This repository only contains implementation of fast disparity range estimation method :
Dependencies
opencv - C++ v3.2.0
opencv - C++ contrib v3.2.0
Visual Studio2017
Citation
If you find the code or datasets helpful in your research, please cite :
C.Zhang, C.He, Z.Chen, W.Liu, M.Li, and J.Wu, ''Edge - preserving stereo matching using minimum spanning tree,'' IEEE Access, vol. 7, pp. 177909¨C177921, 2019.
//------------------------------------------------------------------------------------------------*/
#include <opencv2/features2d/features2d.hpp> 
#include "opencv2/xfeatures2d/nonfree.hpp" 
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
//Census transform
float census(int hL, int wL, int hR, int wR, Size censusWin, Mat leftImage, Mat rightImage, int row, int col)
{
	float dist = 0;
	int colorrefl, colorrefl1, colorrefl2, colorrefR, colorrefR1, colorrefR2, colorLP, colorLP1, colorLP2, colorRP, colorRP1, colorRP2;
	colorrefl = leftImage.at<Vec3b>(hL, wL)[0];
	colorrefl1 = leftImage.at<Vec3b>(hL, wL)[1];
	colorrefl2 = leftImage.at<Vec3b>(hL, wL)[2];

	colorrefR = rightImage.at<Vec3b>(hR, wR)[0];
	colorrefR1 = rightImage.at<Vec3b>(hR, wR)[1];
	colorrefR2 = rightImage.at<Vec3b>(hR, wR)[2];

	for (int h = -censusWin.height / 2; h <= censusWin.height / 2; ++h)
	{
		for (int w = -censusWin.width / 2; w <= censusWin.width / 2; ++w)
		{
			int hl = 0, hr = 0;
			int wl = 0, wr = 0;

			hl = h + hL;
			hr = h + hR;

			wl = w + wL;
			wr = w + wR;

			if (hl < 0) hl = 0;
			if (hr < 0) hr = 0;

			if (wl < 0) wl = 0;
			if (wr < 0) wr = 0;

			if (row <= hl) hl = row - 1;
			if (row <= hr) hr = row - 1;

			if (col <= wl) wl = col - 1;
			if (col <= wr) wr = col - 1;


			colorLP = leftImage.at<Vec3b>(hl, wl)[0];
			colorLP1 = leftImage.at<Vec3b>(hl, wl)[1];
			colorLP2 = leftImage.at<Vec3b>(hl, wl)[2];

			colorRP = rightImage.at<Vec3b>(hr, wr)[0];
			colorRP1 = rightImage.at<Vec3b>(hr, wr)[1];
			colorRP2 = rightImage.at<Vec3b>(hr, wr)[2];

			// bool diff = (colorLP[color] < colorRefL[color]) ^ (colorRP[color] < colorRefR[color]);
			bool diff = (colorLP - colorrefl) * (colorRP - colorrefR) < 0;
			dist += (diff) ? 1 : 0;

			diff = (colorLP1 - colorrefl1) * (colorRP1 - colorrefR1) < 0;
			dist += (diff) ? 1 : 0;

			diff = (colorLP2 - colorrefl2) * (colorRP2 - colorrefR2) < 0;
			dist += (diff) ? 1 : 0;


		}
	}
	//cout << dist << endl;
	return dist;
}

void SURF_disparity_range_estimation(Mat img_1, Mat img_2, Mat img_11, Mat img_22, int& max, int& min, int truncation1, int truncation2, Mat& img_matches)
{
	int minHessian = 1;
	Ptr<xfeatures2d::SURF> surfDetector = xfeatures2d::SURF::create(minHessian);
	vector<KeyPoint> keypoints_1, keypoints_2;
	surfDetector->detect(img_1, keypoints_1);
	surfDetector->detect(img_2, keypoints_2);

	Mat descriptors_1, descriptors_2;
	Mat imageDesc1, imageDesc2;
	surfDetector->compute(img_1, keypoints_1, descriptors_1);
	surfDetector->compute(img_2, keypoints_2, descriptors_2);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	double max_dist = 0; double min_dist = 1000;
	int minn;
	Size window(5, 5);
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		//if (matches[i].distance < 15 * min_dist)
		{
			int test, test1, xx, xx1, yy, yy1, dd, dd1, ss, hd;
			test = matches[i].queryIdx;
			test1 = matches[i].trainIdx;
			xx = keypoints_1[test].pt.y;
			xx1 = keypoints_2[test1].pt.y;
			yy = keypoints_1[test].pt.x;
			yy1 = keypoints_2[test1].pt.x;

			ss = census(xx, yy, xx1, yy1, window, img_11, img_22, img_11.rows, img_11.cols);
			hd = abs(img_11.at<uchar>(xx, yy) - img_22.at<uchar>(xx1, yy1));
			if ((abs(keypoints_1[test].pt.y - keypoints_2[test1].pt.y) < 1) && ((keypoints_1[test].pt.x - keypoints_2[test1].pt.x) >= 0) && (ss < truncation1) && (hd < truncation2) && ((keypoints_1[test].pt.x - keypoints_2[test1].pt.x) < 400))
				good_matches.push_back(matches[i]);
		}
	}
	int min_ = 10000, max_ = 0, ccc;
	for (int i = 0; i < good_matches.size(); i++)
	{
		int zc, pp, ppt;
		pp = good_matches[i].queryIdx;
		ppt = good_matches[i].trainIdx;
		zc = (keypoints_1[pp].pt.x - keypoints_2[ppt].pt.x);
		if (zc < min_ && zc != 0)
		{
			min_ = zc;
			ccc = i;
		}
		if (zc > max_)
		{
			max_ = zc;
		}
	}
	max = max_;
	min = min_;


}
void AKAZE_disparity_range_estimation(Mat img_1, Mat img_2, Mat img_11, Mat img_22, int& max1, int& min1, int truncation1, int truncation2, Mat& img_matches)
{
	Ptr<AKAZE> akaze = AKAZE::create();
	vector<KeyPoint> keypoints_1, keypoints_2;
	akaze->detect(img_1, keypoints_1);
	akaze->detect(img_2, keypoints_2);
	Mat descriptors_1, descriptors_2;
	akaze->compute(img_1, keypoints_1, descriptors_1);
	akaze->compute(img_2, keypoints_2, descriptors_2);
	descriptors_1.convertTo(descriptors_1, CV_32F);
	descriptors_2.convertTo(descriptors_2, CV_32F);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	descriptors_1.convertTo(descriptors_1, CV_32F);
	descriptors_2.convertTo(descriptors_2, CV_32F);
	matcher.match(descriptors_1, descriptors_2, matches);
	double max_dist = 0; double min_dist = 1000;
	int minn;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;

		if (dist < min_dist) { min_dist = dist, minn = i; }
		if (dist > max_dist) max_dist = dist;
	}
	Size window(5, 5);
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		//if (matches[i].distance < 15 * min_dist)
		{
			int test, test1, xx, xx1, yy, yy1, dd, dd1, ss, hd;
			test = matches[i].queryIdx;
			test1 = matches[i].trainIdx;
			xx = keypoints_1[test].pt.y;
			xx1 = keypoints_2[test1].pt.y;
			yy = keypoints_1[test].pt.x;
			yy1 = keypoints_2[test1].pt.x;

			ss = census(xx, yy, xx1, yy1, window, img_11, img_22, img_11.rows, img_11.cols);
			hd = abs(img_11.at<uchar>(xx, yy) - img_22.at<uchar>(xx1, yy1));
			if ((abs(keypoints_1[test].pt.y - keypoints_2[test1].pt.y) < 1) && ((keypoints_1[test].pt.x - keypoints_2[test1].pt.x) >= 0) && (ss < truncation1) && (hd < truncation2) && ((keypoints_1[test].pt.x - keypoints_2[test1].pt.x) < 400))
				good_matches.push_back(matches[i]);
		}
	}
	int min_ = 10000, max_ = 0, ccc;
	for (int i = 0; i < good_matches.size(); i++)
	{
		int zc, pp, ppt;
		pp = good_matches[i].queryIdx;
		ppt = good_matches[i].trainIdx;
		zc = (keypoints_1[pp].pt.x - keypoints_2[ppt].pt.x);
		if (zc < min_ && zc != 0)
		{
			min_ = zc;
			ccc = i;
		}
		if (zc > max_)
		{
			max_ = zc;
		}
	}
	if ((max1 / max_) > 2 || (max_ / max1) > 2)
		max1 = min(max_, max1);
	if (max1 < max_)
		max1 = max_;
	if (min1 > min_)
		min1 = min_;
}


//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	system("color 4F");

	//load left and right image
	Mat img_1 = imread("im0.png", 1);
	Mat img_2 = imread("im1.png", 1);
	Mat img_11 = imread("im0.png", 0);
	Mat img_22 = imread("im1.png", 0);



	int max, min, max1, min1;
	int T = 11, k = 25;
	Mat img_matches;
	SURF_disparity_range_estimation(img_1, img_2, img_11, img_22, max, min, T, k, img_matches);
	AKAZE_disparity_range_estimation(img_1, img_2, img_11, img_22, max, min, T, k, img_matches);

	cout << "The estimated disparity range is " << min << " to " << max << endl;
	waitKey(0);
	return 0;
}

