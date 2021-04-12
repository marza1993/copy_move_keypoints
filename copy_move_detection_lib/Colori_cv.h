#pragma once
#include <opencv2/imgproc.hpp>

class Colori
{

public:

	static const cv::Scalar RED;
	static const cv::Scalar GREEN;
	static const cv::Scalar BLUE;
	static const cv::Scalar CYAN;
	static const cv::Scalar YELLOW;
	static const cv::Scalar BLACK;
	static const cv::Scalar VIOLET;
	static const cv::Scalar DARK_GREEN;
	static const cv::Scalar WHITE;
	// ...
};


const cv::Scalar Colori::RED = cv::Scalar(0, 0, 255);
const cv::Scalar Colori::GREEN = cv::Scalar(0, 255, 0);
const cv::Scalar Colori::BLUE = cv::Scalar(255, 0, 0);
const cv::Scalar Colori::CYAN = cv::Scalar(255, 255, 0);
const cv::Scalar Colori::YELLOW = cv::Scalar(0, 255, 255);
const cv::Scalar Colori::BLACK = cv::Scalar(0, 0, 0);
const cv::Scalar Colori::VIOLET = cv::Scalar(255, 0, 187);
const cv::Scalar Colori::DARK_GREEN = cv::Scalar(56, 115, 53);
const cv::Scalar Colori::WHITE = cv::Scalar(255, 255, 255);


