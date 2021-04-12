#include <iostream>
#include "copy_move_detection_lib.h"

#include "opencv2/highgui.hpp"
#include <string>



std::string nomeFile = "DSC_0812tamp1.jpg";
//string nomeFile = "DSC_1535tamp133.jpg";
//string nomeFile = "CRW_4809_scale.jpg";
//string nomeFile = "CRW_4853tamp132.jpg";
//string nomeFile = "DSCN45tamp131.jpg";
//string nomeFile = "DSCN45tamp25.jpg";
//string nomeFile = "P1000231_scale.jpg";

//string nomeFile = "CRW_4809_scale.jpg";
//string nomeFile = "sony_61_scale.jpg";
//string nomeFile = "DSCN45tamp1.jpg";
//string nomeFile = "CRW_4815_scale.jpg";

//std::string nomeFile = "nikon_7_scale.jpg";


const std::string DATA_SET_PATH = "D:\\dottorato\\copy_move\\MICC-F220\\";

const std::string WINDOW_NAME = "prova_copy_move";

int main()
{
    std::cout << "************\n driver copy_move_detection_lib \n**************" << std::endl;


    const cv::Mat img = cv::imread(DATA_SET_PATH + nomeFile, cv::IMREAD_GRAYSCALE);


	const uint sogliaSift = 0;
	const uint minPuntiIntorno = 3;
	const float sogliaLowe = 0.43;
	const float eps = 50;
	const float sogliaDescInCluster = 0.19;

	int resultForgedOrNot = -1;
	copy_move_det_lib::KeyPointsMatchStruct* foundMatches;
	uchar* outputImg;
	uint N_found_matches = -1;
	uint outImg_W = -1;
	uint outImg_H = -1;
	int getOutputImg = 1;

	copy_move_det_lib::SIFT_copy_move_detection(img.data, img.cols, img.rows, sogliaSift, minPuntiIntorno,
												sogliaLowe, eps, sogliaDescInCluster, &resultForgedOrNot,
												&foundMatches, &N_found_matches, getOutputImg, &outputImg, &outImg_W, &outImg_H);



	// visualizzo i risultati della detection
	if (getOutputImg)
	{
		const cv::Mat outMat = cv::Mat(cv::Size(outImg_W, outImg_H), CV_8UC3, outputImg);

		cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
		cv::resizeWindow(WINDOW_NAME, 800, 600);
		cv::imshow(WINDOW_NAME, outMat);
		cv::waitKey(0);
	}




	for (int i = 0; i < N_found_matches; i++)
	{
		std::cout << "match " << i << "-> p1 = (" << foundMatches[i].p1->x << ", " << foundMatches[i].p1->y << "), " <<
			"p2 = (" << foundMatches[i].p2->x << ", " << foundMatches[i].p2->y << "), " <<
			"is valid: " << (foundMatches[i].isValid ? "yes" : "no") << std::endl;
	}


	copy_move_det_lib::SIFT_copy_move_free_mem(&foundMatches, N_found_matches, &outputImg);


    system("pause");
}
