#include "pch.h"

#include "copy_move_detection_lib.h"
#include "CopyMoveDetectorSIFT.h"



void copy_move_det_lib::SIFT_copy_move_detection(const uchar* input_img, const uint img_W, const uint img_H, 
												 const uint sogliaSIFT, const uint minPuntiIntorno,
												 const float sogliaLowe, const float eps, const float sogliaDescInCluster,
												 int* resultForgedOrNot, KeyPointsMatchStruct** foundMatches, uint* N_found_matches, 
												 const int getOutputImg, uchar** outputImg,
	                                             uint* output_img_W, uint* output_img_H)
{
	// creo la matrice opencv di input
	const cv::Mat inputMat = cv::Mat(cv::Size(img_W, img_H), CV_8U, (uchar *) input_img);

	// creo l'oggetto per la detection e lancio la detection
	CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe, eps, sogliaDescInCluster);
	detector.detect(inputMat, (bool) getOutputImg);

	// risultato forged/autentica
	(*resultForgedOrNot) = detector.getIsForged() ? 1 : 0;

	if (getOutputImg)
	{
		cv::Mat outputImgMat;
		detector.getOuputImg(outputImgMat);

		(*output_img_W) = outputImgMat.cols;
		(*output_img_H) = outputImgMat.rows;

		// copio il buffer dei dati
		(*outputImg) = new uchar[outputImgMat.rows * outputImgMat.cols * 3];
		std::copy(outputImgMat.data, outputImgMat.data + (img_H * img_W * 3), (*outputImg));
	}
	else
	{
		(*outputImg) = nullptr;
	}


	// restituisco anche i match individuati
	std::vector<KeyPointsMatch*>& matches = detector.getMatchesPointers();
	(*foundMatches) = new KeyPointsMatchStruct[matches.size()];
	(*N_found_matches) = matches.size();

	for (int i = 0; i < matches.size(); i++)
	{
		KeyPointsMatchStruct& keyPointsMatchStruct = (*foundMatches)[i];
		// controllo se il match è valido
		if (matches[i]->kp1->getClusterID() >= 0)
		{
			keyPointsMatchStruct.isValid = 1;
		}
		else
		{
			keyPointsMatchStruct.isValid = 0;
		}

		keyPointsMatchStruct.p1 = new Point2DStruct();
		keyPointsMatchStruct.p1->x = matches[i]->kp1->component(0);
		keyPointsMatchStruct.p1->y = matches[i]->kp1->component(1);

		keyPointsMatchStruct.p2 = new Point2DStruct();
		keyPointsMatchStruct.p2->x = matches[i]->kp2->component(0);
		keyPointsMatchStruct.p2->y = matches[i]->kp2->component(1);

		keyPointsMatchStruct.descriptorsDistance = -1; // TODO..

	}
}



void copy_move_det_lib::SIFT_copy_move_free_mem(KeyPointsMatchStruct** foundMatches, uint N_matches, uchar** outputImg)
{
	for (int i = 0; i < N_matches; i++)
	{
		delete (*foundMatches)[i].p1;
		delete (*foundMatches)[i].p2;
	}

	delete[](*foundMatches);
	delete[](*outputImg);
	(*foundMatches) = nullptr;
	(*outputImg) = nullptr;
}

