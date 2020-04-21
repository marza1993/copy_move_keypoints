#pragma once

#include "UtilityDataStructures.h"
#include <vector>
#include <opencv2/imgproc.hpp>

class MyUtility
{

public:

	// Data una lista di coppie di numeri interi, elimina le coppie ripetute uguali o uguali a meno dell'ordine.
	// Ad es: (4,5) .. (5,4) => viene mantenuta solo la prima.
	// Restituisce anche la lista degli indici delle coppie rimaste dopo il filtraggio (cioè: listaCoppieNoDoppioni = listaCoppie[indiciRigheRimaste][:]).
	// NB: il vettore di output passato non può essere lo stesso di quello di input (infatti quest'ultimo è const).
	static void eliminaDoppioni(const std::vector<std::vector<int>>& listaCoppie, std::vector<std::vector<int>>& listaCoppieNoDoppioni,
								std::vector<int>& indiciRigheRimaste);



	static void writeKeyPoints(std::vector<cv::KeyPoint>& keypoints);


	static void writeDescriptors(cv::Mat& descriptors);


	static void writeKnnMatches(std::vector<std::vector<cv::DMatch>>& knn_matches);


	static void writeMatches(std::vector<KeyPointsMatch>& matches);



};

