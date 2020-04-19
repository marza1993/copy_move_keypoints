#include "GeometryUtility.h"
#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>


template <class T>
float GeometryUtility::pointVectorDistance(const T& A, const T& B, const T& P, T& POrt)
{
	// la retta passante per i punti A e B ha i seguenti parametri a,b,c:
	float a = B.y - A.y;
	float b = A.x - B.x;
	float c = -A.x * B.y + A.y * B.x;

	float norma = sqrt(pow(a, 2) + pow(b, 2));
	if (norma == 0)
	{
		// i punti A e B sono coincidenti: restituisco la distanza tra P e A
		POrt.x = A.x;
		POrt.y = A.y;
		return sqrt(pow(P.x - A.x, 2) + pow(P.y - A.y, 2));
	}

	// POrt è la proiezione ortogonale di P sulla retta passante per AB
	POrt.y = (pow(a, 2) * P.y - b * c - a * b * P.x) / pow(norma, 2);
	POrt.x = -(b * POrt.y + c) / a;

	// formula distanza punto-retta
	return abs(a * P.x + b * P.y + c) / norma;
}

template float GeometryUtility::pointVectorDistance<cv::Point2f>(const cv::Point2f& A, const cv::Point2f& B,
	const cv::Point2f& P, cv::Point2f& POrt);


template <class T>
float GeometryUtility::pointVectorDistance(const T& A, const T& B, const T& P)
{
	// la retta passante per i punti A e B ha i seguenti parametri a,b,c:
	float a = B.y - A.y;
	float b = A.x - B.x;
	float c = -A.x * B.y + A.y * B.x;

	float norma = sqrt(pow(a, 2) + pow(b, 2));
	if (norma == 0)
	{
		// i punti A e B sono coincidenti: restituisco la distanza tra P e A
		return sqrt(pow(P.x - A.x, 2) + pow(P.y - A.y, 2));
	}

	// formula distanza punto-retta
	return abs(a * P.x + b * P.y + c) / norma;
}

template float GeometryUtility::pointVectorDistance<cv::Point2f>(const cv::Point2f& A, const cv::Point2f& B,
	const cv::Point2f& P, cv::Point2f& POrt);


int GeometryUtility::findElbow(const std::vector<float>& X, const std::vector<float>& Y)
{

	int index_elbow = -1;

	// gestione errori: almeno 3 punti, altrimenti è una retta, e non può avere una forma "a gomito"
	if (X.size() < 3)
	{
		throw std::exception("la curva deve avere almeno 3 punti!");
	}

	if (X.size() != Y.size())
	{
		throw std::exception("i vettori X e Y devono avere lo stesso numero di elementi!");
	}

	// chiamo A il primo punto e B l'ultimo
	cv::Point2f A(X[0], Y[0]);
	cv::Point2f B(X.back(), Y.back());

	// calcolo le distanze tra ogni punto della curva e il vettore AB(che
	// connette il primo e l'ultimo punto della curva); il punto (X[i*],Y[i*]) per
	// cui questa distanza risulta maggiore è il punto "elbow".
	double distMax = 0;

	cv::Point2f Ptemp;
	for (size_t i = 0; i < X.size(); i++)
	{
		Ptemp.x = X[i];
		Ptemp.y = Y[i];
		double dist = pointVectorDistance(A, B, Ptemp);
		if (dist > distMax)
		{
			distMax = dist;
			index_elbow = i;
		}
	}

	return index_elbow;

}