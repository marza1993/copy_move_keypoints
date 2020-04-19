#pragma once
#include <opencv2/core.hpp>
#include "dbscan.h"


// label per indicare se un match è valido o meno (e per quale motivo non è considerato valido)
enum matchLabel
{
	VALID,
	MATCH_SAME_CLUSTER,
	MATCH_CLUSTER_PICCOLI,
	MATCH_OUTLIERS,
	UNKNOWN
};

struct KeyPointsMatch;



class ClusteredKeyPoint : IClusterPoint
{

	int clusterID;
	cv::KeyPoint* kp;

	// identificatore univoco
	int pointID;

	// riferimento ad un eventuale match di cui può essere parte (insieme ad un altro keypoint).
	// Utile nel caso si voglia risalire al match avendo un riferimento ad uno dei due punti che lo costituiscono
	KeyPointsMatch* parentMatch;

public:

	ClusteredKeyPoint(cv::KeyPoint* kp, int pointID)
	{
		this->kp = kp;
		this->clusterID = UNCLASSIFIED;
		this->parentMatch = nullptr;
		this->pointID = pointID;
	}

	cv::Point2f getPoint()
	{
		return kp->pt;
	}

	// imposta il riferimento al match di cui questo punto fa parte
	void setParentMatch(KeyPointsMatch* parentMatch)
	{
		this->parentMatch = parentMatch;
	}

	KeyPointsMatch* getParentMatch()
	{
		return parentMatch;
	}

	float operator [] (int i)
	{
		return component(i);
	}

	float component(int i) const
	{
		if (i > 1)
		{
			throw std::exception("l'indice passato sfora");
		}
		switch (i)
		{
		case 0:
			return kp->pt.x;
		case 1:
			return kp->pt.y;
		}
	}

	double distanceFrom(const IClusterPoint& otherPoint) 
	{
		ClusteredKeyPoint& other = (ClusteredKeyPoint&)otherPoint;
		return distanceFrom(other);
	}

	double distanceFrom(const ClusteredKeyPoint& otherPoint)
	{
		return sqrt(pow(component(0) - otherPoint.component(0), 2) + pow(component(1) - otherPoint.component(1), 2));
	}

	void setClusterID(int clusterID)
	{
		this->clusterID = clusterID;
	}


	int getClusterID()
	{
		return clusterID;
	}

	bool equals(const IClusterPoint& otherPoint, double tolerance = 0)
	{
		ClusteredKeyPoint& other = (ClusteredKeyPoint&)otherPoint;
		return equals(other, tolerance);
	}

	bool equals(const ClusteredKeyPoint& otherPoint, double tolerance = 0)
	{
		return (abs(component(0) - otherPoint.component(0)) <= tolerance && abs(component(1) - otherPoint.component(1)) <= tolerance);
	}
};



// struttura Wrapper che rappresenta un match tra due keypoints (es: SIFT, SURF).
// possiede i riferimenti ai punti stessi e ai loro descrittori (oltre che alla distanza tra essi).
// i Keypoints implementano l'interface IClusterPoint, in modo che possano essere elaborati dal dbscan, e che abbiano la label del cluster
struct KeyPointsMatch
{

	// primo keypoint della coppia (con le reference mi dava mille menate, quindi lo faccio con i puntatori)
	ClusteredKeyPoint* kp1;
	// secondo keypoint della coppia
	ClusteredKeyPoint* kp2;

	// descrittore del primo keypoint: è un vettore di dimensione 1xN (con N=128 di solito)
	cv::Mat* descriptorP1;
	// descrittore del secondo keypoint
	cv::Mat* descriptorP2;

	// distanza tra i due descrittori
	float descriptorDistance;

	// distanza spaziale tra i due keypoints (coordinate x,y)
	float spatialDistance;

	// label che descrive la validità o meno del match. Nel caso in cui questo non sia valido specifica anche il motivo
	matchLabel label;

	int matchID;

	// costruttore
	KeyPointsMatch(ClusteredKeyPoint* kp1, ClusteredKeyPoint* kp2, cv::Mat* descriptorP1, cv::Mat* descriptorP2, float descriptorDistance, int matchID)
		: kp1(kp1), kp2(kp2), descriptorP1(descriptorP1), descriptorP2(descriptorP2),
		descriptorDistance(descriptorDistance), label(matchLabel::UNKNOWN), matchID(matchID)
	{
		this->spatialDistance = kp1->distanceFrom(*kp2);
	}

};
