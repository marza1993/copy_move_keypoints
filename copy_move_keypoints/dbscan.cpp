#include "dbscan.h"
#include <iostream>
#include <algorithm>
#include <opencv2/xfeatures2d.hpp>
#include "GeometryUtility.h"

using namespace std;
using namespace cv;

void DBSCAN::run()
{
	if (m_epsilon == -1)
	{
		throw exception(" non è stato impostato il parametro espilon!");
	}

	int clusterID = 1;
	for (auto point : m_points)
	{
		if (point->getClusterID() == UNCLASSIFIED)
		{
			if (expandCluster(*point, clusterID))
			{
				// ho formato un nuovo cluster: aggiungo la label alla lista
				clusterLabels.push_back(clusterID);
				clusterID++;
			}
		}
	}

	// aggiungo i punti contrassegnati come NOISE alla mappa dei cluster.
	// NB: posso farlo solo alla fine, facendo scorrere tutti i punti, perchè i punti che in un certo momento
	// sono contrassegnati come NOISE poi possono cambiare durante l'algoritmo (vedere articolo).
	for (auto point : m_points)
	{
		if (point->getClusterID() == NOISE)
		{
			foundClustersMap[NOISE].push_back(point);
		}
	}
}


bool DBSCAN::expandCluster(IClusterPoint& point, int clusterID)
{
	vector<IClusterPoint*> seeds = getEpsNeighborhood(point);
	if (seeds.size() < m_minPoints)
	{
		// no core point. Per il momento è impostato a noise (poi potrebbe cambiare..)
		point.setClusterID(NOISE);
		return false;
	}

	// all points in clusterSeeds are density reachable from point
	// imposto il clusterID di tutti i punti del neighborhood con il clusterID passato
	for (auto seed : seeds)
	{
		seed->setClusterID(clusterID);
		// aggiungo il punto al cluster con questo clusterID
		foundClustersMap[clusterID].push_back(seed);
	}

	// elimino dalla lista dei seed il core point (quello passato alla funzione)
	seeds.erase(std::remove(seeds.begin(), seeds.end(), &point));

	while (seeds.size() != 0)
	{
		IClusterPoint* currentP = seeds.front();
		vector<IClusterPoint*> result = getEpsNeighborhood(*currentP);

		if (result.size() >= m_minPoints)
		{
			for (auto resultP : result)
			{
				if (resultP->getClusterID() == UNCLASSIFIED || resultP->getClusterID() == NOISE)
				{
					if (resultP->getClusterID() == UNCLASSIFIED)
					{
						seeds.push_back(resultP);
					}
					resultP->setClusterID(clusterID);

					// aggiungo il punto al cluster con questo clusterID
					foundClustersMap[clusterID].push_back(resultP);
				}
			}
		}

		seeds.erase(std::remove(seeds.begin(), seeds.end(), currentP));
	}
	return true;
}


vector<IClusterPoint*> DBSCAN::getEpsNeighborhood(IClusterPoint& point)
{
	vector<IClusterPoint*> neighborhood;
	neighborhood.reserve(m_points.size());
	for (auto p : m_points)
	{
		if (p->distanceFrom(point) <= m_epsilon)
		{
			neighborhood.push_back(p);
		}
	}
	return neighborhood;
}


float DBSCAN::findOptimalEps()
{

	// L'idea consiste nel calcolare le distanze tra ogni punto e i k più 
	// vicini ad esso(con k = minPts) e ordinarle in ordine non descrescente.
	// Il valore di eps ottimale è quello per cui si ha una
	// flessione("elbow") del grafico delle distanze.

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	

	// salvo le coordinate x,y di tutti i punti in una matrice di dimensione: m_points.size() x 2
	Mat puntiXY(m_points.size(), 2, DataType<float>::type);
	float* buffer = (float*)puntiXY.ptr<float>(0);
	for (size_t i = 0; i < m_points.size(); i++)
	{
		buffer[i * puntiXY.cols + 0] = m_points[i]->component(0);
		buffer[i * puntiXY.cols + 1] = m_points[i]->component(1);
	}

	// calcolo la distanza tra ogni punto e i k più vicini (si ipotizza di lavorare con la curva k - dist, 
	// con k = minPts, secondo quanto suggerito nell'articolo sul DBSCAN).
	vector<vector<cv::DMatch>> knn_matches;
	try
	{
		matcher->knnMatch(puntiXY, puntiXY, knn_matches, m_minPoints);
	}
	catch (const exception & e)
	{
		// TODO
		m_epsilon = 0.1;
		return m_epsilon;
	}
	
	if (knn_matches.size() == 0)
	{
		m_epsilon = 0.1;
		return m_epsilon;
	}

	// ordino la quarta colonna delle distanze e la salvo(la prima è
	// quella dei punti con loro stessi, perciò è tutta di zeri)
	if (knn_matches[0].size() == 2)
	{
		m_epsilon = knn_matches[0][1].distance;
		return m_epsilon;
	}

	std::vector<float> distanze;
	distanze.reserve(knn_matches.size());
	if (knn_matches[0].size() >= 4)
	{
		for (auto& knn_match : knn_matches)
		{
			distanze.push_back(knn_match[3].distance);
		}
	}
	else if (knn_matches[0].size() == 3)
	{
		for (auto& knn_match : knn_matches)
		{
			distanze.push_back(knn_match[2].distance);
		}
	}

	// ordino in ordine non descrescente le distanze dai punti
	std::sort(distanze.begin(), distanze.end());
	vector<float> X;
	X.reserve(distanze.size());
	for (size_t i = 0; i < distanze.size(); i++)
	{
		X.push_back(i + 1);
	}
	int indiceElbow = GeometryUtility::findElbow(X, distanze);
	m_epsilon = distanze[indiceElbow];
	return m_epsilon;
}



std::vector<IClusterPoint*>& DBSCAN::getPointsInCluster(int clusterID)
{
	if (foundClustersMap.count(clusterID) == 0)
	{
		throw exception("non esiste un cluster con la label richiesta!");
	}
	return foundClustersMap[clusterID];
}

// restituisce la lista delle label create per i cluster individuati (compreso NOISE), es: -1,1,2,3,..,Num_cluster
std::vector<int>& DBSCAN::getClusterLabels()
{
	return clusterLabels;
}

std::vector<int> DBSCAN::getClusterIDs()
{
	std::vector<int> clusterIDs;
	clusterIDs.reserve(m_points.size());
	for (auto& p : m_points)
	{
		clusterIDs.push_back(p->getClusterID());
	}
	return clusterIDs;
}




