#include <iostream>
#include <iomanip>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "CopyMoveDetectorSIFT.h"
#include "MyUtility.h"
#include "Colori_cv.h"


using namespace std;
using namespace cv;
using namespace xfeatures2d;

std::mutex CopyMoveDetectorSIFT::mtxCuda;

enum nonValidPointLabel
{
	NOISE_POINT = NOISE,
	POINT_SAME_CLUSTER = -2,
	POINT_SMALL_CLUSTER = -3,
};

CopyMoveDetectorSIFT::CopyMoveDetectorSIFT() : CopyMoveDetectorSIFT(100, 3, 0.6)
{
	// costruttore con parametri di default
}


CopyMoveDetectorSIFT::CopyMoveDetectorSIFT(const unsigned int soglia_SIFT, const unsigned int minPtsNeighb, const float soglia_Lowe,
	const float eps, const float sogliaDescInCluster)
{
	this->soglia_SIFT = soglia_SIFT;
	this->minPtsNeighb = minPtsNeighb;
	this->soglia_Lowe = soglia_Lowe;
	this->N_clusterValidi = 0;
	this->N_medioElementiCluster = 0;
	this->dbscan_eps = eps;
	this->sogliaDescInCluster = sogliaDescInCluster;
}

// genera l'immagine di output con i risultati dell'elaborazione.
void CopyMoveDetectorSIFT::drawOutputImg()
{

	if (!isDetectionOk)
	{
		cout << "la forgery detection non è andata a buon fine! Impossibile ottenere l'immagine elaborata!";
		return;
	}

	cv::Mat tempOutput;

	// inizializzo l'immagine di output con l'immagine di input (convertendola a 3 canali)
	cv::cvtColor(*inputImg, tempOutput, cv::COLOR_GRAY2BGR);

	
	std::unordered_map<int, cv::Scalar> coloriClusterValidi;
	for (size_t i = 0; i < labelClusterValidi.size(); i++)
	{
		coloriClusterValidi[labelClusterValidi[i]] = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
	}


	// Disegno i match con linee che collegano i corrispondenti punti.
	// Disegno anche i punti relativi a match scartati, con diversi marker/colori a seconda del motivo dello scarto
	for (auto& match : matchesPointers)
	{
		switch (match->kp1->getClusterID())
		{

		case nonValidPointLabel::NOISE_POINT:

			drawMarker(tempOutput, match->kp1->getPoint(), Colori::RED, cv::MARKER_TILTED_CROSS, 8, 1, cv::LINE_AA);
			drawMarker(tempOutput, match->kp2->getPoint(), Colori::RED, cv::MARKER_TILTED_CROSS, 8, 1, cv::LINE_AA);
			break;

		case nonValidPointLabel::POINT_SAME_CLUSTER:

			circle(tempOutput, match->kp1->getPoint(), 5, Colori::CYAN, 1, cv::LINE_AA);
			circle(tempOutput, match->kp2->getPoint(), 5, Colori::CYAN, 1, cv::LINE_AA);
			break;

		case nonValidPointLabel::POINT_SMALL_CLUSTER:

			circle(tempOutput, match->kp1->getPoint(), 5, Colori::BLUE, 1, cv::LINE_AA);
			circle(tempOutput, match->kp2->getPoint(), 5, Colori::BLUE, 1, cv::LINE_AA);
			break;

		default:

			circle(tempOutput, match->kp1->getPoint(), 5, coloriClusterValidi[match->kp1->getClusterID()], cv::FILLED, cv::LINE_AA);
			circle(tempOutput, match->kp2->getPoint(), 5, coloriClusterValidi[match->kp2->getClusterID()], cv::FILLED, cv::LINE_AA);

			circle(tempOutput, match->kp1->getPoint(), 5, Colori::GREEN, 1, cv::LINE_AA);
			circle(tempOutput, match->kp2->getPoint(), 5, Colori::GREEN, 1, cv::LINE_AA);

			line(tempOutput, match->kp1->getPoint(), match->kp2->getPoint(), Colori::YELLOW, 1, cv::LINE_AA);
			break;

		}
	}
	
	unsigned int dimTestoInfoElab = 30;
	cv::Mat temp(tempOutput.rows + dimTestoInfoElab, tempOutput.cols, tempOutput.type(), Colori::WHITE);
	outputImg = temp;

	tempOutput.copyTo(outputImg(cv::Rect(0, dimTestoInfoElab - 1, tempOutput.cols, tempOutput.rows)));

	std::stringstream stream;

	stream << "N. valid clusters: " << to_string(N_clusterValidi) << ", ";
	stream << std::fixed << std::setprecision(2) << "eps: " << dbscan_eps << ", ";
	stream << /*std::fixed << std::setprecision(2) <<*/ "Min pts cluster: " << minPtsNeighb << ", ";
	stream << /*std::fixed << std::setprecision(2) <<*/ "Mean pts cluster: " << N_medioElementiCluster << ", ";
	stream << (forgedOrNot ? "forged" : "original");
	string elabInfoText = stream.str();

	int y_start = 20;
	int x_start = 20;
	cv::putText(outputImg, elabInfoText, cv::Point(x_start, y_start), FONT_HERSHEY_PLAIN, 1, Colori::BLACK);
	isOutputImgDrawn = true;

}


bool CopyMoveDetectorSIFT::detect()
{

	if (!isSetInputImg)
	{
		cout << "l'immagine di input non è stata impostata o non è valida!" << endl;
		return false;
	}

	// se viene effettuata una nuova detection controllo che i risultati siano stati svuotati e nel caso lo faccio
	if (keypoints.size() != 0)
	{
		clearResults();
	}

	// estraggo i keypoints e faccio il matching
	extractKeyPoints();
	doKeyPointsMatching();

	// 1o filtraggio: elimino i match ambigui(test di Lowe).
	filtraggioMatchLowe();

	// 2o filtraggio: elimino i match per cui i keypoints sono troppo vicini spazialmente (coordinate x,y)
	filtraggioDistanzaLocale();

	filtraggioClustering();

	isDetectionOk = true;


	// se è richiesta viene disegnata l'immagine di output con l'elaborazione
	if (visualizeElab)
	{
		drawOutputImg();
	}

	return isDetectionOk;
}


void CopyMoveDetectorSIFT::extractKeyPoints()
{
	// estraggo i keypoints SIFT
	Ptr<SIFT> detectorPtr = SIFT::create(soglia_SIFT);
	detectorPtr->detect(*inputImg, keypoints);

	//writeKeyPoints(keypoints);

	clusteredKeyPoints.reserve(keypoints.size());
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		clusteredKeyPoints.push_back(ClusteredKeyPoint(&keypoints[i], i));
	}

	// ottengo i descrittori dei keypoints
	detectorPtr->compute(*inputImg, keypoints, descriptors);
	//MyUtility::writeDescriptors(descriptors);
}


void CopyMoveDetectorSIFT::doKeyPointsMatching()
{
	int k = 3;
	if (keypoints.size() < k)
	{
		return;
	}

	// effettuo il match tra i descrittori con la dll parallelizzata.
	// ottengo il puntatore ai dati dei descrittori
	float* descriptorsData = (float*)descriptors.ptr<float>(0);

	// costruisco l'oggetto CudaMatrix passandogli il puntatore ai dati dei descrittori
	CudaMatrix<float> cudaDescriptors(descriptors.rows, descriptors.cols, descriptorsData);

	auto start = std::chrono::steady_clock::now();
	std::lock_guard<std::mutex> lock(CopyMoveDetectorSIFT::mtxCuda);
	
	// calcolo le distanze tra i descrittori (i,j) e salvo gli indici dei migliori 3 match per ogni descrittore
	if (!cudaDescriptors.computeSelfDistances(descriptorDistances, bestMatchIndices))
	{
		cout << "n.descrittori: " << descriptors.rows << endl;
		cout << "Errore nel calcolo dei match con cuda!" << endl;
		throw exception();
		return;
	}
	auto end = std::chrono::steady_clock::now();
	cout << "n.descrittori: " << descriptors.rows << ", tempo per matching: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
}


void CopyMoveDetectorSIFT::filtraggioMatchLowe()
{
	if (bestMatchIndices.Rows() == 0)
	{
		return;
	}

	// costruisco la lista delle coppie di indici dei punti relativi ai match validi secondo il test di lowe
	vector<vector<int>> match_indici_lowe;
	vector<float> descriptorDistancesLowe;
	match_indici_lowe.reserve(bestMatchIndices.Rows());
	descriptorDistancesLowe.reserve(bestMatchIndices.Rows());

	
	for (size_t i = 0; i < bestMatchIndices.Rows(); i++)
	{
		// NB: considero la seconda e terza colonna delle distanze(la prima è sempre 0, 
		// perchè contiene le distanze tra i punti e loro stessi).
		if(descriptorDistances(i, bestMatchIndices(i, 1)) < soglia_Lowe * descriptorDistances(i, bestMatchIndices(i, 2)))
		{
			match_indici_lowe.push_back({ (int) i, (int) bestMatchIndices(i, 1) });
			descriptorDistancesLowe.push_back(descriptorDistances(i, bestMatchIndices(i, 1)));
		}
	}

	// elimino le coppie di indici doppie a meno dell'ordine, es: (4,5) e (5,4) => viene mantenuta solo la prima.
	// Questo perchè un match è individuato dalla coppia non ordinata di due punti
	vector<vector<int>> match_indici_no_doppioni;
	vector<int> indici_match_validi;
	MyUtility::eliminaDoppioni(match_indici_lowe, match_indici_no_doppioni, indici_match_validi);

	// creo la lista dei match con tutte le informazioni relative, utilizzando gli indici senza doppioni
	tempMatches.reserve(indici_match_validi.size());

	// E' necessario evitare di considerare due volte un match costituito da keypoints con stesse coordinate ma diverso "angle"
	// (la SIFT detection può generare più keypoints punti con stesse coordinate x,y ma orientamento differente).
	// Utilizzo dunque una hash-map in cui la chiave è una stringa costituita dalla concatenazione delle coordinate x,y 
	// dei due keypoints che costituiscono il match.
	unordered_map<string, int> n_matchForCoordsCouple;

	int matchID = 0;
	for (int id : indici_match_validi)
	{
		// creo la struttura wrapper che mantiene le info complete sul match (coppia di punti, descrittori, distanza, validità del match)
		// e la aggiungo alla lista dei match (campo della classe)
		ClusteredKeyPoint* kp1 = &clusteredKeyPoints[match_indici_lowe[id][0]];
		ClusteredKeyPoint* kp2 = &clusteredKeyPoints[match_indici_lowe[id][1]];

		string keyXcoords = kp1->component(0) < kp2->component(0) ? (std::to_string((int)kp1->component(0)) + std::to_string((int)kp2->component(0)))
			: (std::to_string((int)kp2->component(0)) + std::to_string((int)kp1->component(0)));

		string keyYcoords = kp1->component(1) < kp2->component(1) ? (std::to_string((int)kp1->component(1)) + std::to_string((int)kp2->component(1)))
			: (std::to_string((int)kp2->component(1)) + std::to_string((int)kp1->component(1)));
		string key = keyXcoords + keyYcoords;

		// verifico che non esista già un match per la stessa coppia di coppia di coordinate
		if (n_matchForCoordsCouple.count(key) == 0)
		{
			n_matchForCoordsCouple[key] = 1;
			KeyPointsMatch kpm(kp1, kp2, &descriptors.row(match_indici_lowe[id][0]), &descriptors.row(match_indici_lowe[id][1]), descriptorDistancesLowe[id], matchID);
			tempMatches.push_back(kpm);

			// imposto i riferimenti dai keypoints al match costituito da questi
			kp1->setParentMatch(&tempMatches.back());
			kp2->setParentMatch(&tempMatches.back());

			matchID++;
		}

	}
	//MyUtility::writeMatches(tempMatches);
}


void CopyMoveDetectorSIFT::filtraggioDistanzaLocale()
{
	// rimuovo i match per cui la distanza spaziale (coordinate x,y) 
	// dei keypoints è minore di una certa soglia
	float soglia_spaziale = inputImg->rows < inputImg->cols ? inputImg->rows / 20 : inputImg->cols / 20;

	matchesPointers.reserve(tempMatches.size());

	for (size_t i = 0; i < tempMatches.size(); i++)
	{
		if (tempMatches[i].spatialDistance > soglia_spaziale)
		{
			matchesPointers.push_back(&tempMatches[i]);
		}
	}

}



void CopyMoveDetectorSIFT::filtraggioClustering()
{
	if (matchesPointers.size() == 0)
	{
		return;
	}

	// creo la lista dei punti da clusterizzare con DBSCAN
	vector<IClusterPoint*> punti;
	punti.reserve(matchesPointers.size() * 2);
	for (auto& match : matchesPointers)
	{
		punti.push_back((IClusterPoint*)match->kp1);
		punti.push_back((IClusterPoint*)match->kp2);
	}

	DBSCAN dbscan(punti, minPtsNeighb);

	if (dbscan_eps == -1)
	{
		dbscan.findOptimalEps();
		dbscan_eps = dbscan.getEpsilon();
	}
	else
	{
		dbscan.setEpsilon(dbscan_eps);
	}
	
	dbscan.run();
	
	// ora all'interno del vettore matches i punti sono etichettati con le label dei cluster
	
	float maxDescriptorDist = 0;
	float minDescriptorDist = 1.e6;
	for (auto& match : matchesPointers)
	{
		if (match->descriptorDistance > maxDescriptorDist)
		{
			maxDescriptorDist = match->descriptorDistance;
		}
		if (match->descriptorDistance < minDescriptorDist)
		{
			minDescriptorDist = match->descriptorDistance;
		}
	}

	float rangeDescriptorDist = maxDescriptorDist - minDescriptorDist;


	// filtraggio dei match individuati
	for (auto& match : matchesPointers)
	{
		// i match per cui almeno un punto risulta essere un outlier non sono più validi. 
		if (match->kp1->getClusterID() == NOISE || match->kp2->getClusterID() == NOISE)
		{
			// se un punto della coppia è un outlier, imposto anche l'altro come outlier.
			match->kp1->getClusterID() == NOISE ? match->kp2->setClusterID(NOISE) : match->kp1->setClusterID(NOISE);
		}
		else if (match->kp1->getClusterID() == match->kp2->getClusterID())
		{
			if (match->descriptorDistance >= minDescriptorDist + sogliaDescInCluster * rangeDescriptorDist)
			{
				// escludo i match per cui i punti appartengono allo stesso cluster
				match->kp1->setClusterID(nonValidPointLabel::POINT_SAME_CLUSTER);
				match->kp2->setClusterID(nonValidPointLabel::POINT_SAME_CLUSTER);
			}

		}
	}


	// filtraggio dei match sulla base della dimensione dei cluster individuati:
	// escludo i match i cui punti appartengono a cluster troppo piccoli

	// faccio scorrere ogni cluster, tramite le label ottenute
	vector<int>& clusterLabels = dbscan.getClusterLabels();

	bool isChanging;
	int debug_n_change = 0;

	vector<bool> isValidCluster(clusterLabels.size(), true);
	isValidCluster[0] = false;	// noise
	do
	{
		isChanging = false;
		for (size_t i = 1; i < clusterLabels.size(); i++)	// NB: parto da 1 perchè la prima label corrisponde a NOISE
		{
			if (isValidCluster[i])
			{
				vector<IClusterPoint*>& puntiCluster_i = dbscan.getPointsInCluster(clusterLabels[i]);
				int N_puntiCluster_validi = 0;
				for (auto p : puntiCluster_i)
				{
					if (p->getClusterID() == clusterLabels[i])
					{
						N_puntiCluster_validi++;
					}
				}
				if (N_puntiCluster_validi < minPtsNeighb)
				{
					for (auto p : puntiCluster_i)
					{
						if (p->getClusterID() == clusterLabels[i])
						{
							((ClusteredKeyPoint*)p)->getParentMatch()->kp1->setClusterID(nonValidPointLabel::POINT_SMALL_CLUSTER);
							((ClusteredKeyPoint*)p)->getParentMatch()->kp2->setClusterID(nonValidPointLabel::POINT_SMALL_CLUSTER);
						}
					}
					isChanging = true;
					isValidCluster[i] = false;
				}
			}
		}
		debug_n_change++;
	} while (isChanging);


	N_clusterValidi = 0;
	N_medioElementiCluster = 0.0;
	labelClusterValidi.reserve(clusterLabels.size() - 1);
	
	for (size_t i = 1; i < clusterLabels.size(); i++)
	{
		if (isValidCluster[i])
		{
			int N_puntiCluster_validi = 0;
			vector<IClusterPoint*>& puntiCluster_i = dbscan.getPointsInCluster(clusterLabels[i]);
			for (auto p : puntiCluster_i)
			{
				if (p->getClusterID() == clusterLabels[i])
				{
					N_puntiCluster_validi++;
				}
			}

			N_clusterValidi++;
			N_medioElementiCluster += N_puntiCluster_validi;
			labelClusterValidi.push_back(clusterLabels[i]);
		}
	}

	N_medioElementiCluster = N_medioElementiCluster / N_clusterValidi;
	forgedOrNot = N_clusterValidi >= 1;
	
}


// metodo in overload, che permette di passare l'immagine di input come parametro
bool CopyMoveDetectorSIFT::detect(const cv::Mat& inputImg, const bool drawOutputImg)
{
	setInputImg(inputImg);
	setVisualizeElab(drawOutputImg);
	return detect();
}


