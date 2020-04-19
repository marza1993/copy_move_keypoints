#include <iostream>
#include <iomanip>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include "CopyMoveDetectorSIFT.h"
#include "MyUtility.h"
#include "Colori_cv.h"
#include "GeometryUtility.h"
#include <fstream>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

CopyMoveDetectorSIFT::CopyMoveDetectorSIFT() : CopyMoveDetectorSIFT(100, 3, 0.6)
{
	// costruttore con parametri di default
}



void writeKeyPoints(std::vector<cv::KeyPoint>& keypoints)
{
	ofstream myfile;
	myfile.open("keypoints.csv");

	for(auto& kp : keypoints)
	{
		myfile << kp.pt.x << ", " << kp.pt.y << ", " << kp.angle << ", " << kp.class_id << ", " << kp.octave << ", " << kp.response << ", " << kp.size << endl;
	}

	myfile.close();
}

void writeDescriptors(cv::Mat& descriptors)
{
	ofstream myfile;
	myfile.open("descriptors.csv");

	float* data = (float*)descriptors.ptr<float>(0);
	for (size_t r = 0; r < descriptors.rows; r++)
	{
		for (size_t c = 0; c < descriptors.cols; c++)
		{
			myfile << data[r * descriptors.cols + c] << ", ";
		}
		myfile << endl;
	}


	myfile.close();
}

void writeKnnMatches(vector<vector<cv::DMatch>>& knn_matches)
{
	ofstream myfile;
	myfile.open("knn.csv");

	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		for (size_t j = 0; j < knn_matches[0].size(); j++)
		{
			myfile << knn_matches[i][j].trainIdx << ", " << knn_matches[i][j].queryIdx << ", " << knn_matches[i][j].distance << ", ";
		}
		myfile << endl;
	}

	myfile.close();
}


void writeMatches(vector<KeyPointsMatch>& matches)
{
	ofstream myfile;
	myfile.open("matches.csv");

	for (auto& m : matches)
	{
		myfile << m.kp1->component(0) << ", " << m.kp1->component(1) << ", " << m.kp2->component(0) << ", " << m.kp2->component(1) << ", "
			<< m.descriptorDistance << ", " << m.spatialDistance << ", " << m.label << endl;
	}

	myfile.close();
}


CopyMoveDetectorSIFT::CopyMoveDetectorSIFT(const unsigned int soglia_SIFT, const unsigned int minPtsNeighb, const float soglia_Lowe)
{

	this->soglia_SIFT = soglia_SIFT;
	this->minPtsNeighb = minPtsNeighb;
	this->soglia_Lowe = soglia_Lowe;
	this->N_clusterValidi = 0;
	this->N_medioElementiCluster = 0;
	this->dbscan_eps = -1;
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

		case NOISE:

			drawMarker(tempOutput, match->kp1->getPoint(), Colori::RED, cv::MARKER_TILTED_CROSS, 8, 1, cv::LINE_AA);
			drawMarker(tempOutput, match->kp2->getPoint(), Colori::RED, cv::MARKER_TILTED_CROSS, 8, 1, cv::LINE_AA);
			break;

		case -2:

			circle(tempOutput, match->kp1->getPoint(), 5, Colori::CYAN, 1, cv::LINE_AA);
			circle(tempOutput, match->kp2->getPoint(), 5, Colori::CYAN, 1, cv::LINE_AA);
			break;

		case -3:

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
	stream << std::fixed << std::setprecision(2) << "N. medio elementi cluster: " << N_medioElementiCluster << ", ";
	stream << "result: " << (forgedOrNot ? "forged" : "original");
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
	Ptr<SIFT> detectorPtr = SIFT::create(4000 * 0);
	detectorPtr->detect(*inputImg, keypoints);

	//writeKeyPoints(keypoints);

	//int numOctaves = 3;
	//int numScaleLevels = 4;
	//Ptr<SURF> detectorPtr = SURF::create(soglia_SIFT, numOctaves, numScaleLevels);
	//detectorPtr->detect(*inputImg, keypoints);

	clusteredKeyPoints.reserve(keypoints.size());
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		clusteredKeyPoints.push_back(ClusteredKeyPoint(&keypoints[i], i));
	}

	//cout << "n. keypoints: " << keypoints.size() << endl;
	// ottengo i descrittori dei keypoints
	detectorPtr->compute(*inputImg, keypoints, descriptors);
	//writeDescriptors(descriptors);
}


void CopyMoveDetectorSIFT::doKeyPointsMatching()
{

	if (keypoints.size() == 0)
	{
		return;
	}

	// effettuo il match tra i descrittori ricavando, per ogni descrittore, i suoi k più vicini
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	int k = 3;
	matcher->knnMatch(descriptors, descriptors, knn_matches, k);
	//writeKnnMatches(knn_matches);
}


void CopyMoveDetectorSIFT::filtraggioMatchLowe()
{
	if (knn_matches.size() == 0)
	{
		return;
	}

	// costruisco la lista delle coppie di indici dei punti relativi ai match validi secondo il test di lowe
	vector<vector<int>> match_indici;
	vector<float> descriptorDistances;
	match_indici.reserve(knn_matches.size());
	descriptorDistances.reserve(knn_matches.size());
	
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		// NB: considero la seconda e terza colonna delle distanze(la prima è sempre 0, 
		// perchè contiene le distanze tra i punti e loro stessi).
		if (knn_matches[i][1].distance < soglia_Lowe * knn_matches[i][2].distance)
		{
			match_indici.push_back({ knn_matches[i][1].trainIdx, knn_matches[i][1].queryIdx });
			descriptorDistances.push_back(knn_matches[i][1].distance);
		}
	}

	// elimino le coppie di indici doppie a meno dell'ordine, es: (4,5) e (5,4) => viene mantenuta solo la prima.
	// Questo perchè un match è individuato dalla coppia non ordinata di due punti
	vector<vector<int>> match_indici_no_doppioni;
	vector<int> indici_match_validi;
	MyUtility::eliminaDoppioni(match_indici, match_indici_no_doppioni, indici_match_validi);

	// creo la lista dei match con tutte le informazioni relative, utilizzando gli indici senza doppioni
	tempMatches.reserve(indici_match_validi.size());

	int matchID = 0;
	for (int id : indici_match_validi)
	{
		// creo la struttura wrapper che mantiene le info complete sul match (coppia di punti, descrittori, distanza, validità del match)
		// e la aggiungo alla lista dei match (campo della classe)
		ClusteredKeyPoint* kp1 = &clusteredKeyPoints[match_indici[id][0]];
		ClusteredKeyPoint* kp2 = &clusteredKeyPoints[match_indici[id][1]];

		KeyPointsMatch kpm(kp1, kp2, &descriptors.row(match_indici[id][0]), &descriptors.row(match_indici[id][1]), descriptorDistances[id], matchID);
		tempMatches.push_back(kpm);
		
		// imposto i riferimenti dai keypoints al match costituito da questi
		kp1->setParentMatch(&tempMatches.back());
		kp2->setParentMatch(&tempMatches.back());

		matchID++;
	}
	//writeMatches(tempMatches);
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

	vector<IClusterPoint*> punti;
	punti.reserve(matchesPointers.size() * 2);
	for (auto& match : matchesPointers)
	{
		punti.push_back((IClusterPoint*)match->kp1);
		punti.push_back((IClusterPoint*)match->kp2);
	}

	//DBSCAN dbscan(punti, minPtsNeighb);
	DBSCAN dbscan(punti, minPtsNeighb, 50);
	//dbscan.findOptimalEps();
	dbscan_eps = dbscan.getEpsilon();	// per poterlo stampare come info alla fine
	dbscan.run();
	
	// ora all'interno del vettore matches i punti sono etichettati con le label dei cluster
	
	int debug_stesso_cluster = 0;
	int debug_noise_noise = 0;
	int debug_noise_valid = 0;
	int debug_valid_valid = 0;

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
			if (match->kp1->getClusterID() != NOISE || match->kp2->getClusterID() != NOISE)
			{
				debug_noise_valid++;
			}
			else
			{
				debug_noise_noise++;
			}

			// se un punto della coppia è un outlier, imposto anche l'altro come outlier.
			match->kp1->getClusterID() == NOISE ? match->kp2->setClusterID(NOISE) : match->kp1->setClusterID(NOISE);
		}
		else if (match->kp1->getClusterID() == match->kp2->getClusterID())
		{
			if (match->descriptorDistance >= minDescriptorDist + 0.2 * rangeDescriptorDist)
			{
				// escludo i match per cui i punti appartengono allo stesso cluster
				match->kp1->setClusterID(-2);
				match->kp2->setClusterID(-2);
				debug_stesso_cluster++;
			}

		}
		else
		{
			debug_valid_valid++;
		}
	}


	// filtraggio dei match sulla base della dimensione dei cluster individuati:
	// escludo i match i cui punti appartengono a cluster troppo piccoli

	// faccio scorrere ogni cluster, tramite le label ottenute
	vector<int>& clusterLabels = dbscan.getClusterLabels();

	//for (size_t i = 1; i < clusterLabels.size(); i++)	// NB: parto da 1 perchè la prima label corrisponde a NOISE
	//{
	//	vector<IClusterPoint*>& puntiCluster_i = dbscan.getPointsInCluster(clusterLabels[i]);
	//	int N_puntiCluster_validi = 0;
	//	for (auto p : puntiCluster_i)
	//	{
	//		if (p->getClusterID() == clusterLabels[i])
	//		{
	//			N_puntiCluster_validi++;
	//		}
	//	}
	//	if (N_puntiCluster_validi < minPtsNeighb)
	//	{
	//		for (auto p : puntiCluster_i)
	//		{
	//			if (p->getClusterID() == clusterLabels[i])
	//			{
	//				((ClusteredKeyPoint*)p)->getParentMatch()->kp1->setClusterID(-3);
	//				((ClusteredKeyPoint*)p)->getParentMatch()->kp2->setClusterID(-3);
	//			}
	//		}
	//	}
	//	
	//}

	
	unordered_map<int, int> numElementiValidiPerCluster;
	

	for (size_t i = 1; i < clusterLabels.size(); i++)	// NB: parto da 1 perchè la prima label corrisponde a NOISE
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
		numElementiValidiPerCluster[clusterLabels[i]] = N_puntiCluster_validi;
	}

	N_medioElementiCluster = 0.0;

	labelClusterValidi.reserve(clusterLabels.size() - 1);
	for (size_t i = 1; i < clusterLabels.size(); i++)
	{
		int n = numElementiValidiPerCluster[clusterLabels[i]];
		if (n >= minPtsNeighb)
		{
			N_clusterValidi++;
			N_medioElementiCluster += n;
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


