#pragma once
#include "CopyMoveDetector.h"
#include "UtilityDataStructures.h"
#include "CudaMatrix.h"
#include "CudaMatrixSimmetric.h"

/*
* Classe che implementa il metodo di forgery detection tramite un matching tra keypoints di tipo SIFT. 
* L'idea sfrutta è la seguente: regioni copiate e incollate (ed eventualmente trasformate tramite rotazioni,
* rescaling, filtraggi etc..) avranno presumibilmente keypoints simili.
* Dopo aver estratto i keypoints, viene effettuata una serie di filtraggi per mantenere solo i match più promettenti.
*/
class CopyMoveDetectorSIFT : public CopyMoveDetector
{
private:

	// lista dei match individuati: ogni match possiede i riferimenti ai due punti, ai loro descrittori, la loro distanza
	// e altre informazioni, quali la validità o meno del match stesso.
	std::vector<KeyPointsMatch*> matchesPointers;


	std::vector<KeyPointsMatch> tempMatches;

	// vettore che conterrà i keypoints SIFT estratti
	std::vector<cv::KeyPoint> keypoints;

	// matrice di dimensione keypoints.size() x N (N = 128); contiene, per ognuno dei keypoints estratti
	// il corrispondente descrittore (vettore di 128 valori)
	cv::Mat descriptors;

	// vettore che contiene i puntatori ai keipoints estratti e la relativa label del cluster di appartenenza
	std::vector<ClusteredKeyPoint> clusteredKeyPoints;

	CudaMatrixSimmetric<float> descriptorDistances;
	CudaMatrix<unsigned int> bestMatchIndices;

	// TODO aggiugere il numero di elementi per cluster

	unsigned int N_clusterValidi;

	std::vector<int>labelClusterValidi;

	float N_medioElementiCluster;

	// se non viene passato un valore viene calcolato il valore ottimale con l'euristica presente nella classe DBSCAN
	float dbscan_eps;

	// soglia utilizzata per l'estrazione dei keypoints SIFT
	unsigned int soglia_SIFT;
	
	// dimensione dell'intorno da specificare per il custering (basato su DBSCAN)
	unsigned int minPtsNeighb;

	// soglia utilizzata per scartare i match poco significativi secondo il metodo descritto da Lowe nell'articolo originale
	float soglia_Lowe;

	// soglia utilizzata per determinare se due keypoints match appartenenti ad uno stesso cluster possono essere comunque
	// considerati ancora match.
	float sogliaDescInCluster;

	// disegna il risultato dell'elaborazione sull'immagine di output, in particolare i keypoints individuati, i match e
	// i colori in base alle label assegnate ai match (il "motivo dello scarto")
	void drawOutputImg();

	// Estrazione e salvataggio nei rispettivi campi dei keypoints SIFT e dei relativi descrittori.
	void extractKeyPoints();

	// effettua il match tra i keypoints individuati, valutando la distanza tra i descrittori
	void doKeyPointsMatching();

	// Effettua il filtraggio dei match sulla base del test di Lowe (descritto nell'articolo).
	// Post-condizione: il vettore dei match "matches" è popolato con i match che superano questo tipo di filtraggio
	void filtraggioMatchLowe();

	// Effettua il filtraggio dei match sulla base della distanza spaziale delle coordinate x,y dei keypoints. 
	// I due keypoints che identificano un match devono avere una distanza maggiore di una certa soglia
	void filtraggioDistanzaLocale();

	// Effettua un clustering spaziale dei keypoints dei match rimasti ed utilizza la clusterizzazione ottenuta
	// per effettuare un ulteriore filtraggio dei match (ad es. vengono eliminati i match tra punti riconosciuti come outliers)
	void filtraggioClustering();

	// mutex per sincronizzare l'accesso alla dll parallela (per la GPU)
	static std::mutex mtxCuda;


public:

	CopyMoveDetectorSIFT();

	// costruttore in overload che permette di passare subito i parametri
	CopyMoveDetectorSIFT(const unsigned int soglia_SIFT, const unsigned int minPtsNeighb, const float soglia_Lowe, 
		const float eps = -1, const float sogliaDescInCluster = 0.25);

	void setSogliaSIFT(unsigned int soglia_SIFT) {
		this->soglia_SIFT = soglia_SIFT;
	}

	void setMinPtsNeighb(unsigned int minPtsNeighb) {
		this->minPtsNeighb = minPtsNeighb;
	}

	void setSogliaLowe(float sogliaLowe) {
		this->soglia_Lowe = sogliaLowe;
	}

	bool detect();

	// metodo in overload, che permette di passare l'immagine di input come parametro
	bool detect(const cv::Mat& inputImg, const bool drawOutputImg = false);


	// metodo per eliminare i keypoints, descrittori e match individuati, per poter effettuare una nuova detection
	void clearResults()
	{
		matchesPointers.clear();
		tempMatches.clear();
		keypoints.clear();
		clusteredKeyPoints.clear();
		descriptorDistances.clearData();
		bestMatchIndices.clearData();
		N_clusterValidi = 0;
		N_medioElementiCluster = 0;
		labelClusterValidi.clear();
		dbscan_eps = 0;
		forgedOrNot = false;
	}


};

