#pragma once

#include <vector>
#include <cmath>
#include <memory>
#include <unordered_map>
#include "ClusterPointDataStructures.h"

class DBSCAN {

private:

    std::vector<IClusterPoint*> getEpsNeighborhood(IClusterPoint& point);
    bool expandCluster(IClusterPoint& point, int clusterID);

    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
    
    // lista dei cluster individuati. Ogni cluster è rappresentato da un vettore di punti ed è indicizzato
    // dalla sua label (gli outliers sono indicizzati dalla label NOISE).
    // Questo campo è utile per ottenere velocemente tutti i punti appartenenti ad un determinato cluster, nota la sua label
    std::unordered_map<int, std::vector<IClusterPoint *>> foundClustersMap;

    // contiene la lista di tutte le label dei cluster individuati (es: -1 (NOISE), 1, 2, .., NumCluster).
    std::vector<int> clusterLabels;

public:

    // Se non viene passato un valore, prima di invocare il metodo run(), epsilon dovrà essere impostato tramite il metodo setEpsilon() 
    // oppure potrà essere determinato con il metodo findOptimalEps(), secondo l'euristica descritta
    // nell'articolo del DBSCAN
    DBSCAN(std::vector<IClusterPoint*>& points, unsigned int minPts, float eps = -1) : m_points(points)
    {
        m_minPoints = minPts;
        m_epsilon = eps;
        m_pointSize = points.size();
        // aggiungo la label degli outlier alla lista
        clusterLabels.push_back(NOISE);
    }

    ~DBSCAN(){}

    void run();

    // ottiene la lista di tutti i punti assegnati al cluster con label clusterID (per ottenere gli outliers è necessario
    // chiamare il metodo passando clusterID = NOISE)
    std::vector<IClusterPoint*>& getPointsInCluster(int clusterID);

    // restituisce la lista delle label create per i cluster individuati (compreso NOISE), es: -1,1,2,3,..,Num_cluster
    std::vector<int>& getClusterLabels();

    // restituisce la lista dei cluster ID assegnati a tutti i punti, come un vettore di m_points.size() elementi
    std::vector<int> getClusterIDs();

    // metodo che permette di ottenere il valore ottimale di eps sulla base della distribuzione delle distanze
    // tra i punti, secondo quanto descritto nell'articolo originale del DBSCAN.
    float findOptimalEps();

    void setEpsilon(float eps) { this->m_epsilon = eps; };

    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilon() {return m_epsilon;}

    std::vector<IClusterPoint*>& m_points;

};

