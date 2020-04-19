// copy_move_keypoints.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

// Ehi Jackson, do you want some fish and chips for dinner?

#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <chrono>
#include "CopyMoveDetectorSIFT.h";


using namespace std;

const string DATA_SET_PATH = "D:\\progetti\\dottorato\\copy_move\\MICC-F220\\";

// path di output delle immagini classificate erroneamente(falsi negativi e falsi positivi)
const string OUTPUT_PATH_FN = "D:\\progetti\\dottorato\\copy_move\\output_cpp\\FN\\";
const string OUTPUT_PATH_FP = "D:\\progetti\\dottorato\\copy_move\\output_cpp\\FP\\";
const string OUTPUT_PATH_TN = "D:\\progetti\\dottorato\\copy_move\\output_cpp\\TN\\";
const string OUTPUT_PATH_TP = "D:\\progetti\\dottorato\\copy_move\\output_cpp\\TP\\";
const string OUTPUT_PATH = "D:\\progetti\\dottorato\\copy_move\\output_cpp\\";

// hashmap che contiene, per ogni immagine del data-set, il flag forged o meno. La chiave è il nome dell'immagine
unordered_map<string, bool> groundTruth;

vector<string> listaNomiImmagini;

unsigned int tot_forged = 0;
unsigned int tot_orig = 0;


void provaDetectionSingola()
{
    //string nomeFile = "DSC_0812tamp1.jpg";
    //string nomeFile = "DSC_1535tamp133.jpg";
    //string nomeFile = "CRW_4809_scale.jpg";
    //string nomeFile = "CRW_4853tamp132.jpg";
    //string nomeFile = "DSCN45tamp131.jpg";
    //string nomeFile = "DSCN45tamp25.jpg";
    //string nomeFile = "P1000231_scale.jpg";

    //string nomeFile = "CRW_4809_scale.jpg";
    //string nomeFile = "sony_61_scale.jpg";
    string nomeFile = "DSCN45tamp1.jpg";

    const cv::Mat input = cv::imread(DATA_SET_PATH + nomeFile, cv::IMREAD_GRAYSCALE);
    const cv::Mat input2 = cv::imread(DATA_SET_PATH + nomeFile, cv::IMREAD_GRAYSCALE);
    cv::Mat outputImg;
    cv::Mat outputImg2;

    unsigned int minPuntiIntorno = 3;
    float sogliaLowe = 0.47;
    unsigned int sogliaSIFT = 92;
    {
        CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe);
        detector.detect(input, true);

        detector.getOuputImg(outputImg);
        cv::imwrite(OUTPUT_PATH + nomeFile, outputImg);
        cv::imshow("elaborazione", outputImg);
        cv::waitKey(0);
    }
    
    //{
    //    CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe);
    //    detector.detect(input2, true);
    //    detector.getOuputImg(outputImg2);
    //    cv::imwrite(OUTPUT_PATH + "2_" + nomeFile, outputImg2);
    //    cv::imshow("elaborazione", outputImg2);
    //    cv::waitKey(0);
    //}


    
}

void getGroundTruth()
{
    // effettuo il parsing del file di testo per costruire la hashmap
    std::ifstream infile(DATA_SET_PATH + "groundtruthDB_220.txt");

    if (infile.is_open())
    {

        string nomeImmagine, strForgedOrNot;
        while (infile >> nomeImmagine >> strForgedOrNot)
        {
            groundTruth[nomeImmagine] = strForgedOrNot == "1";
            if (groundTruth[nomeImmagine])
            {
                tot_forged++;
            }
            else
            {
                tot_orig++;
            }
            listaNomiImmagini.push_back(nomeImmagine);
        }
        infile.close();
    }
}


void provaDataSet()
{
    getGroundTruth();
    unsigned int minPuntiIntorno = 4;
    float sogliaLowe = 0.47;
    unsigned int sogliaSIFT = 92;

    // true positive
    unsigned int TP = 0;
    // false positive
    unsigned int FP = 0;
    // false negative
    unsigned int FN = 0;
    // true negative
    unsigned int TN = 0;

    unsigned int N_tampered_found = 0;

    string risp = "n";
    cout << "visualizzare le immagini elaborate? [s/n]" << endl;
    cin >> risp;
    bool visualizza = false;
    if (risp == "s")
    {
        visualizza = true;
    }
    
    bool salva = false;
    cout << "salvare le immagini elaborate? [s/n]" << endl;
    cin >> risp;
    if (risp == "s")
    {
        salva = true;
    }

    // eseguo l'analisi di copy-move su ogni immagine del data-set
    int n = 0;
    cv::Mat input, output;
    srand(0);

    // inizio a misurare il tempo di elaborazione
    auto start = std::chrono::steady_clock::now();
    //for (auto& nomeImmagine : listaNomiImmagini)
    for(size_t i = 0; i < listaNomiImmagini.size(); i++)
    {
        string& nomeImmagine = listaNomiImmagini[i];
        cout << "immagine n. " << ++n << endl;
        input = cv::imread(DATA_SET_PATH + nomeImmagine, cv::IMREAD_GRAYSCALE);

        CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe);

        detector.detect(input, visualizza || salva);
        if (visualizza || salva)
        {
            detector.getOuputImg(output);
        }
        
        bool tampered = detector.getIsForged();

        if (tampered)
        {
            N_tampered_found++;
            // se era effettivamente tampered
            if (groundTruth[nomeImmagine])
            {
                TP++;
                // salvo le TP solo con una certa probabilità
                double random = ((double) rand()) / RAND_MAX;
                if (visualizza)
                {
                    cv::imshow("elab", output);
                    cv::waitKey(1);
                }
                if (false && salva && random >1)
                { 
                    cv::imwrite(OUTPUT_PATH_TP + "TP_" + nomeImmagine, output);
                }
            }
            // falso positivo
            else
            {
                FP++;
                if (visualizza)
                {
                    cv::imshow("elab", output);
                    cv::waitKey(1);
                }
                if (salva)
                {
                    cv::imwrite(OUTPUT_PATH_FP + "FP_" + nomeImmagine, output);
                }
            }
        }
        else
        {
            // falso negativo (l'immagine era tampered ma l'algoritmo non l'ha riconosciuta)
            if (groundTruth[nomeImmagine])
            {
                FN++;
                if (visualizza)
                {
                    cv::imshow("elab", output);
                    cv::waitKey(1);
                }
                if (salva)
                {
                    cv::imwrite(OUTPUT_PATH_FN + "FN_" + nomeImmagine, output);
                }
            }
            else
            {
                TN++;
                if (visualizza)
                {
                    cv::imshow("elab", output);
                    cv::waitKey(1);
                }
                if (salva && false)
                {
                    cv::imwrite(OUTPUT_PATH_TN + "TN_" + nomeImmagine, output);
                }
            }
        }
    }
    // fermo il cronometro
    auto end = std::chrono::steady_clock::now();

    // calcolo metriche
    double precision = ((double)TP) / N_tampered_found;
    double TPR = ((double)TP) / tot_forged;
    double recall = TPR;
    double FPR = ((double)FP) / tot_orig;
    double F1 = 2 / (1 / precision + 1 / recall);
    double FNR = ((double)FN) / tot_forged;
    double TNR = ((double)TN) / tot_orig;
    double accuracy = ((double)(TP + TN)) / (TP + TN + FP + FN);
    cout << "****************************" << endl;
    cout << std::setprecision(3) << "precision: " << precision << ", recall: " << recall << ", F1-score: " << F1 << ", accuracy: " << accuracy << endl;
    cout << "TPR: " << TPR << ", FPR: " << FPR << ", FNR: " << FNR << ", TNR: " << TNR << endl;
    cout << "tempo di elaborazione: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << std::endl;
    cout << "****************************" << endl;
    
    // scrivo i risultati su file con i relativi parametri
    std::ofstream outfile;

    string sep = ";";
    outfile.open(OUTPUT_PATH + "risultati.csv", std::ios_base::app); // append instead of overwrite
    if (outfile.is_open())
    {
        outfile << std::setprecision(3) << sogliaLowe << sep << minPuntiIntorno << sep << precision << sep << recall << sep << F1 << sep << accuracy << sep << TPR
            << sep << FPR << sep << FNR << sep << TNR << sep << FP << sep << FN << endl;
        outfile.close();
    }
    system("pause");
    
}




int main(int argc, char** argv)
{
    
    provaDataSet();

    //provaDetectionSingola();

    return 0;
}




