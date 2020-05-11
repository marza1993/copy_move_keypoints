// copy_move_keypoints.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

// Ehi Jackson, do you want some fish and chips for dinner?
// prova nuovo vs2019

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

const string DB_0 = "MICC-F220";
const string DB_1 = "MICC-F600";
const string DB_2 = "MICC-F2000";

string DATA_SET_PATH;

// path di output delle immagini classificate erroneamente(falsi negativi e falsi positivi)
string OUTPUT_PATH;
string OUTPUT_PATH_FN;
string OUTPUT_PATH_FP;
string OUTPUT_PATH_TN;
string OUTPUT_PATH_TP;
string NOME_FILE_GT;


const string WINDOW_NAME = "elaborazione";

// hashmap che contiene, per ogni immagine del data-set, il flag forged o meno. La chiave è il nome dell'immagine
unordered_map<string, bool> groundTruth;

vector<string> listaNomiImmagini;

unsigned int tot_forged = 0;
unsigned int tot_orig = 0;

// parametri
unsigned int minPuntiIntorno = 2;
float sogliaLowe = 0.43;
unsigned int sogliaSIFT = 4000 * 0;
float eps = -1; // 100;
float sogliaDescInCluster = 0.14;
bool useFLANN = false;
bool visualizza = false;
bool salva = false;


// per parallelizzazione: un vettore per oguno dei thread, con i valori TP, FP, FN, TN
vector<vector<unsigned int>> threadResults;


void setPaths(const string& DB_NAME)
{
    DATA_SET_PATH = "D:\\dottorato\\copy_move\\" + DB_NAME + "\\";

    // path di output delle immagini classificate erroneamente(falsi negativi e falsi positivi)
    OUTPUT_PATH = "D:\\dottorato\\copy_move\\" + DB_NAME + "_output_cpp\\";
    OUTPUT_PATH_FN = OUTPUT_PATH + "FN\\";
    OUTPUT_PATH_FP = OUTPUT_PATH + "FP\\";
    OUTPUT_PATH_TN = OUTPUT_PATH + "TN\\";
    OUTPUT_PATH_TP = OUTPUT_PATH + "TP\\";
    NOME_FILE_GT = "groundtruth_" + DB_NAME + ".txt";
}


void provaDetectionSingola()
{
    // ****** MICC-F220 ******
    //string nomeFile = "DSC_0812tamp1.jpg";
    //string nomeFile = "DSC_1535tamp133.jpg";
    //string nomeFile = "CRW_4809_scale.jpg";
    //string nomeFile = "CRW_4853tamp132.jpg";
    //string nomeFile = "DSCN45tamp131.jpg";
    //string nomeFile = "DSCN45tamp25.jpg";
    //string nomeFile = "P1000231_scale.jpg";

    //string nomeFile = "CRW_4809_scale.jpg";
    //string nomeFile = "sony_61_scale.jpg";
    //string nomeFile = "DSCN45tamp1.jpg";
    //string nomeFile = "CRW_4815_scale.jpg";
    string nomeFile = "nikon_7_scale.jpg";



    // ****** MICC-F600 ******
    //string nomeFile = "_r30_s1200sweets.png";

    setPaths(DB_0);
    //setPaths(DB_1);
    //setPaths(DB_3);
    

    const cv::Mat input = cv::imread(DATA_SET_PATH + nomeFile, cv::IMREAD_GRAYSCALE);
    const cv::Mat input2 = cv::imread(DATA_SET_PATH + nomeFile, cv::IMREAD_GRAYSCALE);

    cv::Mat outputImg;
    cv::Mat outputImg2;

    CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe, eps, sogliaDescInCluster, useFLANN);
    detector.detect(input, true);

    detector.getOuputImg(outputImg);
    cv::imwrite(OUTPUT_PATH + nomeFile, outputImg);
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 800, 600);
    cv::imshow(WINDOW_NAME, outputImg);
    cv::waitKey(0);
   
    
}

void getGroundTruth()
{
    // effettuo il parsing del file di testo per costruire la hashmap
    std::ifstream infile(DATA_SET_PATH + NOME_FILE_GT);

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


// mutex per sincronizzare l'accesso al std::cout
std::mutex mtx_cout;

// per parallelizzazione:
// "start" e "end" sono gli indici che identificano il sottoinsieme di nomi di immagini
// nel vettore globale "listaNomiImmagini".
void runOnImageSubset(int threadID, int start, int end)
{
    // true positive
    unsigned int TP = 0;
    // false positive
    unsigned int FP = 0;
    // false negative
    unsigned int FN = 0;
    // true negative
    unsigned int TN = 0;

    unsigned int N_tampered_found = 0;

    cv::Mat input, output;

    {
        std::lock_guard<std::mutex> lock(mtx_cout);
        cout << "*******************" << endl;
        cout << "thread with ID: " << threadID << endl;
        cout << "start index: " << start << ", end index: " << end << endl;
        cout << "*******************" << endl;
    }

    for (size_t i = start; i <= end; i++)
    {
        if (threadID == 0)
        {
            cout << "thread " << threadID << ", immagine n. " << (i + 1) << endl;
            cout << std::setprecision(2) << "stima avanzamento: " << (i / (float)(end - start + 1) * 100) << "%" << endl;
        }
        string& nomeImmagine = listaNomiImmagini[i];
        
        input = cv::imread(DATA_SET_PATH + nomeImmagine, cv::IMREAD_GRAYSCALE);

        CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe, eps, sogliaDescInCluster, useFLANN);

        detector.detect(input, salva);
        if (salva)
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
                double random = ((double)rand()) / RAND_MAX;
                
                if (false && salva && random > 1)
                {
                    cv::imwrite(OUTPUT_PATH_TP + "TP_" + nomeImmagine, output);
                }
            }
            // falso positivo
            else
            {
                FP++;

                if (salva)
                {
                    cv::imwrite(OUTPUT_PATH_FP + "FP_" + nomeImmagine, output);
                    //cv::imwrite(OUTPUT_PATH_FP + nomeImmagine, input);
                }
            }
        }
        else
        {
            // falso negativo (l'immagine era tampered ma l'algoritmo non l'ha riconosciuta)
            if (groundTruth[nomeImmagine])
            {
                FN++;

                if (salva)
                {
                    cv::imwrite(OUTPUT_PATH_FN + "FN_" + nomeImmagine, output);
                    //cv::imwrite(OUTPUT_PATH_FN + nomeImmagine, input);
                }
            }
            else
            {
                TN++;

                if (salva && false)
                {
                    cv::imwrite(OUTPUT_PATH_TN + "TN_" + nomeImmagine, output);
                }
            }
        }
    }

    // aggiungo i risultati al vettore globale
    if (threadResults[threadID].size() == 0)
    {
        threadResults[threadID].reserve(5);
        threadResults[threadID].push_back(TP);
        threadResults[threadID].push_back(FP);
        threadResults[threadID].push_back(FN);
        threadResults[threadID].push_back(TN);
        threadResults[threadID].push_back(N_tampered_found);
    }
    else
    {
        threadResults[threadID][0] += TP;
        threadResults[threadID][1] += FP;
        threadResults[threadID][2] += FN;
        threadResults[threadID][3] += TN;
        threadResults[threadID][4] += N_tampered_found;
    }

}



void provaParallelDataSet()
{
    string risp = "0";

    cout << "seleziona data-set: " << endl << "MICC-F220: 0 " << endl << "MICC-F600: 1" << endl << "MICC-F2000: 2" << endl;
    cin >> risp;

    if (risp == "0")
    {
        setPaths(DB_0);
    }
    else if(risp == "1")
    {
        setPaths(DB_1);
    }
    else
    {
        setPaths(DB_2);
    }

    getGroundTruth();

    salva = false;
    cout << "salvare le immagini elaborate? [s/n]" << endl;
    cin >> risp;
    if (risp == "s")
    {
        salva = true;
    }

    // ottengo il numero di processori logici disponibili
    const auto processor_count = std::thread::hardware_concurrency();

    // creo la lista dei thread: uno per ogni processore
    vector<thread> threads(processor_count);
    threadResults.resize(processor_count);

    // inizio a misurare il tempo di elaborazione
    auto start = std::chrono::steady_clock::now();

    int startIndex;
    int endIndex;

    int N_immaginiPerThread = (int)(listaNomiImmagini.size() / processor_count);
    //for (size_t i = 0; i < threads.size(); i++)
    for (int i = threads.size() - 1; i >= 0; i--)
    {
        startIndex = N_immaginiPerThread * i;
        endIndex = i == threads.size() - 1 ? listaNomiImmagini.size() - 1 : startIndex + N_immaginiPerThread - 1;
        threads[i] = std::thread(runOnImageSubset, i, startIndex, endIndex);
        //runOnImageSubset(i, startIndex, endIndex);
    }

    for (size_t i = 0; i < threads.size(); i++)
    {
        threads[i].join();
    }


    // fermo il cronometro
    auto end = std::chrono::steady_clock::now();

    // calcolo metriche: raccolgo i risultati dai vari thread
    // true positive
    unsigned int TP = 0;
    // false positive
    unsigned int FP = 0;
    // false negative
    unsigned int FN = 0;
    // true negative
    unsigned int TN = 0;

    unsigned int N_tampered_found = 0;

    for (auto& results : threadResults)
    {
        TP += results.at(0);
        FP += results.at(1);
        FN += results.at(2);
        TN += results.at(3);
        N_tampered_found += results.at(4);
    }

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
    cout << "TPR " << TPR << ", FPR: " << FPR << ", FNR: " << FNR << ", TNR: " << TNR << endl 
        << "FP: " << FP << ", FN: " << FN << ", TP: "  << TP << ", TN: " << TN << ", tot: " << (TP + TN + FP + FN) << endl;
    cout << "tempo di elaborazione: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << std::endl;
    cout << "****************************" << endl;

    // scrivo i risultati su file con i relativi parametri
    std::ofstream outfile;

    string sep = ";";
    outfile.open(OUTPUT_PATH + "risultati.csv", std::ios_base::app); // append instead of overwrite
    if (outfile.is_open())
    {
        outfile << std::setprecision(3) << sogliaSIFT << sep << sogliaLowe << sep << minPuntiIntorno << sep
            << eps << sep << sogliaDescInCluster << sep << useFLANN << sep
            << precision << sep << recall << sep << F1 << sep << accuracy << sep << TPR << sep
            << FPR << sep << FNR << sep << TNR << sep << FP << sep << FN << endl;
        outfile.close();
    }

}


void provaDataSet()
{

    string risp = "0";

    cout << "seleziona data-set: " << endl << "MICC-F220: 0 " << endl << "MICC-F600: 1" << endl;
    cin >> risp;

    
    if (risp == "0")
    {
        setPaths(DB_0);
    }
    else
    {
        setPaths(DB_1);
    }

    getGroundTruth();

    // true positive
    unsigned int TP = 0;
    // false positive
    unsigned int FP = 0;
    // false negative
    unsigned int FN = 0;
    // true negative
    unsigned int TN = 0;

    unsigned int N_tampered_found = 0;

    risp = "n";
    cout << "visualizzare le immagini elaborate? [s/n]" << endl;
    cin >> risp;
    visualizza = false;
    if (risp == "s")
    {
        visualizza = true;
    }
    
    salva = false;
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

    for(size_t i = 0; i < listaNomiImmagini.size(); i++)
    {
        string& nomeImmagine = listaNomiImmagini[i];
        cout << "immagine n. " << ++n << endl;
        input = cv::imread(DATA_SET_PATH + nomeImmagine, cv::IMREAD_GRAYSCALE);

        CopyMoveDetectorSIFT detector(sogliaSIFT, minPuntiIntorno, sogliaLowe, eps, sogliaDescInCluster, useFLANN);

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
                    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
                    cv::resizeWindow(WINDOW_NAME, 800, 600);
                    cv::imshow(WINDOW_NAME, output);
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
                    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
                    cv::resizeWindow(WINDOW_NAME, 800, 600);
                    cv::imshow(WINDOW_NAME, output);
                    cv::waitKey(1);
                }
                if (salva)
                {
                    cv::imwrite(OUTPUT_PATH_FP + "FP_" + nomeImmagine, output);
                    //cv::imwrite(OUTPUT_PATH_FP + nomeImmagine, input);
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
                    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
                    cv::resizeWindow(WINDOW_NAME, 800, 600);
                    cv::imshow(WINDOW_NAME, output);
                    cv::waitKey(1);
                }
                if (salva)
                {
                    cv::imwrite(OUTPUT_PATH_FN + "FN_" + nomeImmagine, output);
                    //cv::imwrite(OUTPUT_PATH_FN + nomeImmagine, input);
                }
            }
            else
            {
                TN++;
                if (visualizza)
                {
                    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
                    cv::resizeWindow(WINDOW_NAME, 800, 600);
                    cv::imshow(WINDOW_NAME, output);
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
        outfile << std::setprecision(3) << sogliaSIFT << sep << sogliaLowe << sep << minPuntiIntorno << sep << precision << sep << recall << sep << F1 << sep << accuracy << sep << TPR
            << sep << FPR << sep << FNR << sep << TNR << sep << FP << sep << FN << endl;
        outfile.close();
    }
    
}




int main(int argc, char** argv)
{
    // argomenti da linea di comando
    bool areParametersSet = true;
    if (argc > 1)
    {
        if (argc < 7)
        {
            cout << "argomenti insufficienti!" << endl;
            areParametersSet = false;
        }
        else
        {
            minPuntiIntorno = std::stoi(std::string(argv[1]));
            sogliaLowe = std::stof(std::string(argv[2]));
            sogliaSIFT = std::stof(std::string(argv[3]));
            eps = std::stoi(std::string(argv[4]));
            sogliaDescInCluster = std::stof(std::string(argv[5]));
            useFLANN = (bool) std::stoi(std::string(argv[6]));
            cout << "parametri: " << endl;
            cout << "minPuntiIntorno: " << minPuntiIntorno << endl
                << "sogliaLowe: " << sogliaLowe << endl
                << "sogliaSIFT: " << sogliaSIFT << endl
                << "eps: " << eps << endl
                << "sogliaDescInCluster: " << sogliaDescInCluster << endl
                << "useFlann: " << useFLANN << endl;
        }
    }
    else
    {
        cout << "parametri: " << endl;
        cout << "minPuntiIntorno: " << endl;
        cin >> minPuntiIntorno;
        cout << "sogliaLowe: " << endl;
        cin >> sogliaLowe;
        cout << "sogliaSIFT: " << endl;
        cin >> sogliaSIFT;
        cout << "eps: " << endl;
        cin >> eps;
        cout << "sogliaDescInCluster: " << endl;
        cin >> sogliaDescInCluster;
        cout << "useFlann: " << endl;
        cin >> useFLANN;

        cout << "parametri: " << endl;
        cout << "minPuntiIntorno: " << minPuntiIntorno << endl
            << "sogliaLowe: " << sogliaLowe << endl
            << "sogliaSIFT: " << sogliaSIFT << endl
            << "eps: " << eps << endl
            << "sogliaDescInCluster: " << sogliaDescInCluster << endl
            << "useFlann: " << useFLANN << endl;
    }

    if (areParametersSet)
    {
        provaParallelDataSet();
        
        //provaDataSet();

        //provaDetectionSingola();

    }


    system("pause");
    
    return 0;
}




