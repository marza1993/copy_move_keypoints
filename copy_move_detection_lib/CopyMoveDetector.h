#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

/*
 * classe astratta che specifica il comportamento e gli attributi base di un generico algoritmo per l'analisi di copy-move forgery.
 * In particolare, dovrà fornire un metodo che permetta di eseguire la detection, dei metodi per impostare eventuali parametri base
 * e metodi per ottenere i risultati (tra cui eventualmente l'immagine con l'elaborazione).
*/
class CopyMoveDetector
{

protected:

    // Riferimento all'immagine di input su cui fare l'analisi di copy-move forgery.
    // Può essere impostata all'inizio o passata direttamente come argomento del metodo che fa la detection.
    const cv::Mat* inputImg = nullptr;

    // Immagine di output su cui mostrare il risultato dell'elaborazione
    cv::Mat outputImg;

    // flag che permette di specificare se si vuole ottenere anche l'immagine di output con il risultato dell'elaborazione
    bool visualizeElab;

    // flag che viene impostato a true quando l'immagine di input viene passata ed è valida (tramite il metodo setInputImg
    // o durante la chiamata al metodo detect).
    bool isSetInputImg;

    // Risultato dell'analisi di forgery detection: booleano che viene impostato a true se l'analisi
    // ha dato esito positivo, false altrimenti
    bool forgedOrNot;

    // flag per sapere se l'analisi di forgery detection è andata a buon fine o se per qualche motivo non è stata completata
    // con successo
    bool isDetectionOk;

    // flag per sapere se l'immagine di output è già stata disegnata
    bool isOutputImgDrawn;

    // genera l'immagine di output con i risultati dell'elaborazione.
    virtual void drawOutputImg() = 0;

    // Costruttore per inizializzare i campi. NB: essendo questa classe astratta, non verrà mai chiamato direttamente, però potrà essere
    // chiamato dalle classi derivate
    CopyMoveDetector();


public:


    // metto virtual sul distruttore per farlo sovrascrivere da una classe derivata
    virtual ~CopyMoveDetector() {};

    // Imposta l'immagine di input su cui effettuare l'analisi.
    void setInputImg(const cv::Mat& inputImg);

    // Metodo che effettua l'analisi di copy-move e popola i campi con i risultati; inoltre richiamerà il metodo per disegnare i risultati
    // dell'elaborazione sull'immagine di output, nel caso che sia impostato il flag "visualizeElab".
    // Ogni classe che erediterà da questa dovrà fornire un'implementazione di questo metodo.
    // Resituisce true se l'elaborazione è andata a buon fine (oltre ad impostare a true il campo "isDetectionOk").
    virtual bool detect() = 0;

    // metodo in overload, che permette di passare l'immagine di input come parametro
    virtual bool detect(const cv::Mat& inputImg, const bool drawOutputImg = false) = 0;

    // Restituisce il risultato dell'analisi: forged o meno
    virtual bool getIsForged();

    // Restituisce il flag isDetectionOk
    virtual bool getIsDetectionOk();

    // imposta il flag per la creazione dell'immagine con l'elaborazione
    virtual void setVisualizeElab(const bool value);

    // Restituisce un puntatore all'immagine di output o null (se questa non esisteva)
    virtual bool getOuputImg(cv::Mat& outputImg);


};

