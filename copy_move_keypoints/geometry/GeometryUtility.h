#pragma once
#include <vector>

class GeometryUtility
{

public:

	// calcola la distanza tra il punto P e il vettore AB = B - A.
	// Restituisce anche le coordinate del punto P_ort, proiezione ortogonale
	// del punto P sulla retta passante per A, B.
	template <class T>
	static float pointVectorDistance(const T& A, const T& B, const T& P, T& POrt);

	// overloading senza il calcolo del punto proiezione
	template <class T>
	static float pointVectorDistance(const T& A, const T& B, const T& P);


	// Data una curva identificata dalle coordinate contenute nei vettori X
	// e Y, restituisce l'indice del punto in cui si ha la massima piegatura
	// ("elbow") nella curva stessa(che si suppone avere una forma a
	// "gomito").
	static int findElbow(const std::vector<float>& X, const std::vector<float>& Y);

};

