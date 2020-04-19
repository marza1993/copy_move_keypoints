#pragma once

#include <vector>

class MyUtility
{

public:

	// Data una lista di coppie di numeri interi, elimina le coppie ripetute uguali o uguali a meno dell'ordine.
	// Ad es: (4,5) .. (5,4) => viene mantenuta solo la prima.
	// Restituisce anche la lista degli indici delle coppie rimaste dopo il filtraggio (cio�: listaCoppieNoDoppioni = listaCoppie[indiciRigheRimaste][:]).
	// NB: il vettore di output passato non pu� essere lo stesso di quello di input (infatti quest'ultimo � const).
	static void eliminaDoppioni(const std::vector<std::vector<int>>& listaCoppie, std::vector<std::vector<int>>& listaCoppieNoDoppioni,
								std::vector<int>& indiciRigheRimaste);

};

