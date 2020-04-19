#include "MyUtility.h"
#include <string>
#include <unordered_map>

using namespace std;

void MyUtility::eliminaDoppioni(const std::vector<std::vector<int>>& listaCoppie, std::vector<std::vector<int>>& listaCoppieNoDoppioni,
                                std::vector<int>& indiciRigheRimaste) {

    listaCoppieNoDoppioni.reserve(listaCoppie.size());
    indiciRigheRimaste.reserve(listaCoppie.size());

    std::unordered_map<std::string, int> hashMap;

    for (int i = 0; i < listaCoppie.size(); i++) {
        string key = (listaCoppie[i][0] <= listaCoppie[i][1] ?
            std::to_string(listaCoppie[i][0]) + std::to_string(listaCoppie[i][1])
            : std::to_string(listaCoppie[i][1]) + std::to_string(listaCoppie[i][0]));

        if (hashMap.count(key) == 0) {
            listaCoppieNoDoppioni.push_back({ listaCoppie[i][0], listaCoppie[i][1] });
            indiciRigheRimaste.push_back(i);
            hashMap[key] = 1;
        }
    }
}
