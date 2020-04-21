#include "MyUtility.h"
#include <string>
#include <unordered_map>
#include <fstream>

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



void MyUtility::writeKeyPoints(std::vector<cv::KeyPoint>& keypoints)
{
	ofstream myfile;
	myfile.open("keypoints.csv");

	for (auto& kp : keypoints)
	{
		myfile << kp.pt.x << ", " << kp.pt.y << ", " << kp.angle << ", " << kp.class_id << ", " << kp.octave << ", " << kp.response << ", " << kp.size << endl;
	}

	myfile.close();
}

void MyUtility::writeDescriptors(cv::Mat& descriptors)
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

void MyUtility::writeKnnMatches(std::vector<std::vector<cv::DMatch>>& knn_matches)
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


void MyUtility::writeMatches(std::vector<KeyPointsMatch>& matches)
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

