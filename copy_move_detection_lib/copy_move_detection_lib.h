#pragma once


#ifdef COPYMOVEDETECTIONLIB_EXPORTS
#define COPY_MOVE_DETECTION_LIB_API __declspec(dllexport)
#else
#define COPY_MOVE_DETECTION_LIB_API __declspec(dllimport)
#endif

typedef unsigned char uchar;
typedef unsigned int uint;


// questa roba serve per alcune robe che usano la libreria (es: wrapper python)
extern "C" {

	namespace copy_move_det_lib
	{

		struct Point2DStruct
		{
			float x;
			float y;
		};

		struct KeyPointsMatchStruct
		{
			Point2DStruct* p1;
			Point2DStruct* p2;
			float descriptorsDistance;
			int isValid;
		};


		void COPY_MOVE_DETECTION_LIB_API SIFT_copy_move_detection(const uchar* input_img, const uint img_W, const uint img_H, 
																  const uint sogliaSIFT, const uint minPuntiIntorno,
																  const float sogliaLowe, const float eps, const float sogliaDescInCluster, 
																  int* resultForgedOrNot, KeyPointsMatchStruct** foundMatches, uint* N_found_matches,
																  const int getOutputImg = 0, uchar** outputImg = nullptr, 
																  uint* output_img_W = nullptr, uint* output_img_H = nullptr);

		void COPY_MOVE_DETECTION_LIB_API SIFT_copy_move_free_mem(KeyPointsMatchStruct** foundMatches, uint N_matches, uchar** outputImg = nullptr);

	}
}