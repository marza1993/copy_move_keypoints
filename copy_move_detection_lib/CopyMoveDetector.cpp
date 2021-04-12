#include "pch.h"

#include "CopyMoveDetector.h"
#include <iostream>

CopyMoveDetector::CopyMoveDetector()
{
	this->isSetInputImg = false;
	this->visualizeElab = true;
	this->forgedOrNot = false;
	this->isDetectionOk = false;
	this->isOutputImgDrawn = false;
}

void CopyMoveDetector::setInputImg(const cv::Mat& inputImg)
{
	if (!inputImg.empty())
	{
		this->inputImg = &inputImg;
		this->isSetInputImg = true;
	}
	else
	{
		std::cout << "The input image is empty!" << std::endl;
	}
}

bool CopyMoveDetector::getIsForged()
{
	return forgedOrNot;
}

bool CopyMoveDetector::getIsDetectionOk()
{
	return isDetectionOk;
}

bool CopyMoveDetector::getOuputImg(cv::Mat& outputImg)
{
	if (isOutputImgDrawn)
	{
		outputImg = this->outputImg;
		return true;
	}

	if (isDetectionOk)
	{
		drawOutputImg();
		return getOuputImg(outputImg);
	}
	std::cout << "non è stato invocato il metodo detect o questo non è andato a buon fine. Impossibile generare l'immagine di output" << std::endl;
	return false;
}

void CopyMoveDetector::setVisualizeElab(bool value)
{
	this->visualizeElab = value;
}
