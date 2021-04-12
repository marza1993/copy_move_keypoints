#pragma once
#include "IClusterPoint.h"
#include <math.h>
#include <iostream>


class ClusterPoint2d : IClusterPoint
{

	int clusterID;

public:

	ClusterPoint2d(float x, float y) : x(x), y(y), clusterID(UNCLASSIFIED) {}

	float operator [] (int i)
	{
		return component(i);
	}

	float component(int i)
	{
		if (i > 1)
		{
			throw std::exception("l'indice passato sfora");
		}
		switch (i)
		{
		case 0:
			return x;
		case 1:
			return y;
		}
	}

	double distanceFrom(const IClusterPoint& otherPoint)
	{
		ClusterPoint2d& other = (ClusterPoint2d&)otherPoint;
		return sqrt(pow(component(0) - other.component(0), 2)
			+ pow(component(1) - other.component(1), 2));
	}

	void setClusterID(int clusterID)
	{
		this->clusterID = clusterID;
	}


	int getClusterID()
	{
		return clusterID;
	}

	bool equals(const IClusterPoint& otherPoint, double tolerance = 0)
	{
		ClusterPoint2d& other = (ClusterPoint2d&)otherPoint;
		return (abs(component(0) - other.component(0)) <= tolerance
			&& abs(component(1) - other.component(1)) <= tolerance);
	}

	float x, y;

};


class ClusterPoint3d : IClusterPoint
{

	int clusterID;

public:

	ClusterPoint3d(float x, float y, float z) : x(x), y(y), z(z), clusterID(UNCLASSIFIED) {}

	float operator [] (int i)
	{
		return component(i);
	}

	float component(int i)
	{
		if (i > 2)
		{
			throw std::exception("l'indice passato sfora");
		}
		switch (i)
		{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
	}

	double distanceFrom(const IClusterPoint& otherPoint)
	{
		ClusterPoint3d& other = (ClusterPoint3d&)otherPoint;
		return sqrt(pow(component(0) - other.component(0), 2)
			+ pow(component(1) - other.component(1), 2)
			+ pow(component(2) - other.component(2), 2));
	}

	void setClusterID(int clusterID)
	{
		this->clusterID = clusterID;
	}


	int getClusterID()
	{
		return clusterID;
	}

	bool equals(const IClusterPoint& otherPoint, double tolerance = 0)
	{
		ClusterPoint3d& other = (ClusterPoint3d&)otherPoint;
		return (abs(component(0) - other.component(0)) <= tolerance
			&& abs(component(1) - other.component(1)) <= tolerance
			&& abs(component(2) - other.component(2)) <= tolerance);
	}

	float x, y, z;

};
