#pragma once

#define UNCLASSIFIED -2
#define NOISE -1


class IClusterPoint
{
public:

	virtual ~IClusterPoint() {};
	virtual float operator [] (int i) = 0;
	virtual float component(int i) const = 0;
	virtual double distanceFrom(const IClusterPoint& otherPoint) = 0;
	virtual bool equals(const IClusterPoint& otherPoint, double tolerance = 0) = 0;
	virtual void setClusterID(int clusterID) = 0;
	virtual int getClusterID() = 0;

};

