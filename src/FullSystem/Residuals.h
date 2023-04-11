#pragma once
 
#include "util/globalCalib.h"
#include "vector"
 
#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "util/globalFuncs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace sdv_loam
{
class PointHessian;
class FrameHessian;
class CalibHessian;

class EFResidual;

enum ResLocation {ACTIVE=0, LINEARIZED, MARGINALIZED, NONE};
enum ResState {IN=0, OOB, OUTLIER};

struct FullJacRowT
{
	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

class PointFrameResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	EFResidual* efResidual;

	static int instanceCounter;


	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	double state_NewEnergyWithOutlier;


	void setState(ResState s) {state_state = s;}


	PointHessian* point;
	FrameHessian* host;
	FrameHessian* target;
	RawResidualJacobian* J;

	void findMatches();
	bool hasMatcher = false;
	Eigen::Vector2d matcher;

	bool isNew;


	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
	Vec3f centerProjectedTo;

	~PointFrameResidual();
	PointFrameResidual();
	PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_);
	double linearize(CalibHessian* HCalib);


	void resetOOB()
	{
		state_NewEnergy = state_energy = 0;
		state_NewState = ResState::OUTLIER;

		setState(ResState::IN);
	};
	void applyRes( bool copyJacobians);

	void debugPlot();

	void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
};
}

