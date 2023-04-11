#pragma once
 
#include "util/NumType.h"
 
#include "FullSystem/HessianBlocks.h"
namespace sdv_loam
{

struct ImmaturePointTemporaryResidual
{
public:
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	FrameHessian* target;
};


enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good

	IPS_OOB,					// OOB: end tracking & marginalize!

	IPS_OUTLIER,				// energy too high: if happens again: outlier!

	IPS_SKIPPED,				// traced well and good (but not actually traced).

	IPS_BADCONDITION,			// not traced because of bad condition.
	IPS_UNINITIALIZED};			// not even traced once.


class ImmaturePoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color[MAX_RES_PER_POINT];
	float weights[MAX_RES_PER_POINT];

    float idepth_fromSensor;

	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	float u,v;
	FrameHessian* host;
	int idxInImmaturePoints;

	float quality;

	float my_type;

	float idepth_min;
	float idepth_max;

	bool isFromSensor;
	bool ck = false;

	enum PointType {CORNER, EDGELET};
	PointType type;
	float score;

	ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib);
	~ImmaturePoint();

	ImmaturePointStatus traceOn(FrameHessian* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, CalibHessian* HCalib, bool debugPrint=false);

	ImmaturePointStatus lastTraceStatus;
	Vec2f lastTraceUV;
	float lastTracePixelInterval;

	float idepth_GT;

	double linearizeResidual(
	 		CalibHessian *  HCalib, const float outlierTHSlack,
	 		ImmaturePointTemporaryResidual* tmpRes,
	 		float &Hdd, float &bd,
	 		float idepth);

	float getdPixdd(
			CalibHessian *  HCalib,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	float calcResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

private:
};
}

