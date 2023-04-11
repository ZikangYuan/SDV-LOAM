#pragma once

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

namespace sdv_loam
{

EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
{
	return (dxInterp*drescale * (t[0]-t[2]*u)
			+ dyInterp*drescale * (t[1]-t[2]*v))*SCALE_IDEPTH;
}

EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		float &Ku, float &Kv)
{
	Vec3f ptp = KRKi * Vec3f(u_pt,v_pt, 1) + Kt*idepth;
	Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];
	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}

EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		CalibHessian* const &HCalib,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
	KliP = Vec3f(
			(u_pt+dx-HCalib->cxl())*HCalib->fxli(),
			(v_pt+dy-HCalib->cyl())*HCalib->fyli(),
			1);

	Vec3f ptp = R * KliP + t*idepth;
	drescale = 1.0f/ptp[2];
	new_idepth = idepth*drescale;

	if(!(drescale>0)) return false;

	u = ptp[0] * drescale;
	v = ptp[1] * drescale;

	Ku = u*HCalib->fxl() + HCalib->cxl();
	Kv = v*HCalib->fyl() + HCalib->cyl();

	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}

EIGEN_STRONG_INLINE Vec3f point2world(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		const float cx, const float cy, const float fxi, const float fyi,
		const Mat33f &R, const Vec3f &t)
{
	Vec3f KliP = Vec3f(
			(u_pt+dx-cx)*fxi,
			(v_pt+dy-cy)*fyi,
			1);

	Vec3f ptRef = KliP / idepth;

	Vec3f ptWorld = R * ptRef + t;

	return ptWorld;
}

EIGEN_STRONG_INLINE bool world2frame(
		const Vec3f &ptWorld,
		const float cx, const float cy, const float fx, const float fy,
		const Mat33f &R, const Vec3f &t, Vec3f &ptFrame, float &Ku, float &Kv)
{
	ptFrame = R * ptWorld + t;

	Vec3f unitFrame;
	unitFrame[0] = ptFrame[0] / ptFrame[2]; unitFrame[1] = ptFrame[1] / ptFrame[2]; unitFrame[2] = ptFrame[2] / ptFrame[2];

	Ku = unitFrame[0] * fx + cx;
	Kv = unitFrame[1] * fy + cy;

	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}

EIGEN_STRONG_INLINE void pixel2unit(
		const float cx, const float cy, const float fxi, const float fyi,
		const float Ku, const float Kv, float &u, float &v)
{
	u = (Ku - cx) * fxi;
	v = (Kv - cy) * fyi;
}




}

