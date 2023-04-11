#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace sdv_loam
{

PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;
	for(FrameHessian* fh : frameHessians)
	{
		if(fh != point->host)
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;

	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;
	float trueDepth = currentIdepth;

	if(point->isFromSensor == false)
	{
		for(int i=0;i<nres;i++)
		{
			lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth);
			residuals[i].state_state = residuals[i].state_NewState;
			residuals[i].state_energy = residuals[i].state_NewEnergy;
		}

		if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
		{
			if(print)
				printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres, lastHdd, lastEnergy);
			return 0;
		}

		if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
				nres, lastHdd,lastEnergy,currentIdepth);

		float lambda = 0.1;
		for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
		{
			float H = lastHdd;
			H *= 1+lambda;
			float step = (1.0/H) * lastbd;
			float newIdepth = currentIdepth - step;

			float newHdd=0; float newbd=0; float newEnergy=0;
			for(int i=0;i<nres;i++)
				newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

			if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
			{
				if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
						nres,
						newHdd,
						lastEnergy);
				return 0;
			}

			if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
					(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
					iteration,
					log10(lambda),
					"",
					lastEnergy, newEnergy, newIdepth);

			if(newEnergy < lastEnergy)
			{
				currentIdepth = newIdepth;
				lastHdd = newHdd;
				lastbd = newbd;
				lastEnergy = newEnergy;
				for(int i=0;i<nres;i++)
				{
					residuals[i].state_state = residuals[i].state_NewState;
					residuals[i].state_energy = residuals[i].state_NewEnergy;
				}

				lambda *= 0.5;
			}
			else
			{
				lambda *= 5;
			}

			if(fabsf(step) < 0.0001*currentIdepth)
				break;
		}
	}

	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);

		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		//! niubility
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	p->isFromSensor = point->isFromSensor;
	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;

	if(p->isFromSensor == true)
	{
		p->setIdepthZero(trueDepth);
		p->setIdepth(trueDepth);
	}
	else
	{
		p->setIdepthZero(currentIdepth);
		p->setIdepth(currentIdepth);
	}

	p->setPointStatus(PointHessian::ACTIVE);

	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN)
		{
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);

			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN);
			p->residuals.push_back(r);

			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}

}
