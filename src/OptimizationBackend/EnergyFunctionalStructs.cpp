#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace sdv_loam
{


void EFResidual::takeDataF()
{
	std::swap<RawResidualJacobian*>(J, data->J);

	for(int i=0;i<6;i++){
		JpJdF[i] = J->Jpdxi[0][i]*J->Jpdd[0] + J->Jpdxi[1][i] * J->Jpdd[1];
	}
	Vec2f tmp;
	tmp.setZero();
	JpJdF.segment<2>(6) = tmp;
}

void EFFrame::takeData()
{
	prior = data->getPrior().head<6>();
	delta = data->get_state_minus_stateZero().head<6>();
	delta_prior =  (data->get_state() - data->getPriorZero()).head<6>();

	assert(data->frameID != -1);

	frameID = data->frameID;
}

void EFPoint::takeData()
{
	priorF = data->hasDepthPrior ? setting_idepthFixPrior*SCALE_IDEPTH*SCALE_IDEPTH : 0;
	if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) priorF=0;

	deltaF = data->idepth - data->idepth_zero;
}

void EFResidual::fixLinearizationF(EnergyFunctional* ef)
{
	Vec6f dp = ef->adHTdeltaF[hostIDX+ef->nFrames*targetIDX];

	float Jp_delta_x = J->Jpdxi[0].dot(dp)+J->Jpdc[0].dot(ef->cDeltaF)+J->Jpdd[0]*(point->deltaF);
	float Jp_delta_y = J->Jpdxi[1].dot(dp)+J->Jpdc[1].dot(ef->cDeltaF)+J->Jpdd[1]*(point->deltaF);
	res_toZeroF = J->resF - Vec2f(Jp_delta_x, Jp_delta_y);

	isLinearized = true;
}

}
