#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace sdv_loam
{

template<int mode>
void AccumulatedTopHessianSSE::addPoint(EFPoint* p, EnergyFunctional const * const ef, int tid)	// tid 0 = active, 1 = linearized, 2=marginalize
{


	assert(mode==0 || mode==1 || mode==2);

	VecCf dc = ef->cDeltaF;
	float dd = p->deltaF;

	float bd_acc=0;
	float Hdd_acc=0;
	VecCf Hcd_acc = VecCf::Zero();

	for(EFResidual* r : p->residualsAll)
	{
		if(mode==0) 
		{
			if(r->isLinearized || !r->isActive()) continue;
		}
		if(mode==1)
		{
			if(!r->isLinearized || !r->isActive()) continue;
		}
		if(mode==2)
		{
			if(!r->isActive()) continue;
			assert(r->isLinearized);
		}
		if(mode == 1)
		{
			printf("yeah I'm IN !");
		}

		RawResidualJacobian* rJ = r->J;
		int htIDX = r->hostIDX + r->targetIDX*nframes[tid]; 

		Mat16f dp = ef->adHTdeltaF[htIDX];

		Vec2f resApprox;

		if(mode==0)
			resApprox = rJ->resF;
		if(mode==2)
			resApprox = r->res_toZeroF;
		if(mode==1)
		{
			float Jp_delta_x = rJ->Jpdxi[0].dot(dp)+rJ->Jpdc[0].dot(dc)+rJ->Jpdd[0]*dd;
			float Jp_delta_y = rJ->Jpdxi[1].dot(dp)+rJ->Jpdc[1].dot(dc)+rJ->Jpdd[1]*dd;
			resApprox = r->res_toZeroF + Vec2f(Jp_delta_x, Jp_delta_y);
		}

		Vec2f JI_r = resApprox;
		float rr = resApprox.dot(resApprox);

		acc[tid][htIDX].update(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				1, 0, 1);

		acc[tid][htIDX].updateBotRight(
				0, 0, 0,
				0, 0,rr);

		acc[tid][htIDX].updateTopRight(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				0, 0,
				0, 0,
				JI_r[0], JI_r[1]);

		Vec2f Ji2_Jpdd = rJ->Jpdd;

		bd_acc +=  JI_r[0]*rJ->Jpdd[0] + JI_r[1]*rJ->Jpdd[1];
		Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
		Hcd_acc += rJ->Jpdc[0]*Ji2_Jpdd[0] + rJ->Jpdc[1]*Ji2_Jpdd[1];

		nres[tid]++;
	}

	if(mode==0)
	{
		p->Hdd_accAF = Hdd_acc;
		p->bd_accAF = bd_acc;
		p->Hcd_accAF = Hcd_acc;
	}
	if(mode==1 || mode==2)
	{
		p->Hdd_accLF = Hdd_acc;
		p->bd_accLF = bd_acc;
		p->Hcd_accLF = Hcd_acc;
	}
	if(mode==2)
	{
		p->Hcd_accAF.setZero();
		p->Hdd_accAF = 0;
		p->bd_accAF = 0;
	}

}

template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint* p, EnergyFunctional const * const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint* p, EnergyFunctional const * const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint* p, EnergyFunctional const * const ef, int tid);

void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const * const EF, bool usePrior, bool useDelta, int tid)
{
	H = MatXX::Zero(nframes[tid]*6+CPARS, nframes[tid]*6+CPARS);
	b = VecX::Zero(nframes[tid]*6+CPARS);

	for(int h=0;h<nframes[tid];h++)
		for(int t=0;t<nframes[tid];t++)
		{
			int hIdx = CPARS+h*6;
			int tIdx = CPARS+t*6;
			int aidx = h+nframes[tid]*t;

			acc[tid][aidx].finish();
			if(acc[tid][aidx].num==0) continue;

			MatPCPC accH = acc[tid][aidx].H.cast<double>();

			H.block<6,6>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<6,6>(CPARS,CPARS) * EF->adHost[aidx].transpose();

			H.block<6,6>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<6,6>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

			H.block<6,6>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<6,6>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

			H.block<6,CPARS>(hIdx,0).noalias() += EF->adHost[aidx] * accH.block<6,CPARS>(CPARS,0);

			H.block<6,CPARS>(tIdx,0).noalias() += EF->adTarget[aidx] * accH.block<6,CPARS>(CPARS,0);

			H.topLeftCorner<CPARS,CPARS>().noalias() += accH.block<CPARS,CPARS>(0,0);

			b.segment<6>(hIdx).noalias() += EF->adHost[aidx] * accH.block<6,1>(CPARS,8+CPARS);

			b.segment<6>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<6,1>(CPARS,8+CPARS);

			b.head<CPARS>().noalias() += accH.block<CPARS,1>(0,8+CPARS);
		}

	for(int h=0;h<nframes[tid];h++)
	{
		int hIdx = CPARS+h*6;
		H.block<CPARS,6>(0,hIdx).noalias() = H.block<6,CPARS>(hIdx,0).transpose();
		
		for(int t=h+1;t<nframes[tid];t++)
		{
			int tIdx = CPARS+t*6;
			H.block<6,6>(hIdx, tIdx).noalias() += H.block<6,6>(tIdx, hIdx).transpose();
			H.block<6,6>(tIdx, hIdx).noalias() = H.block<6,6>(hIdx, tIdx).transpose();
		}
	}


	if(usePrior)
	{
		assert(useDelta);
		H.diagonal().head<CPARS>() += EF->cPrior;
		b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
		for(int h=0;h<nframes[tid];h++)
		{
            H.diagonal().segment<6>(CPARS+h*6) += EF->frames[h]->prior;
            b.segment<6>(CPARS+h*6) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
		}
	}
}

void AccumulatedTopHessianSSE::stitchDoubleInternal(
		MatXX* H, VecX* b, EnergyFunctional const * const EF, bool usePrior,
		int min, int max, Vec10* stats, int tid)
{
	int toAggregate = NUM_THREADS;

	if(tid == -1) { toAggregate = 1; tid = 0; }
	if(min==max) return;


	for(int k=min;k<max;k++)
	{
		int h = k%nframes[0];
		int t = k/nframes[0];

		int hIdx = CPARS+h*6;
		int tIdx = CPARS+t*6;

		int aidx = h+nframes[0]*t;

		assert(aidx == k);

		MatPCPC accH = MatPCPC::Zero();

		for(int tid2=0;tid2 < toAggregate;tid2++)
		{
			acc[tid2][aidx].finish();
			if(acc[tid2][aidx].num==0) continue;
			accH += acc[tid2][aidx].H.cast<double>();
		}
		
		H[tid].block<6,6>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<6,6>(CPARS,CPARS) * EF->adHost[aidx].transpose();

		H[tid].block<6,6>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<6,6>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

		H[tid].block<6,6>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<6,6>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

		H[tid].block<6,CPARS>(hIdx,0).noalias() += EF->adHost[aidx] * accH.block<6,CPARS>(CPARS,0);

		H[tid].block<6,CPARS>(tIdx,0).noalias() += EF->adTarget[aidx] * accH.block<6,CPARS>(CPARS,0);

		H[tid].topLeftCorner<CPARS,CPARS>().noalias() += accH.block<CPARS,CPARS>(0,0);

		b[tid].segment<6>(hIdx).noalias() += EF->adHost[aidx] * accH.block<6,1>(CPARS,CPARS+8);

		b[tid].segment<6>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<6,1>(CPARS,CPARS+8);

		b[tid].head<CPARS>().noalias() += accH.block<CPARS,1>(0,CPARS+8);
	}

	// only do this on one thread.
	if(min==0 && usePrior)
	{
		H[tid].diagonal().head<CPARS>() += EF->cPrior;
		b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
		for(int h=0;h<nframes[tid];h++)
		{
            H[tid].diagonal().segment<6>(CPARS+h*6) += EF->frames[h]->prior;
            b[tid].segment<6>(CPARS+h*6) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
		}
	}
}



}


