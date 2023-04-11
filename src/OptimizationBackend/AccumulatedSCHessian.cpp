#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace sdv_loam
{

void AccumulatedSCHessianSSE::addPoint(EFPoint* p, bool shiftPriorToZero, int tid)
{
	int ngoodres = 0;
	for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
	if(ngoodres==0)
	{
		p->HdiF=0;
		p->bdSumF=0;
		p->data->idepth_hessian=0;
		p->data->maxRelBaseline=0;
		return;
	}

	float H = p->Hdd_accAF+p->Hdd_accLF+p->priorF;

	if(H < 1e-10) H = 1e-10;

	p->data->idepth_hessian=H;
	p->HdiF = 1.0 / H;

	p->bdSumF = p->bd_accAF + p->bd_accLF;

	if(shiftPriorToZero) p->bdSumF += p->priorF*p->deltaF;
	
	VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;

	if(p->data->isFromSensor == true)
		return;

	//* schur complement
	//! Hcd * Hdd_inv * Hcd^`T
	accHcc[tid].update(Hcd,Hcd,p->HdiF);
	//! Hcd * Hdd_inv * bd
	accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

	int nFrames2 = nframes[tid]*nframes[tid];
	for(EFResidual* r1 : p->residualsAll)
	{
		if(!r1->isActive()) continue;
		int r1ht = r1->hostIDX + r1->targetIDX*nframes[tid];

		for(EFResidual* r2 : p->residualsAll)
		{
			if(!r2->isActive()) continue;

			accD[tid][r1ht+r2->targetIDX*nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
		}
		//!< Hfd * Hdd_inv * Hcd^T
		accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
		//! Hfd * Hdd_inv * bd
		accEB[tid][r1ht].update(r1->JpJdF,p->HdiF*p->bdSumF);
	}
}

void AccumulatedSCHessianSSE::stitchDoubleInternal(
		MatXX* H, VecX* b, EnergyFunctional const * const EF,
		int min, int max, Vec10* stats, int tid)
{
	int toAggregate = NUM_THREADS;
	if(tid == -1) { toAggregate = 1; tid = 0; }	// special case: if we dont do multithreading, dont aggregate.
	if(min==max) return;


	int nf = nframes[0];
	int nframes2 = nf*nf;

	for(int k=min;k<max;k++)
	{
		int i = k%nf;
		int j = k/nf;

		int iIdx = CPARS+i*6;
		int jIdx = CPARS+j*6;
		int ijIdx = i+nf*j;

		Mat8C Hpc = Mat8C::Zero();
		Vec8 bp = Vec8::Zero();

		for(int tid2=0;tid2 < toAggregate;tid2++)
		{
			accE[tid2][ijIdx].finish();
			accEB[tid2][ijIdx].finish();
			Hpc += accE[tid2][ijIdx].A1m.cast<double>();
			bp += accEB[tid2][ijIdx].A1m.cast<double>();
		}
		
		H[tid].block<6,CPARS>(iIdx,0) += EF->adHost[ijIdx] * Hpc.block<6,CPARS>(0, 0);
		H[tid].block<6,CPARS>(jIdx,0) += EF->adTarget[ijIdx] * Hpc.block<6,CPARS>(0, 0);

		b[tid].segment<6>(iIdx) += EF->adHost[ijIdx] * bp.head<6>();
		b[tid].segment<6>(jIdx) += EF->adTarget[ijIdx] * bp.head<6>();
		
		for(int k=0;k<nf;k++)
		{
			int kIdx = CPARS+k*6;
			int ijkIdx = ijIdx + k*nframes2;
			int ikIdx = i+nf*k;

			Mat88 accDM = Mat88::Zero();

			for(int tid2=0;tid2 < toAggregate;tid2++)
			{
				accD[tid2][ijkIdx].finish();
				if(accD[tid2][ijkIdx].num == 0) continue;
				accDM += accD[tid2][ijkIdx].A1m.cast<double>();
			}

			H[tid].block<6,6>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM.block<6,6>(0, 0) * EF->adHost[ikIdx].transpose();
			H[tid].block<6,6>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM.block<6,6>(0, 0) * EF->adTarget[ikIdx].transpose();
			H[tid].block<6,6>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM.block<6,6>(0, 0) * EF->adHost[ikIdx].transpose();
			H[tid].block<6,6>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM.block<6,6>(0, 0) * EF->adTarget[ikIdx].transpose();
		}
	}

	if(min==0)
	{
		for(int tid2=0;tid2 < toAggregate;tid2++)
		{
			accHcc[tid2].finish();
			accbc[tid2].finish();

			H[tid].topLeftCorner<CPARS,CPARS>() += accHcc[tid2].A1m.cast<double>();
			b[tid].head<CPARS>() += accbc[tid2].A1m.cast<double>();
		}
	}
}

void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const * const EF, int tid)
{

	int nf = nframes[0];
	int nframes2 = nf*nf;

	H = MatXX::Zero(nf*6+CPARS, nf*6+CPARS);
	b = VecX::Zero(nf*6+CPARS);

	for(int i=0;i<nf;i++)
		for(int j=0;j<nf;j++)
		{
			int iIdx = CPARS+i*6;
			int jIdx = CPARS+j*6;
			int ijIdx = i+nf*j;

			accE[tid][ijIdx].finish();
			accEB[tid][ijIdx].finish();

			Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();
			Vec8 accEBV = accEB[tid][ijIdx].A1m.cast<double>();

			H.block<6,CPARS>(iIdx,0) += EF->adHost[ijIdx] * accEM.block<6,CPARS>(0, 0);
			H.block<6,CPARS>(jIdx,0) += EF->adTarget[ijIdx] * accEM.block<6,CPARS>(0, 0);

			b.segment<6>(iIdx) += EF->adHost[ijIdx] * accEBV.head<6>();
			b.segment<6>(jIdx) += EF->adTarget[ijIdx] * accEBV.head<6>();
			
			for(int k=0;k<nf;k++)
			{
				int kIdx = CPARS+k*6;
				int ijkIdx = ijIdx + k*nframes2;
				int ikIdx = i+nf*k;

				accD[tid][ijkIdx].finish();
				if(accD[tid][ijkIdx].num == 0) continue;
				Mat88 accDM = accD[tid][ijkIdx].A1m.cast<double>();

				H.block<6,6>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM.block<6,6>(0, 0) * EF->adHost[ikIdx].transpose();

				H.block<6,6>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM.block<6,6>(0, 0) * EF->adTarget[ikIdx].transpose();

				H.block<6,6>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM.block<6,6>(0, 0) * EF->adHost[ikIdx].transpose();

				H.block<6,6>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM.block<6,6>(0, 0) * EF->adTarget[ikIdx].transpose();
			}
		}

	accHcc[tid].finish();
	accbc[tid].finish();
	H.topLeftCorner<CPARS,CPARS>() = accHcc[tid].A1m.cast<double>();
	b.head<CPARS>() = accbc[tid].A1m.cast<double>();

	// ----- new: copy transposed parts for calibration only.
	for(int h=0;h<nf;h++)
	{
		int hIdx = CPARS+h*6;
		H.block<CPARS,6>(0,hIdx).noalias() = H.block<6,CPARS>(hIdx,0).transpose();
	}
}

}
