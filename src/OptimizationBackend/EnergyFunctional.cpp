#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace sdv_loam
{


bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;

	adHost = new Mat66[nFrames*nFrames];
	adTarget = new Mat66[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			FrameHessian* host = frames[h]->data;
			FrameHessian* target = frames[t]->data;

			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
			
			Mat66 AH = Mat66::Identity();
			Mat66 AT = Mat66::Identity();

			AH = -hostToTarget.Adj().transpose();
			AT = Mat66::Identity();

			AH.block<3,6>(0,0) *= SCALE_XI_TRANS;
			AH.block<3,6>(3,0) *= SCALE_XI_ROT;

			AT.block<3,6>(0,0) *= SCALE_XI_TRANS;
			AT.block<3,6>(3,0) *= SCALE_XI_ROT;

			adHost[h+t*nFrames] = AH;
			adTarget[h+t*nFrames] = AT;
		}
	cPrior = VecC::Constant(setting_initialCalibHessian);

	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;

	adHostF = new Mat66f[nFrames*nFrames];
	adTargetF = new Mat66f[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
			adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
		}

	cPriorF = cPrior.cast<float>();

	EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;

	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS);
	bM = VecX::Zero(CPARS);

	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
	currentLambda=0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames)
	{
		for(EFPoint* p : f->points)
		{
			for(EFResidual* r : p->residualsAll)
			{
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;



	delete accSSE_top_L;
	delete accSSE_top_A;
	delete accSSE_bot;
}

void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat16f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			int idx = h+t*nFrames;
			
			adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<6>().cast<float>().transpose() * adHostF[idx]
					        +frames[t]->data->get_state_minus_stateZero().head<6>().cast<float>().transpose() * adTargetF[idx];
		}

	cDeltaF = HCalib->value_minus_value_zero.cast<float>();

	for(EFFrame* f : frames)
	{	
		f->delta = f->data->get_state_minus_stateZero().head<6>();
		f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<6>();

		for(EFPoint* p : f->points)
			p->deltaF = p->data->idepth - p->data->idepth_zero;
	}

	EFDeltaValid = true;
}

void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
				accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);

		accSSE_top_A->stitchDoubleMT(red,H,b,this,true,true);
		resInA = accSSE_top_A->nres[0];
	}
	else
	{
		accSSE_top_A->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_A->addPoint<0>(p,this);

		accSSE_top_A->stitchDoubleMT(red,H,b,this,true,false);
		resInA = accSSE_top_A->nres[0];
	}
}

void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
				accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
		resInL = accSSE_top_L->nres[0];
	}
	else
	{
		accSSE_top_L->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_L->addPoint<1>(p,this);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
		resInL = accSSE_top_L->nres[0];
	}
}

void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
				accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_bot->stitchDoubleMT(red,H,b,this,true);
	}
	else
	{
		accSSE_bot->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_bot->addPoint(p, true);
		accSSE_bot->stitchDoubleMT(red, H, b,this,false);
	}
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*6);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>();

	Mat16f* xAd = new Mat16f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();

	for(EFFrame* h : frames)
	{
		h->data->step.head<6>() = - x.segment<6>(CPARS+6*h->idx);
		h->data->step.tail<4>().setZero();

		for(EFFrame* t : frames)
			xAd[nFrames*h->idx + t->idx] = xF.segment<6>(CPARS+6*h->idx).transpose() *   adHostF[h->idx+nFrames*t->idx]
			            + xF.segment<6>(CPARS+6*t->idx).transpose() * adTargetF[h->idx+nFrames*t->idx];
	}

	if(MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
						this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat16f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0)
		{
			p->data->step = 0;
			continue;
		}

		float b = p->bdSumF;
		b -= xc.dot(p->Hcd_accAF);

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * (r->JpJdF).head<6>();
		}

		if(p->data->isFromSensor==true)
			p->data->step = 0;
		else{
			p->data->step = - b*p->HdiF;
		}

		assert(std::isfinite(p->data->step));
	}
}

double EnergyFunctional::calcMEnergyF()
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2*bM + HM*delta);
}

void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{

	Accumulator11 E;
	E.initialize();
	VecCf dc = cDeltaF;

	for(int i=min;i<max;i++)
	{
		EFPoint* p = allPoints[i];
		float dd = p->deltaF;

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isLinearized || !r->isActive()) continue;

			Mat16f dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
			RawResidualJacobian* rJ = r->J;

			// compute Jp*delta
			float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())
						   +rJ->Jpdc[0].dot(dc)
						   +rJ->Jpdd[0]*dd;

			float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())
						   +rJ->Jpdc[1].dot(dc)
						   +rJ->Jpdd[1]*dd;

			Vec2f Jp_delta = Vec2f(Jp_delta_x_1, Jp_delta_y_1);
			float r0 = ((r->res_toZeroF.transpose()) * Jp_delta + Jp_delta.transpose() * r->res_toZeroF + Jp_delta.transpose() * Jp_delta)(0, 0);
			E.updateSingleNoShift((float)r0);
		}
		E.updateSingle(p->deltaF*p->deltaF*p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}

double EnergyFunctional::calcLEnergyF_MT()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;

	for(EFFrame* f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

	E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
			this, _1, _2, _3, _4), 0, allPoints.size(), 50);

	return E+red->stats[0];
}

EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

	nResiduals++;
	r->efResidual = efr;
	return efr;
}

EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size(); 
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;

	assert(HM.cols() == 6*nFrames+CPARS-6);

	bM.conservativeResize(6*nFrames+CPARS);
	HM.conservativeResize(6*nFrames+CPARS,6*nFrames+CPARS);

	bM.tail<6>().setZero();
	HM.rightCols<6>().setZero();
	HM.bottomRows<6>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib);
	makeIDX();

	for(EFFrame* fh2 : frames)
	{
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}

EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}

void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();

	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;

    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}

void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);

	int ndim = nFrames*6+CPARS-6;
	int odim = nFrames*6+CPARS;
	
	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*6+CPARS;
		int ntail = 6*(nFrames-fh->idx-1);
		assert((io+6+ntail) == nFrames*6+CPARS);

		Vec6 bTmp = bM.segment<6>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io,ntail) = tailTMP;
		bM.tail<6>() = bTmp;

		MatXX HtmpCol = HM.block(0,io,odim,6);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(6) = HtmpCol;

		MatXX HtmpRow = HM.block(io,0,6,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(6) = HtmpRow;
	}
	
    HM.bottomRightCorner<6,6>().diagonal() += fh->prior;
    bM.tail<6>() += fh->prior.cwiseProduct(fh->delta_prior);

	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();

	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	Mat66 hpi = HMScaled.bottomRightCorner<6,6>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);

	MatXX bli = HMScaled.bottomLeftCorner(6,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(6,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<6>();

	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);

	for(unsigned int i=fh->idx; i+1<frames.size();i++)
	{
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	assert((int)frames.size()*6+CPARS == (int)HM.rows());
	assert((int)frames.size()*6+CPARS == (int)HM.cols());
	assert((int)frames.size()*6+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);
	
	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}

void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	allPointsToMarg.clear();
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE)
			{
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}

	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);
	for(EFPoint* p : allPointsToMarg)
	{
		accSSE_top_A->addPoint<2>(p,this);

		accSSE_bot->addPoint(p,false);

		removePoint(p);
	}
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accSSE_top_A->stitchDouble(M,Mb,this,false,false);
	accSSE_bot->stitchDouble(Msc,Mbsc,this);

	resInM+= accSSE_top_A->nres[0];

	MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;

		if(!haveFirstFrame)
			orthogonalize(&b, &H);

	}
	
	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bM, &HM);

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP)
			{
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());

	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();

	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++)
	{
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}

	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose();

	MatXX NNpiT = N*Npi.transpose();
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());
	
	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;
}

void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc;
	VecX  bL_top, bA_top, bM_top, b_sc;

	accumulateAF_MT(HA_top, bA_top,multiThreading);
	
	accumulateLF_MT(HL_top, bL_top,multiThreading);

	accumulateSCF_MT(H_sc, b_sc,multiThreading);

	bM_top = (bM+ HM * getStitchedDeltaF());


	MatXX HFinal_top;
	VecX bFinal_top;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;

		MatXX HT_act =  HA_top - H_sc;
		VecX bT_act =   bA_top - b_sc;

		if(!haveFirstFrame)
			orthogonalize(&bT_act, &HT_act);

		HFinal_top = HT_act + HM;
		bFinal_top = bT_act + bM_top;



		lastHS = HFinal_top;
		lastbS = bFinal_top;

		for(int i=0;i<6*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);
	}
	else
	{
		HFinal_top = HA_top + HM - H_sc;
		bFinal_top = bA_top + bM_top - b_sc;
		lastHS = HFinal_top;
		lastbS = bFinal_top;

		for(int i=0;i<6*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);
	}

	VecX x;
	if(setting_solverMode & SOLVER_SVD)
	{
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;

		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		//! Hx=b --->  U∑V^T*x = b  --->  ∑V^T*x = U^T*b
		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }
			//! V^T*x = ∑^-1*U^T*b
			else Ub[i] /= S[i];
		}
		//! x = V*∑^-1*U^T*b
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;
	}
	else
	{
		VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}

	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
	{
		VecX xOld = x;
		orthogonalize(&x, 0);
	}


	lastX = x;

	currentLambda= lambda;
	resubstituteF_MT(x, HCalib,multiThreading);
	currentLambda=0;

}

void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points) 
		{
			allPoints.push_back(p);

			for(EFResidual* r : p->residualsAll)
			{
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
			}
		}


	EFIndicesValid=true;
}

VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*6); d.head<CPARS>() = cDeltaF.cast<double>();
	for(int h=0; h<nFrames; h++) d.segment<6>(CPARS+6*h) = frames[h]->delta;  
	return d;
}

}
