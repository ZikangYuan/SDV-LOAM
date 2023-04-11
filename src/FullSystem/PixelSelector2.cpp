#include "FullSystem/PixelSelector2.h"
#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace sdv_loam
{


PixelSelector::PixelSelector(int w, int h)
{
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

	currentPotential=3;

	gradHist = new int[100*(1+w/32)*(1+h/32)];
	ths = new float[(w/32)*(h/32)+100];
	thsSmoothed = new float[(w/32)*(h/32)+100];

	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

int computeHistQuantil(int* hist, float below)
{
	int th = hist[0]*below+0.5f;
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return 90;
}

void PixelSelector::makeHists(const FrameHessian* const fh)
{
	gradHistFrame = fh;
	float * mapmax0 = fh->absSquaredGrad[0];

	// weight and height
	int w = wG[0];
	int h = hG[0];
	
	int w32 = w/32;
	int h32 = h/32;
	thsStep = w32;

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float* map0 = mapmax0+32*x+32*y*w;
			int* hist0 = gradHist;
			memset(hist0,0,sizeof(int)*50);

			for(int j=0;j<32;j++) for(int i=0;i<32;i++)
			{
				int it = i+32*x;
				int jt = j+32*y;
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
				int g = sqrtf(map0[i+j*w]);
				if(g>48) g=48;
				hist0[g+1]++;
				hist0[0]++;
			}

			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
		}

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		}
}

int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;

	{
		if(fh != gradHistFrame) makeHists(fh);

		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

		// sub-select!
		numHave = n[0]+n[1]+n[2];
		quotia = numWant / numHave;

		// by default we want to over-sample by 40% just to be sure.
		float K = numHave * (currentPotential+1) * (currentPotential+1);
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

			currentPotential = idealPotential;

			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor); //递归
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	int numHaveSub = numHave;
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

	currentPotential = idealPotential;

	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}

		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
	}

	return numHaveSub;
}

Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{
	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];

	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};

	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));

	float dw1 = setting_gradDownweightPerLevel;
	float dw2 = dw1*dw1;

	int n3=0, n2=0, n4=0;

	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{	  
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;

		Vec2f dir4 = directions[randomPattern[n2] & 0xF];

		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;
			int y34 = y3+y4;

			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];  

			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];

				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					int idx = x1+x234 + w*(y1+y234);
					int xf = x1+x234;
					int yf = y1+y234;

					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;

					
					float ag0 = mapmax0[idx];
					if(ag0 > pixelTH0*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));
						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					
					if(bestIdx3==-2) continue;

					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];

					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}

				if(bestIdx2>0)
				{
					map_out[bestIdx2] = 1;
					bestVal3 = 1e10;
					n2++;
				}
			}

			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}

	return Eigen::Vector3i(n2,n3,n4);
}

int PixelSelector::makeMapsFromLidar(const FrameHessian* const fh, float* map_out, float density, int recursionsLeft, 
		bool plot, float thFactor, std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel)
{
	float numHave=0;
	float numWant=density;
	float quotia;

	int idealPotential = currentPotential;
	{

		if(fh != gradHistFrame) makeHists(fh);

		Eigen::Vector3i n = this->selectFromLidar(fh, map_out, currentPotential, thFactor, vCloudPixel);

		// sub-select!
		numHave = n[0]+n[1]+n[2];

		quotia = numWant / numHave;

		float K = numHave * (currentPotential+1) * (currentPotential+1);
		idealPotential = sqrtf(K/numWant)-1;
		if(idealPotential<1) idealPotential=1;

		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

			currentPotential = idealPotential;

			return makeMapsFromLidar(fh,map_out, density, recursionsLeft-1, plot,thFactor, vCloudPixel);
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!
			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

			currentPotential = idealPotential;
			return makeMapsFromLidar(fh,map_out, density, recursionsLeft-1, plot,thFactor, vCloudPixel);

		}
	}

	int numHaveSub = numHave;

	if(quotia < 0.95)
	{
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<vCloudPixel.size();i++)
		{
			if(map_out[i] != 0)
			{
			    rn = (int)(vCloudPixel[i](0, 0) + vCloudPixel[i](1, 0) * wG[0]);

				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

	currentPotential = idealPotential;

	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i = 0; i < vCloudPixel.size(); i++)
		{
			int idx = vCloudPixel[i](0, 0) + w*vCloudPixel[i](1, 0);
			float c = fh->dI[idx][0]*0.7;
			if(c>255) c=255;
			img.at(idx) = Vec3b(c,c,c);
		}

		IOWrap::displayImage("Selector Image", &img);

		for(int i = 0; i < vCloudPixel.size(); i++)
		{
			int idx = vCloudPixel[i](0, 0) + w*vCloudPixel[i](1, 0);
			if(map_out[idx] == 1)
					img.setPixelCirc(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),Vec3b(0,255,0));
				else if(map_out[idx] == 2)
					img.setPixelCirc(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),Vec3b(255,0,0));
				else if(map_out[idx] == 4)
					img.setPixelCirc(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),Vec3b(0,0,255));
		}

		IOWrap::displayImage("Selector Pixels", &img);
	}

	return numHaveSub;
}

Eigen::Vector3i PixelSelector::selectFromLidar(const FrameHessian* const fh, float* map_out, int pot, float thFactor, 
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel)
{
	std::vector<std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>>> vLvl0;
	std::vector<std::vector<int>> vIndex0;

	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];

	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};

    int numPotH = (h%pot==0) ? h/pot : h/pot+1;
    int numPotW = (w%pot==0) ? w/pot : w/pot+1;

	for(int i = 0; i < numPotH; i++)
		for(int j = 0; j < numPotW; j++)
		{
			std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>> tempPt;
			vLvl0.push_back(tempPt);
			std::vector<int> tempIndex;
			vIndex0.push_back(tempIndex);
		}

	for(int i = 0; i < vCloudPixel.size(); i++)
	{
	    int indexX = (int)vCloudPixel[i](0, 0) / pot;
	    int indexY = (int)vCloudPixel[i](1, 0) / pot;

	    vLvl0[indexY * numPotW + indexX].push_back(Vec2f((float)vCloudPixel[i](0, 0), (float)vCloudPixel[i](1, 0)));
	    vIndex0[indexY * numPotW + indexX].push_back(i);
	}

	memset(map_out, 0, (vCloudPixel.size()) * sizeof(float));

	float dw1 = setting_gradDownweightPerLevel;
	float dw2 = dw1*dw1;

	int n3=0, n2=0, n4=0;

	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{	  
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;

		Vec2f dir4 = directions[randomPattern[n2] & 0xF];

		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;
			int y34 = y3+y4;

			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];  

			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];

				int i = (y4/pot + y3/pot + y2/pot)*numPotW + (x4/pot + x3/pot + x2/pot);
				for(int j = 0; j < vLvl0[i].size(); j++)
				{
				    assert(vLvl0[i][j](0, 0) < w);
					assert(vLvl0[i][j](1, 0) < h);

					int idx = vLvl0[i][j](0, 0) + w * vLvl0[i][j](1, 0);
					if(vLvl0[i][j](0, 0)<4 || vLvl0[i][j](0, 0)>=w-5 || vLvl0[i][j](1, 0)<4 || vLvl0[i][j](1, 0)>h-4) continue;

					float pixelTH0 = thsSmoothed[((int)vLvl0[i][j](0, 0)>>5) + ((int)vLvl0[i][j](1, 0)>>5) * thsStep];
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;

					float ag0 = mapmax0[idx];
					if(ag0 > pixelTH0*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));

						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = vIndex0[i][j]; bestIdx3 = -2; bestIdx4 = -2;}
					}
					if(bestIdx3==-2) continue;

					float ag1 = mapmax1[(int)(vLvl0[i][j](0, 0) * 0.5f + 0.25f) + (int)(vLvl0[i][j](1, 0) * 0.5f + 0.25f) * w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = vIndex0[i][j]; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					float ag2 = mapmax2[(int)(vLvl0[i][j](0, 0) * 0.25f + 0.125) + (int)(vLvl0[i][j](1, 0) * 0.25f + 0.125) * w2];

					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = vIndex0[i][j]; }
					}
				}
				if(bestIdx2 > 0)
				{
					map_out[bestIdx2] = 1;
					bestVal3 = 1e10;
					n2++;
				}
			}
			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}

    return Eigen::Vector3i(n2,n3,n4);
}
}

