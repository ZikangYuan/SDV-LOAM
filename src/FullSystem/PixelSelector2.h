#pragma once
 
#include "util/NumType.h"
#include <queue>
#include "FullSystem/HessianBlocks.h"

namespace sdv_loam
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};


class FrameHessian;

class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int makeMaps(
		    const FrameHessian* const fh,
			float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);

	int makeMapsFromLidar(const FrameHessian* const fh, float* map_out, float density, int recursionsLeft, 
		bool plot, float thFactor, std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel);

	PixelSelector(int w, int h);
	~PixelSelector();
	int currentPotential;

	bool allowFast;
	void makeHists(const FrameHessian* const fh);
private:
	Eigen::Vector3i selectFromLidar(const FrameHessian* const fh, float* map_out, int pot, float thFactor, 
		std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel);

	Eigen::Vector3i select(const FrameHessian* const fh,
			float* map_out, int pot, float thFactor=1);


	unsigned char* randomPattern;


	int* gradHist;
	float* ths;
	float* thsSmoothed;
	int thsStep;
	const FrameHessian* gradHistFrame;
};

}

