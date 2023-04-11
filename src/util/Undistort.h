#pragma once

#include "util/ImageAndExposure.h"
#include "util/MinimalImage.h"
#include "util/NumType.h"
#include "Eigen/Core"

namespace sdv_loam
{


class PhotometricUndistorter
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	PhotometricUndistorter(std::string file, std::string noiseImage, std::string vignetteImage, int w_, int h_);
	~PhotometricUndistorter();

	// removes readout noise, and converts to irradiance.
	// affine normalizes values to 0 <= I < 256.
	// raw irradiance = a*I + b.
	// output will be written in [output].
	template<typename T> void processFrame(T* image_in, float exposure_time, float factor=1);
	void unMapFloatImage(float* image);

	ImageAndExposure* output;

	float* getG() {if(!valid) return 0; else return G;};
private:
    float G[256*256];
    int GDepth; 
	float* vignetteMap;
	float* vignetteMapInv;
	int w,h;
	bool valid;
};


class Undistort
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual ~Undistort();

	virtual void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const = 0;

	
	inline const Mat33 getK() const {return K;};
	inline const Eigen::Vector2i getSize() const {return Eigen::Vector2i(w,h);};
	inline const VecX getOriginalParameter() const {return parsOrg;};
	inline const Eigen::Vector2i getOriginalSize() {return Eigen::Vector2i(wOrg,hOrg);};
	inline bool isValid() {return valid;};

	template<typename T>
	ImageAndExposure* undistort(const MinimalImage<T>* image_raw, float exposure=0, double timestamp=0, float factor=1) const;

	static Undistort* getUndistorterForFile(std::string configFilename, std::string gammaFilename, std::string vignetteFilename);

	void loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage);

	PhotometricUndistorter* photometricUndist;

protected:
    int w, h, wOrg, hOrg, wUp, hUp;
    int upsampleUndistFactor;
	Mat33 K;
	VecX parsOrg;
	bool valid;
	bool passthrough;

	float* remapX;
	float* remapY;

	void applyBlurNoise(float* img) const;

	void makeOptimalK_crop();
	void makeOptimalK_full();

	void readFromFile(const char* configFileName, int nPars, std::string prefix = "");
};

class UndistortFOV : public Undistort
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    UndistortFOV(const char* configFileName, bool noprefix);
	~UndistortFOV();
	void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const;

};

class UndistortRadTan : public Undistort
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortRadTan(const char* configFileName, bool noprefix);
    ~UndistortRadTan();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const;

};

class UndistortEquidistant : public Undistort
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortEquidistant(const char* configFileName, bool noprefix);
    ~UndistortEquidistant();
    void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const;

};

class UndistortPinhole : public Undistort
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortPinhole(const char* configFileName, bool noprefix);
	~UndistortPinhole();
	void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const;

private:
	float inputCalibration[8];
};

class UndistortKB : public Undistort
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortKB(const char* configFileName, bool noprefix);
	~UndistortKB();
	void distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const;

};

}

