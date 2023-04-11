#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include "IOWrapper/Pangolin/PangolinViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "opencv2/highgui/highgui.hpp"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>

typedef pcl::PointXYZI  PointType;

std::string vignette = "";
std::string gammaCalib = "";
std::string calib = "";
std::string resultPath = "";
std::string pathSensorPrameter = "";
std::string imgTopic = "";
std::string lidarTopic = "";

double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;
bool preload=false;
bool useSampleOutput=false;

ImageFolderReader* reader = NULL;
FullSystem* fullSystem = NULL;
bool firstFlag = true;
double firstFrameTime;
bool initialization = false;
double initialtimestamp = 0.0;
double timestamp = 0.0;
int currentId = 0;
struct timeval tv_start;
clock_t started;
double sInitializerOffset=0;

pcl::PointCloud<PointType>::Ptr laserCloudIn;

pcl::PointCloud<PointType>::Ptr fullCloud;
pcl::PointCloud<PointType>::Ptr fullInfoCloud;

pcl::PointCloud<PointType>::Ptr groundCloud;
pcl::PointCloud<PointType>::Ptr segmentedCloud;
pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
pcl::PointCloud<PointType>::Ptr outlierCloud;

PointType nanPoint;

cv::Mat rangeMat;
cv::Mat labelMat;
cv::Mat groundMat;
int labelCount;

float startOrientation;
float endOrientation;

std::vector<std::pair<int8_t, int8_t> > neighborIterator;

uint16_t *allPushedIndX;
uint16_t *allPushedIndY;

uint16_t *queueIndX;
uint16_t *queueIndY;

//Vel 64 Parameters
extern const int N_SCAN = 64;
extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 0.2;
extern const float ang_res_y = 0.427;
extern const float ang_bottom = 24.9;
extern const int groundScanInd = 50;

extern const bool loopClosureEnableFlag = false;
extern const double mappingProcessInterval = 0.3;

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;

extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = 60.0/180.0*M_PI;
extern const int segmentValidPointNum = 5;
extern const int segmentValidLineNum = 3;
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;

void allocateMemory(){
    laserCloudIn.reset(new pcl::PointCloud<PointType>());

    fullCloud.reset(new pcl::PointCloud<PointType>());
    fullInfoCloud.reset(new pcl::PointCloud<PointType>());

    groundCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
    outlierCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);
    fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

    allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
}

void resetParameters(){
    laserCloudIn->clear();
    groundCloud->clear();
    segmentedCloud->clear();
    segmentedCloudPure->clear();
    outlierCloud->clear();

    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

    labelCount = 1;

    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
}

int mode=0;

bool firstRosSpin=false;

using namespace sdv_loam;

void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}

void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}




void parseArgument(ros::NodeHandle &n)
{
	int option;
	float foption;
	char buf[1000];

	int sampleoutput_msg;
    if(n.getParam("sampleoutput", sampleoutput_msg))
    {
        if(sampleoutput_msg==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
    }

    int quiet_msg;
    if(n.getParam("quiet", quiet_msg))
    {
        if(quiet_msg==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
    }

    int preset_msg;
    if(n.getParam("preset", preset_msg))
	{
		settingsDefault(preset_msg);
	}

    int rec_msg;
	if(n.getParam("rec", rec_msg))
	{
		if(rec_msg==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
	}

    int noros_msg;
	if(n.getParam("noros", noros_msg))
	{
		if(noros_msg==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
	}

    int nolog_msg;
	if(n.getParam("nolog", nolog_msg))
	{
		if(nolog_msg==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
	}

	int reverse_msg;
	if(n.getParam("reverse", reverse_msg))
	{
		if(reverse_msg==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
	}

	int nogui_msg;
	if(n.getParam("nogui", nogui_msg))
	{
		if(nogui_msg==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
	}

	int nomt_msg;
	if(n.getParam("nomt", nomt_msg))
	{
		if(nomt_msg==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
	}

	int prefetch_msg;
	if(n.getParam("prefetch", prefetch_msg))
	{
		if(prefetch_msg==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
	}

	int start_msg;
	if(n.getParam("start", start_msg))
	{
		start = start_msg;
		printf("START AT %d!\n",start);
	}

	int end_msg;
	if(n.getParam("end", end_msg))
	{
		end = end_msg;
		printf("END AT %d!\n",start);
	}

	std::string imgTopic_msg;
	if(n.getParam("imgTopic", imgTopic_msg))
	{
		imgTopic = imgTopic_msg;
		printf("loading images from topic %s!\n", imgTopic.c_str());
	}

	std::string lidarTopic_msg;
	if(n.getParam("lidarTopic", lidarTopic_msg))
	{
		lidarTopic = lidarTopic_msg;
		printf("loading images from topic %s!\n", lidarTopic.c_str());
	}

    std::string calib_msg;
	if(n.getParam("calib", calib_msg))
	{
		calib = calib_msg;
		printf("loading calibration from %s!\n", calib.c_str());
	}

	std::string pathSensorPrameter_msg;
	if(n.getParam("pathSensorPrameter", pathSensorPrameter_msg))
	{
		pathSensorPrameter = pathSensorPrameter_msg;
		printf("loading the prameters of camera and lidar from %s!\n", pathSensorPrameter.c_str());
	}

	std::string resultPath_msg;
	if(n.getParam("resultPath", resultPath_msg))
	{
		resultPath = resultPath_msg;
		printf("save result at %s!\n", resultPath.c_str());
	}

    std::string vignette_msg;
	if(n.getParam("vignette", vignette_msg))
	{
		vignette = vignette_msg;
		printf("loading vignette from %s!\n", vignette.c_str());
	}

    std::string gamma_msg;
	if(n.getParam("gamma", gamma_msg))
	{
		gammaCalib = gamma_msg;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
	}

    float rescale_msg;
	if(n.getParam("rescale", rescale_msg))
	{
		rescale = rescale_msg;
		printf("RESCALE %f!\n", rescale);
	}

    float speed_msg;
	if(n.getParam("speed", speed_msg))
	{
		playbackSpeed = speed_msg;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
	}

    int save_msg;
	if(n.getParam("save", save_msg))
	{
		if(save_msg==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

    int mode_msg;
	if(n.getParam("mode", mode_msg))
	{

		mode = mode_msg;
		if(mode_msg==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(mode_msg==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0;
			setting_affineOptModeB = 0;
		}
		if(mode_msg==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1;
			setting_affineOptModeB = -1;
            setting_minGradHistAdd=3;
		}
	}
}

void process()
{
	if(fullSystem->qTimeImg.empty() || fullSystem->qCloudPixel.empty())
		return;

	if(fabs(fullSystem->qTimeImg.front() - fullSystem->qTimeLidarCloud.front()) > 0.01)
		return;

    cv::Mat currentFrame = fullSystem->qImg.front();
    timestamp = fullSystem->qTimeImg.front();

    if(currentId == 0)
    	initialtimestamp = timestamp;

    if(!fullSystem->initialized)
    {
        gettimeofday(&tv_start, NULL);
        started = clock();
        sInitializerOffset = timestamp - initialtimestamp;
    }

    if(firstFlag)
    {
        firstFrameTime = timestamp;
        firstFlag = false;
    }

    ImageAndExposure* img;
    img = reader->getRosImage(currentFrame, timestamp);

    bool skipFrame=false;

    if(!skipFrame) fullSystem->addActiveFrame(img, currentId);

    currentId++;

    delete img;
    fullSystem->qImg.pop();
    fullSystem->qTimeImg.pop();
    fullSystem->qCloudPixel.pop();
    fullSystem->qTimeLidarCloud.pop();

    if(fullSystem->initFailed || setting_fullResetRequested)
    {
        if(currentId < 250 || setting_fullResetRequested)
        {
            printf("RESETTING!\n");

            std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
            delete fullSystem;

            for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

            fullSystem = new FullSystem();
            fullSystem->setGammaFunction(reader->getPhotometricGamma());
            fullSystem->linearizeOperation = (playbackSpeed==0);
            fullSystem->outputWrapper = wraps;

            setting_fullResetRequested=false;
        }
    }

    if(fullSystem->isLost)
    {
        printf("LOST!!\n");
        return;
    }
}

void imgHandler(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat currentFrame = ptr->image;

    fullSystem->qImg.push(currentFrame);
    double timeImg = img_msg->header.stamp.toSec();
    fullSystem->qTimeImg.push(timeImg);
}

void projectPointCloud()
{
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize; 
    PointType thisPoint;

    cloudSize = laserCloudIn->points.size();

    for (size_t i = 0; i < cloudSize; ++i){

        thisPoint.x = laserCloudIn->points[i].x;
        thisPoint.y = laserCloudIn->points[i].y;
        thisPoint.z = laserCloudIn->points[i].z;

        laserCloudIn->points[i].intensity = -1;

        verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
        rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
        if (rowIdn < 0 || rowIdn >= N_SCAN)
            continue;

        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        if (columnIdn >= Horizon_SCAN)
            columnIdn -= Horizon_SCAN;

        if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            continue;

        range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        if (range < 0.1)
            continue;

        rangeMat.at<float>(rowIdn, columnIdn) = range;

        thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

        index = columnIdn  + rowIdn * Horizon_SCAN;
        fullCloud->points[index] = thisPoint;
        fullInfoCloud->points[index] = thisPoint;
        fullInfoCloud->points[index].intensity = range;
        laserCloudIn->points[i].intensity = range;
    }
}

void groundRemoval()
{
    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> candidateGround;

    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < Horizon_SCAN; ++j){
        for (size_t i = 0; i < groundScanInd; ++i){

            lowerInd = j + ( i )*Horizon_SCAN;
            upperInd = j + (i+1)*Horizon_SCAN;

            if (fullCloud->points[lowerInd].intensity == -1 ||
                fullCloud->points[upperInd].intensity == -1)
            {
                groundMat.at<int8_t>(i,j) = -1;
                continue;
            }
                
            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

            if (abs(angle - sensorMountAngle) <= 10)
            {
                groundMat.at<int8_t>(i,j) = 1;
                groundMat.at<int8_t>(i+1,j) = 1;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                labelMat.at<int>(i,j) = -1;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1)
                groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
        }
    }
}

void labelComponents(int row, int col)
{
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY; 
    bool lineCountFlag[N_SCAN] = {false};

    queueIndX[0] = row;
    queueIndY[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;
    
    while(queueSize > 0)
    {
        fromIndX = queueIndX[queueStartInd];
        fromIndY = queueIndY[queueStartInd];
        --queueSize;
        ++queueStartInd;

        labelMat.at<int>(fromIndX, fromIndY) = labelCount;

        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
        {
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;

            if (thisIndX < 0 || thisIndX >= N_SCAN)
                continue;

            if (thisIndY < 0)
                thisIndY = Horizon_SCAN - 1;
            if (thisIndY >= Horizon_SCAN)
                thisIndY = 0;

            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                continue;

            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                          rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                          rangeMat.at<float>(thisIndX, thisIndY));

            if ((*iter).first == 0)
                alpha = segmentAlphaX;
            else
                alpha = segmentAlphaY;

            angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

            if (angle > segmentTheta){

                queueIndX[queueEndInd] = thisIndX;
                queueIndY[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;

                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;

                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;
            }
        }
    }

    bool feasibleSegment = false;

    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum){
        int lineCount = 0;
        for (size_t i = 0; i < N_SCAN; ++i)
            if (lineCountFlag[i] == true)
                ++lineCount;
        if (lineCount >= segmentValidLineNum)
            feasibleSegment = true;            
    }

    if (feasibleSegment == true){
        ++labelCount;
    }else{
        for (size_t i = 0; i < allPushedIndSize; ++i){
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
}

void cloudSegmentation()
{
    for (size_t i = 0; i < N_SCAN; ++i)
        for (size_t j = 0; j < Horizon_SCAN; ++j)
            if (labelMat.at<int>(i,j) == 0)
                labelComponents(i, j);


    int sizeOfSegCloud = 0;

    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                if (labelMat.at<int>(i,j) == 999999){
                    if (i > groundScanInd && j % 5 == 0){
                        outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        continue;
                    }else{
                        continue;
                    }
                }
                
                if(groundMat.at<int8_t>(i,j) == 1)
                	fullCloud->points[j + i*Horizon_SCAN].intensity = -1.0;
                else
                	fullCloud->points[j + i*Horizon_SCAN].intensity = 1.0;

                segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);

                ++sizeOfSegCloud;
            }
        }
    }
}

void lidarCloudHandler(const sensor_msgs::PointCloud2ConstPtr& lidarCloudMsg){
    double timeLidarCloud = lidarCloudMsg->header.stamp.toSec();
    fullSystem->qTimeLidarCloud.push(timeLidarCloud);

    pcl::fromROSMsg(*lidarCloudMsg, *laserCloudIn);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

    projectPointCloud();

    groundRemoval();

    cloudSegmentation();

    int cloudSize = segmentedCloud->points.size();

    Eigen::Vector3d lidarCloudTemp;
    Eigen::Vector3d cloudPixelTemp;

    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel;

    int numGround = 0;
    int numAll = 0;

    for (size_t i = 0; i < cloudSize; ++i){
        lidarCloudTemp(0, 0) = segmentedCloud->points[i].x;
        lidarCloudTemp(1, 0) = segmentedCloud->points[i].y;
        lidarCloudTemp(2, 0) = segmentedCloud->points[i].z;

        Eigen::Vector3d temp = fullSystem->Rlc * lidarCloudTemp + fullSystem->tlc;

        if(temp(2, 0) < 0.2)
		{
			continue;
		}

	    float u = (float)(temp(0, 0) / temp(2, 0));
	    float v = (float)(temp(1, 0) / temp(2, 0));

	    float Ku = u * fullSystem->fx + fullSystem->cx;
        float Kv = v * fullSystem->fy + fullSystem->cy;

	    if((int)Ku<4 || (int)Ku>=wG[0]-5 || (int)Kv<4 || (int)Kv>hG[0]-4) 
    	{
    		continue;
    	}

    	if(Ku<fullSystem->left) fullSystem->left=(int)Ku;
    	if(Ku>fullSystem->right) fullSystem->right=(int)Ku;
    	if(Kv<fullSystem->up) fullSystem->up=(int)Kv;
    	if(Kv>fullSystem->down) fullSystem->down=(int)Kv;

    	cloudPixelTemp(0, 0) = (double)Ku;
    	cloudPixelTemp(1, 0) = (double)Kv;
    	cloudPixelTemp(2, 0) = temp(2, 0);

    	numAll++;

    	if(segmentedCloud->points[i].intensity < 0)
    		numGround++;

        vCloudPixel.push_back(cloudPixelTemp);
    }

    if(float(numGround)/(float)numAll > 0.8)
    	fullSystem->addFeaturePoint = true;
    else
    	fullSystem->addFeaturePoint = false;

    fullSystem->qCloudPixel.push(vCloudPixel);

    resetParameters();
}

void lidarPoseMapHandler(const nav_msgs::Odometry::ConstPtr& lidarPoseMapMsg){
    double timeLidarPoseMap = lidarPoseMapMsg->header.stamp.toSec();
    fullSystem->qTimeLidarPoseMap.push(timeLidarPoseMap);

    Eigen::Quaterniond q;
	q.w()=lidarPoseMapMsg->pose.pose.orientation.w;
	q.x()=lidarPoseMapMsg->pose.pose.orientation.x;
	q.y()=lidarPoseMapMsg->pose.pose.orientation.y;
	q.z()=lidarPoseMapMsg->pose.pose.orientation.z;

	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Vector3d t(lidarPoseMapMsg->pose.pose.position.x, lidarPoseMapMsg->pose.pose.position.y, lidarPoseMapMsg->pose.pose.position.z);

	fullSystem->qRotationMap.push(R);
	fullSystem->qPositionMap.push(t);
}

void lidarPoseOdometerHandler(const nav_msgs::Odometry::ConstPtr& lidarPoseOdometerMsg){
    double timeLidarPoseOdometer = lidarPoseOdometerMsg->header.stamp.toSec();
    fullSystem->qTimeLidarPoseOdometer.push(timeLidarPoseOdometer);

    Eigen::Quaterniond q;
	q.w()=lidarPoseOdometerMsg->pose.pose.orientation.w;
	q.x()=lidarPoseOdometerMsg->pose.pose.orientation.x;
	q.y()=lidarPoseOdometerMsg->pose.pose.orientation.y;
	q.z()=lidarPoseOdometerMsg->pose.pose.orientation.z;

	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Vector3d t(lidarPoseOdometerMsg->pose.pose.position.x, lidarPoseOdometerMsg->pose.pose.position.y, lidarPoseOdometerMsg->pose.pose.position.z);

	fullSystem->qRotationOdometer.push(R);
	fullSystem->qPositionOdometer.push(t);
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "sdv_loam");

	ROS_INFO("\033[1;32m---->\033[SDV-LOAM Started.");

	ros::NodeHandle n;

    parseArgument(n);

    allocateMemory();
    resetParameters();

    reader = new ImageFolderReader(calib, gammaCalib, vignette);
	reader->setGlobalCalibration();

    fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);
	fullSystem->loadSensorPrameters(pathSensorPrameter);

	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    std::thread runthread([&]() {
        gettimeofday(&tv_start, NULL);
        ros::Subscriber sub_img = n.subscribe(imgTopic, 100, imgHandler);
        ros::Subscriber subLidarCloud = n.subscribe<sensor_msgs::PointCloud2>(lidarTopic, 20, lidarCloudHandler);

        started = clock();

        ros::Rate rate(30);
        while(ros::ok())
        {
        	ros::spinOnce();
        	process();

            rate.sleep();
        }

        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->printResult(resultPath);

        int numFramesProcessed = currentId;
        double numSecondsProcessed = fabs(timestamp - firstFrameTime);
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));

        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*currentId) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)currentId << "\n";
            tmlog.flush();
            tmlog.close();
        }
    });

    if(viewer != 0)
        viewer->run();

    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader;

	printf("DSO OVER!\n");

	ros::spin();
	return 0;
}