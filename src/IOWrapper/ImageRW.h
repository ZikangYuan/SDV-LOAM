#pragma once

#include "util/NumType.h"
#include "util/MinimalImage.h"

#include <opencv2/highgui/highgui.hpp>

namespace sdv_loam
{
namespace IOWrap
{

MinimalImageB* readImageBW_8U(std::string filename);
MinimalImageB3* readImageRGB_8U(std::string filename);
MinimalImage<unsigned short>* readImageBW_16U(std::string filename);

MinimalImageB* readRosImageBW_8U(cv::Mat &image);

MinimalImageB* readStreamBW_8U(char* data, int numBytes);

void writeImage(std::string filename, MinimalImageB* img);
void writeImage(std::string filename, MinimalImageB3* img);
void writeImage(std::string filename, MinimalImageF* img);
void writeImage(std::string filename, MinimalImageF3* img);

}
}
