#include "IOWrapper/ImageRW.h"

namespace sdv_loam
{


namespace IOWrap
{

MinimalImageB* readImageBW_8U(std::string filename) {printf("not implemented. bye!\n"); return 0;};
MinimalImageB3* readImageRGB_8U(std::string filename) {printf("not implemented. bye!\n"); return 0;};
MinimalImage<unsigned short>* readImageBW_16U(std::string filename) {printf("not implemented. bye!\n"); return 0;};
MinimalImageB* readStreamBW_8U(char* data, int numBytes) {printf("not implemented. bye!\n"); return 0;};
void writeImage(std::string filename, MinimalImageB* img) {};
void writeImage(std::string filename, MinimalImageB3* img) {};
void writeImage(std::string filename, MinimalImageF* img) {};
void writeImage(std::string filename, MinimalImageF3* img) {};

}

}
