#include "IOWrapper/ImageDisplay.h"

namespace sdv_loam
{


namespace IOWrap
{
void displayImage(const char* windowName, const MinimalImageB* img, bool autoSize) {};
void displayImage(const char* windowName, const MinimalImageB3* img, bool autoSize) {};
void displayImage(const char* windowName, const MinimalImageF* img, bool autoSize) {};
void displayImage(const char* windowName, const MinimalImageF3* img, bool autoSize) {};
void displayImage(const char* windowName, const MinimalImageB16* img, bool autoSize) {};


void displayImageStitch(const char* windowName, const std::vector<MinimalImageB*> images, int cc, int rc) {};
void displayImageStitch(const char* windowName, const std::vector<MinimalImageB3*> images, int cc, int rc) {};
void displayImageStitch(const char* windowName, const std::vector<MinimalImageF*> images, int cc, int rc) {};
void displayImageStitch(const char* windowName, const std::vector<MinimalImageF3*> images, int cc, int rc) {};

int waitKey(int milliseconds) {return 0;};
void closeAllWindows() {};
}

}
