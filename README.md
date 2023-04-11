# SDV-LOAM

**SDV-LOAM** (LiDAR-Inertial Odometry with Sweep Reconstruction) is a cascaded vision-LiDAR odometry and mapping system, which consists of a LiDAR-assisted depth-enhanced visual odometry and a LiDAR odometry. At this stage, the released code is just the **vision module** of **SDV-LOAM**, while the LiDAR module would also be released soon.

The implementation of our vision module is based on [DSO](https://github.com/JakobEngel/dso), while we change it from monocular direct method to LiDAR-assisted semi-direct method with ROS interface. All the contributions of our vision module proposed in [SDV-LOAM](https://ieeexplore.ieee.org/abstract/document/10086694) can be found in this code.

## Related Work

[SDV-LOAM: Semi-Direct Visual-LiDAR Odometry and Mapping](https://ieeexplore.ieee.org/abstract/document/10086694)

Authors: *Zikang Yuan*, *Qingjie Wang*, *Ken Cheng*, *Tianyu Hao* and [*Xin Yang*](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)

## Installation

### 1. Requirements

> GCC >= 5.4.0
>
> Cmake >= 3.0.2
> 
> [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2.8
>
> [PCL](https://pointclouds.org/downloads/) == 1.7 for Ubuntu 16.04, and == 1.8 for Ubuntu 18.04
>
> [ROS](http://wiki.ros.org/ROS/Installation)
>
> [Pangolin](https://github.com/stevenlovegrove/Pangolin/tree/v0.5) == 0.5 for Ubuntu 16.04
>
> [OpenCV](https://opencv.org/releases/) == 2.4.9 for Ubuntu 16.04

##### Have Tested On:

| OS    | GCC  | Cmake | Eigen3 | PCL | Pangolin | OpenCV |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ubuntu 16.04 | 5.4.0  | 3.16.0 | 3.2.8 | 1.7 | 0.5 | 2.4.9 |

### 2. Create ROS workspace

```bash
mkdir -p ~/SDV-LOAM/src
cd SDV-LOAM/src
```

### 3. Clone the directory and build

```bash
git clone https://github.com/ZikangYuan/sr_lio.git
cd ..
catkin_make
```
