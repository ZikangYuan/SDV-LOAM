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
git clone https://github.com/ZikangYuan/SDV-LOAM.git sdv_loam
cd ..
catkin_make
```

## Run on Public Datasets

Noted:

A. Both of the path of output pose, the path of camera parameters, the path of transformation parameters from LiDAR to camera, the ROS topic of images and LiDAR point clouds are set as the input parameters, users can change them on launch file.

B. The message type of input LiDAR point clouds must be *sensor_msgs::PointCloud2*.

C. There is some randomness in this code, and the result is not stable on part of sequences (e.g., *KITTI-01*). However, users can reproduce the results recorded in our paper by running it several times.

###  1. Run on [*KITTI-Odometry*](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Both the frequency of images and LiDAR point clouds of *KITTI-Odometry* are 10 Hz, while they are strictly one-to-one. In addition, the motion distortion of LiDAR pont cluods have been calibrated in advance, therefore, users do not need to consider the effect of motion distortion when evaluation on *KITTI-Odometry*. Users can directly utilize the [kitti2bag](https://github.com/ZikangYuan/kitti2bag) tool to convert data of *KITTI odometry* to ROS bag format.

```bash
python3 nclt_to_rosbag.py PATH_OF_NVLT_SEQUENCE_FOLDER PATH_OF_OUTPUT_BAG
```

Then, please go to the workspace of SR-LIO and type:

```bash
cd SR-LIO
sourcr devel/setup.bash
roslaunch sr_lio lio_nclt.launch
```

Then open the terminal in the path of the bag file, and type:

```bash
rosbag play SEQUENCE_NAME.bag --clock -d 1.0 -r 0.2 
```

### 2. Run on [*UTBM*](https://epan-utbm.github.io/utbm_robocar_dataset/#Downloads)

Before evaluating on *UTBM* dataset, a dependency needs to be installed. If your OS are Ubuntu 16.04, please type:

```bash
sudo apt-get install ros-kinetic-velodyne 
```

If your OS are Ubuntu 18.04, please type:

```bash
sudo apt-get install ros-melodic-velodyne 
```

Then open the terminal in the path of SR-LIO, and type:

```bash
sourcr devel/setup.bash
roslaunch sr_lio lio_utbm.launch
```

Then open the terminal in the path of the bag file, and type:

```bash
rosbag play SEQUENCE_NAME.bag --clock -d 1.0 -r 0.2 
```

### 3. Run on [*ULHK*](https://github.com/weisongwen/UrbanLoco)

For sequence *HK-Data-2019-01-17* and *HK-Data-2019-03-17*, the imu data does not include the gravity acceleration component, and the topic of LiDAR point cloud data is */velodyne_points_0*. For other sequences of *ULHK* used by us, the imu data includes the gravity acceleration component, and the topic of LiDAR point cloud data is */velodyne_points*. Therefore, we provide two launch files for the *ULHK* dataset.

If you test SR-LIO on *HK-Data-2019-01-17* or *HK-Data-2019-03-17*, please type:

```bash
sourcr devel/setup.bash
roslaunch sr_lio lio_ulhk1.launch
```
