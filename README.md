# SDV-LOAM

**SDV-LOAM** (LiDAR-Inertial Odometry with Sweep Reconstruction) is a cascaded vision-LiDAR odometry and mapping system, which consists of a LiDAR-assisted depth-enhanced visual odometry and a LiDAR odometry. At this stage, the released code is just the **vision module** of **SDV-LOAM**, while the LiDAR module would also be released soon.

The implementation of our vision module is based on [DSO](https://github.com/JakobEngel/dso), while we change it from monocular direct method to LiDAR-assisted semi-direct method with ROS interface. All the contributions of our vision module proposed in **SDV-LOAM** can be found in this code.

## Related Work

[SDV-LOAM: Semi-Direct Visual-LiDAR Odometry and Mapping](https://ieeexplore.ieee.org/abstract/document/10086694)

Authors: [*Zikang Yuan*](https://scholar.google.com/citations?hl=zh-CN&user=acxdM9gAAAAJ), *Qingjie Wang*, *Ken Cheng*, *Tianyu Hao* and [*Xin Yang*](https://scholar.google.com/citations?user=lsz8OOYAAAAJ&hl=zh-CN)

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

###  1. Generating ROS bag from [*KITTI-Odometry*](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) data

Both the frequency of images and LiDAR point clouds of *KITTI-Odometry* are 10 Hz, while they are strictly one-to-one. In addition, the motion distortion of LiDAR pont cluods have been calibrated in advance, therefore, users do not need to consider the effect of motion distortion when evaluation on *KITTI-Odometry*. Users can directly utilize the [*KITTI-Odometry to ROS bag*](https://github.com/ZikangYuan/kitti2bag) tool to convert data of *KITTI odometry* to ROS bag format.

###  2. Generating ROS bag from [*KITTI-360*](https://www.cvlibs.net/datasets/kitti-360/) data

Both the frequency of images and LiDAR point clouds of *KITTI-360* are 10 Hz, while they are strictly one-to-one. The motion distortion of LiDAR pont cluods have not been calibrated in advance, therefore, the motion calibration need to be processed in theory. However, we found that when the influence of motion distortion was taken into consideration in our visual module, the final pose estimation result would be worse. Therefore, we did not reserve the motion distortion module in this code. Users can also directly utilize the [*KITTI-360 to ROS bag*](https://github.com/ZikangYuan/kitti360_2bag) tool to convert data of *KITTI-360* to ROS bag format.

###  3. Generating ROS bag from [*KITTI-CARLA*](https://npm3d.fr/kitti-carla) data

Both the frequency of images and LiDAR point clouds of *KITTI-CARLA* are 10 Hz, while they are strictly one-to-one. The motion distortion of LiDAR pont cluods have not been calibrated in advance, but users can perform motion calibration using the [*KITTI-CARLA calibration*](https://github.com/ZikangYuan/kitti_carla_calib) tool. After motion calibration, users can also directly utilize the [*KITTI-CARLA to ROS bag*](https://github.com/ZikangYuan/kitti_carla2bag) tool to convert data of *KITTI-CARLA* to ROS bag format.

###  4. Run

After generating the ROS bag file, please go to the workspace of SDV-LOAM and type:

```bash
cd SDV-LOAM
source devel/setup.bash
roslaunch sdv_loam run.launch
```

Then open the terminal in the path of the bag file, and type:

```bash
rosbag play SEQUENCE_NAME.bag --clock -d 1.0
```

## Citation

If you use our work in your research project, please consider citing:

```
@article{10086694,
  author={Yuan, Zikang and Wang, Qingjie and Cheng, Ken and Hao, Tianyu and Yang, Xin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={SDV-LOAM: Semi-Direct Visual-LiDAR Odometry and Mapping}, 
  year={2023},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2023.3262817}}
```

## Acknowledgments

Thanks for [DSO](https://github.com/JakobEngel/dso).
