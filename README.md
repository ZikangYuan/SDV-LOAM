# SDV-LOAM

This is a project of visual-LiDAR odometry and mapping system, which is formed by cascading a semi-direct LiDAR-assisted visual odometry and a LiDAR odometry. We analyzed several problems of existing systems and designed our own system to address these problem.

In addition, different from existing systems which integrate high-frequency output (i.e., 30~60Hz) from vision module to low-frequency output (i.e., 10Hz) from LiDAR module, we propose a novelty method to make every output pose from vision module can be refined by LiDAR module.

Our system achieves 0.60% translational error on [KITTI odometry benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

Now the article corresponding to this project is under reviewe, and we would release the code as soon as the publicity of the paper.
