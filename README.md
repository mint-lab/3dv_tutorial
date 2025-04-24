## An Invitation to 3D Vision: A Tutorial for Everyone
_An Invitation to 3D Vision_ is an introductory tutorial on _3D computer vision_ (a.k.a. _geometric vision_ or _visual geometry_ or _multiple-view geometry_). It aims to make beginners understand basic theories on 3D vision and implement its applications using [OpenCV](https://opencv.org/).
In addition to tutorial slides, example codes are provided in the purpose of education. They include simple but interesting and practical applications. The example codes are written as short as possible (mostly __less than 100 lines__) to be clear and easy to understand.

* To clone this repository (codes and slides): `git clone https://github.com/mint-lab/3dv_tutorial.git`
* To fork this repository to your Github: [Click here](https://github.com/mint-lab/3dv_tutorial/fork)
* To download codes and slides as a ZIP file: [Click here](https://github.com/mint-lab/3dv_tutorial/archive/master.zip)
* :memo: [How to run example codes in Python](https://github.com/mint-lab/3dv_tutorial/blob/master/HOWTO_RUN_PYTHON.md)
* :memo: [How to run example codes in C++](https://github.com/mint-lab/3dv_tutorial/blob/master/HOWTO_RUN_CPP.md)



### What does its name come from?
* The main title, _An Invitation to 3D Vision_, came from [a legendary book by Yi Ma, Stefano Soatto, Jana Kosecka, and Shankar S. Sastry](http://vision.ucla.edu/MASKS/). We wish that our tutorial will be the first gentle invitation card for beginners to 3D vision and its applications.
* The subtitle, _for everyone_, was inspired from [Prof. Kim's online lecture](https://hunkim.github.io/ml/) (in Korean). Our tutorial is also intended not only for students and researchers in academia, but also for hobbyists and developers in industries. We tried to describe important and typical problems and their solutions in [OpenCV](https://opencv.org/). We hope readers understand it easily without serious mathematical background.



### Lecture Slides
* [Section 1. Introduction](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/01_introduction.pdf)
* [Section 2. Single-view Geometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/02_single-view_geometry.pdf)
* [Section 3. Two-view Geometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/03_two-view_geometry.pdf)
* [Section 4. Solving Problems](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/04_solving_problems.pdf)
* [Section 5. Finding Correspondence](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/05_correspondence.pdf)
* [Section 6. Multiple-view Geometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/06_multi-view_geometry.pdf)
* Special Topic) [Bayesian Filtering](https://github.com/mint-lab/filtering_tutorial)
* [Section 7. Visual SLAM and Odometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/07_visual_slam.pdf)



### Example Codes
* **Section 1. Introduction** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/01_introduction.pdf)
* **Section 2. Single-view Geometry** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/02_single-view_geometry.pdf)
  * Getting Started with 2D
    * 3D rotation conversion [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/3d_rotation_conversion.py)
  * Pinhole Camera Model
    * Object localization [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/object_localization.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/object_localization.cpp)
    * Image formation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/image_formation.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/image_formation.cpp)
  * Geometric Distortion Models
    * Geometric distortion visualization [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/distortion_visualization.py)
    * Geometric distortion correction [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/distortion_correction.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/distortion_correction.cpp) [[result video]](https://youtu.be/HKetupWh4V8)
  * Camera Calibration
    * Camera calibration [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/camera_calibration.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/camera_calibration.cpp)
  * Absolute Camera Pose Estimation (a.k.a. perspective-n-point; PnP)
    * Pose estimation (chessboard) [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_chessboard.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_chessboard.cpp) [[result video]](https://youtu.be/4nA1OQGL-ig)
    * Pose estimation (book) [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_book1.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_book1.cpp)
    * Pose estimation (book) with camera calibration [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_book2.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_book2.cpp)
    * Pose estimation (book) with camera calibration without initial $K$ [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_book3.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_book3.cpp) [[result video]](https://youtu.be/GYp4h0yyB3Y)
* **Section 3. Two-view Geometry** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/03_two-view_geometry.pdf)
  * Planar Homography
    * Perspective distortion correction [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/perspective_correction.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/perspective_correction.cpp)
    * Planar image stitching [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/image_stitching.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/image_stitching.cpp)
    * 2D video stabilization [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/video_stabilization.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/video_stabilization.cpp) [[result video]](https://youtu.be/be_dzYicEzI)
  * Epipolar Geometry
    * Epipolar line visualization [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/epipolar_line_visualization.py)
  * Relative Camera Pose Estimation
    * Fundamental matrix estimation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/fundamental_mat_estimation.py)
    * Monocular visual odometry (epipolar version) [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/vo_epipolar.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/vo_epipolar.cpp) [[result video]](https://youtu.be/Pc_IYrSH3sI)
  * Triangulation
    * Triangulation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/triangulation.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/triangulation.cpp)
* **Section 4. Solving Problems** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/04_solving_problems.pdf)
  * Solving Linear Equations in 3D Vision
    * Affine transformation estimation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/affine_estimation_implement.py)
    * Planar homography estimation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/homography_estimation_implement.py)
      * Appendix) Image warping using homography [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/image_warping_implement.py)
    * Fundamental matrix estimation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/fundamental_mat_estimation_implement.py)
    * Triangulation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/triangulation_implement.py)
  * Solving Nonlinear Equations in 3D Vision
    * Absolute camera pose estimation [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/pose_estimation_implement.py)
    * Camera calibration [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/camera_calibration_implement.py)
* **Section 5. Finding Correspondence** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/05_correspondence.pdf)
  * Feature Points and Descriptors
    * Harris corner [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/harris_corner_implement.py)
    * SuperPoint [[Github]](https://github.com/magicleap/SuperPointPretrainedNetwork)
  * Feature Matching and Tracking
    * Feature matching comparison [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/feature_matching.py)
    * SuperGlue [[Github]](https://github.com/magicleap/SuperGluePretrainedNetwork)
    * Feature tracking with KLT tracker [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/feature_tracking_klt.py)
  * Outlier Rejection
    * Line fitting with RANSAC [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/line_fitting_ransac.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/line_fitting_ransac.cpp)
    * Line fitting with M-estimator [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/line_fitting_m_estimator.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/line_fitting_m_estimator.cpp)
    * Planar homography estimation with RANSAC [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/image_stitching_implement.py)
* **Section 6. Multiple-view Geometry** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/06_multi-view_geometry.pdf)
  * Bundle Adjustment [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/bundle_adjustment.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/bundle_adjustment.cpp)
  * Structure-from-Motion (SfM)
    * Global SfM [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/sfm_global.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/sfm_global.cpp)
    * Incremental SfM [[python]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/sfm_inc.py) [[cpp]](https://github.com/mint-lab/3dv_tutorial/blob/master/examples/sfm_inc.cpp)
* **Section 7. Visual SLAM and Odometry** [[slides]](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/07_visual_slam.pdf)



### License
* [Beerware](http://en.wikipedia.org/wiki/Beerware)



### Authors
* [Sunglok Choi](https://mint-lab.github.io/sunglok/)
* [JunHyeok Choi](https://github.com/cjh1995-ros)



### Acknowledgement
The authors thank the following contributors and projects.

* [Jae-Yeong Lee](https://sites.google.com/site/roricljy/): He motivated many examples.
* [Giseop Kim](https://sites.google.com/view/giseopkim): He contributed the initial version of SfM codes based on [Toy-SfM](https://github.com/royshil/SfM-Toy-Library) and [cvsba](https://www.uco.es/investiga/grupos/ava/node/39).
* [Richard Blais](http://www.richardblais.net/): His book cover and video in [the OpenCV tutorial](http://docs.opencv.org/3.1.0/dc/d16/tutorial_akaze_tracking.html) were used to demonstrate camera pose estimation and augmented reality.
* [Russell Hewett](https://courses.engr.illinois.edu/cs498dh3/fa2013/projects/stitching/ComputationalPhotograph_ProjectStitching.html): His two hill images were used to demonstrate image stitching.
* [Kang Li](http://www.cs.cmu.edu/~kangli/code/Image_Stabilizer.html): His shaking CCTV video was used to demonstrate video stabilization.
* [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/): The KITTI odometry dataset #07 was used to demonstrate visual odometry and SLAM.