## An Invitation to 3D Vision: A Tutorial for Everyone
_An Invitation to 3D Vision_ is a introductory tutorial on 3D vision (a.k.a. visual geometry or geometric vision).
It aims to make beginners understand basic theory of 3D vision and implement their own applications using [OpenCV][].
In addition to tutorial slides, we provide a set of example codes. The example codes are written as short as possible (mostly __less than 100 lines__) to improve readability and involve many interesting and practical applications.

* Download [tutorial slides](https://github.com/sunglok/3dv_tutorial/releases/download/misc/3dv_slides.pdf)
* Download [example codes in a ZIP file](https://github.com/sunglok/3dv_tutorial/archive/master.zip)
* Download [OpenCV binary](https://github.com/sunglok/3dv_tutorial/releases/download/misc/OpenCV_v3.2.0_32bit.zip) and [CLAPACK binary](https://github.com/sunglok/3dv_tutorial/releases/download/misc/CLAPACK_v3.2.1_32bit.zip) for Windows

### What does its name come from?
* The main title, _An Invitation to 3D Vision_, came from [a legendary book by Yi Ma, Stefano Soatto, Jana Kosecka, and Shankar S. Sastry](http://vision.ucla.edu/MASKS/). We wish that our tutorial will be the first gentle invitation card for beginners in 3D vision and explorers from other fields.
* The subtitle, _for everyone_, was inspired from [Prof. Kim's online lecture](https://hunkim.github.io/ml/) (in Korean). Our tutorial is also intended not only for students and researchers in academia, but also for hobbyists and developers in industries. We tried to describe important and typical problems and their solutions in [OpenCV][]. We hope readers understand it easily without serious mathematical background.

### Examples
* __Single-view Geometry__
  * Camera Projection Model
    * Object Localization and Measurement: [object_localization.cpp][]
    * Image Formation: [image_formation.cpp][] (result: [0](https://drive.google.com/file/d/0B_iOV9kV0whLY2luc05jZGlkZ2s/view?usp=sharing), [1](https://drive.google.com/file/d/0B_iOV9kV0whLS3M4S09ZZHpjTkU/view?usp=sharing), [2](https://drive.google.com/file/d/0B_iOV9kV0whLV2dLZHd0MmVkd28/view?usp=sharing), [3](https://drive.google.com/file/d/0B_iOV9kV0whLS1ZBR25WekpMYjA/view?usp=sharing), [4](https://drive.google.com/file/d/0B_iOV9kV0whLYVB0dm9Fc0dvRzQ/view?usp=sharing))
    * Geometric Distortion Correction: [distortion_correction.cpp][] ([result](https://www.youtube.com/watch?v=HKetupWh4V8))
  * General 2D-3D Geometry
    * Camera Calibration: [camera_calibration.cpp][] ([result](https://drive.google.com/file/d/0B_iOV9kV0whLZ0pDbWdXNWRrZ00/view?usp=sharing))
    * Camera Pose Estimation (Chessboard): [pose_estimation_chessboard.cpp][] ([result](https://www.youtube.com/watch?v=4nA1OQGL-ig))
    * Camera Pose Estimation (Book): [pose_estimation_book1.cpp][]
    * Camera Pose Estimation and Calibration: [pose_estimation_book2.cpp][]
    * Camera Pose Estimation and Calibration w/o Initially Given Camera Parameters: [pose_estimation_book3.cpp][] ([result](https://www.youtube.com/watch?v=GYp4h0yyB3Y))
* __Two-view Geometry__
  * Planar 2D-2D Geometry (Projective Geometry)
    * Perspective Distortion Correction: [perspective_correction.cpp][] (result: [original](https://drive.google.com/file/d/0B_iOV9kV0whLVlFpeFBzYWVadlk/view?usp=sharing), [rectified](https://drive.google.com/file/d/0B_iOV9kV0whLMi1UTjN5QXhnWFk/view?usp=sharing))
    * Planar Image Stitching: [image_stitching.cpp][] ([result](https://drive.google.com/file/d/0B_iOV9kV0whLOEQzVmhGUGVEaW8/view?usp=sharing))
    * 2D Video Stabilization: [video_stabilization.cpp][] ([result](https://www.youtube.com/watch?v=be_dzYicEzI))
  * General 2D-2D Geometry (Epipolar Geometry)
    * Monocular Visual Odometry (Epipolar Version): [visual_odometry_epipolar.cpp][]
    * Triangulation (Two-view Reconstruction): [triangulation.cpp][]
* __Multi-view Geometry__
  * Bundle Adjustment
    * Global Version: [bundle_adjustment_global.cpp][]
    * Incremental Version: [bundle_adjustment_inc.cpp][]
  * Structure-from-Motion
    * Global SfM: [sfm_global.cpp][]
    * Incremental SfM: [sfm_inc.cpp][]
  * Visual Odometry
    * Epipolar Version: [visual_odometry_epipolar.cpp][]
    * PnP Version
    * Bundle Adjustment Version
  * Visual SLAM
  * c.f. The above examples need [cvsba][] for bundle adjustment.
* __Correspondence Problem__
  * Line Fitting with RANSAC: [line_fitting_ransac.cpp][]
  * Line Fitting with M-estimators: [line_fitting_m_est.cpp][]
* **Appendix**
  * Planar Homograph Estimation
  * Fundamental Matrix Estimation

### Dependencies
* [OpenCV][] (> 3.0.0, 3-clause BSD License)
  * _OpenCV_ is a base of all example codes for linear algebra, vision algorithms, image/video manipulation, and GUI.
* [cvsba][] (GPL): An OpenCV wrapper for sba library
  * _cvsba_ is used by bundle adjustment. It is optional for bundle adjustment.
  * It is included in EXTERNAL directory in the sake of your convenience.
* [CLAPACK][] (Public Domain): f2c'ed version of LAPACK
  * _CLAPACK_ is used by cvsba. It is optional for bundle adjustment.

### License
* [Beerware](http://en.wikipedia.org/wiki/Beerware)

### Authors
* [Sunglok Choi](http://sites.google.com/site/sunglok/) (sunglok AT hanmail DOT net)

### Acknowledgement
The authors thank the following contributors and projects.

* [Jae-Yeong Lee](https://sites.google.com/site/roricljy/): We sincerely thank him for motivating many examples and providing [OpenCV][] binaries for Windows.
* [Giseop Kim](https://sites.google.com/view/giseopkim): He contributed the initial version of SfM codes with [cvsba][] and [Toy-SfM](https://github.com/royshil/SfM-Toy-Library).
* [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/): We include some of KITTI odometry dataset for demonstrating visual odometry.
* [Russell Hewett](https://courses.engr.illinois.edu/cs498dh3/fa2013/projects/stitching/ComputationalPhotograph_ProjectStitching.html): We include his two hill images for demonstrating image stitching.
* [Kang Li](http://www.cs.cmu.edu/~kangli/code/Image_Stabilizer.html): We include his shaking CCTV video for demonstrating video stabilization.
* [Richard Blais](http://www.richardblais.net/): We include his book cover used in [the OpenCV tutorial](http://docs.opencv.org/3.1.0/dc/d16/tutorial_akaze_tracking.html).

[OpenCV]: http://opencv.org/
[cvsba]: https://www.uco.es/investiga/grupos/ava/node/39
[CLAPACK]: http://www.netlib.org/clapack/

[object_localization.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/object_localization.cpp
[image_formation.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/image_formation.cpp
[distortion_correction.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/distortion_correction.cpp
[camera_calibration.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/camera_calibration.cpp
[pose_estimation_chessboard.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/pose_estimation_chessboard.cpp
[pose_estimation_book1.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/pose_estimation_book1.cpp
[pose_estimation_book2.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/pose_estimation_book2.cpp
[pose_estimation_book3.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/pose_estimation_book3.cpp
[perspective_correction.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/perspective_correction.cpp
[image_stitching.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/image_stitching.cpp
[video_stabilization.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/video_stabilization.cpp
[visual_odometry_epipolar.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/visual_odometry_epipolar.cpp
[triangulation.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/triangulation.cpp
[bundle_adjustment_global.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/bundle_adjustment_global.cpp
[bundle_adjustment_inc.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/bundle_adjustment_inc.cpp
[sfm_global.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/sfm_global.cpp
[sfm_inc.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/sfm_inc.cpp
[line_fitting_ransac.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/line_fitting_ransac.cpp
[line_fitting_m_est.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/line_fitting_m_est.cpp