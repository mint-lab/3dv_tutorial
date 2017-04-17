## An Invitation to 3D Vision: A Tutorial for Everyone

_An Invitation to 3D Vision_ is a introductory tutorial on 3D vision (a.k.a. visual geometry or geometric vision). It aims to make beginners understand basic theory of 3D vision and implement their own applications using [OpenCV][]. In addition to tutorial slides, we provide a set of example codes. The example codes are written as short as possible (__less than 100 lines__) to improve readability and involve many interesting and practical applications.

* [Download tutorial slides](https://github.com/sunglok/3dv_tutorial/releases/download/misc/3dv_slides.pdf)
* [Download example codes in a ZIP file](https://github.com/sunglok/3dv_tutorial/archive/master.zip)
* [Download binaries and headers for OpenCV and CLAPACK for Windows](https://github.com/sunglok/3dv_tutorial/releases/download/misc/EXTERNAL_for_Windows_32bit.zip)

### What does its name come from?
* The main title, _An Invitation to 3D Vision_, came from [a legendary book by Yi Ma, Stefano Soatto, Jana Kosecka, and Shankar S. Sastry](http://vision.ucla.edu/MASKS/). We wish that our tutorial will be the first gentle invitation card for beginners in 3D vision and explorers from other fields.
* The subtitle, _for everyone_, was inspired from [Prof. Kim's online lecture](https://hunkim.github.io/ml/) (in Korean). Our tutorial is also intended not only for students and researchers in academia, but also for hobbyists and developers in industries. We tried to describe important and typical problems and their solutions in [OpenCV][]. We hope readers understand it easily without serious mathematical background.

### Example descriptions
* __Single-view Geometry__
  * Camera Projection Model
    * Simple Camera Calibration and Object Localization: [simple_object_proposal.cpp][]
    * Image Formation: [image_formation.cpp][] (screenshots: [0](https://drive.google.com/file/d/0B_iOV9kV0whLY2luc05jZGlkZ2s/view?usp=sharing), [1](https://drive.google.com/file/d/0B_iOV9kV0whLS3M4S09ZZHpjTkU/view?usp=sharing), [2](https://drive.google.com/file/d/0B_iOV9kV0whLV2dLZHd0MmVkd28/view?usp=sharing), [3](https://drive.google.com/file/d/0B_iOV9kV0whLS1ZBR25WekpMYjA/view?usp=sharing), [4](https://drive.google.com/file/d/0B_iOV9kV0whLYVB0dm9Fc0dvRzQ/view?usp=sharing))
    * Geometric Distortion Correction: [distortion_correction.cpp][]
  * General 2D-3D Geometry
    * Camera Calibration: [camera_calibration.cpp][] ([result](https://drive.google.com/file/d/0B_iOV9kV0whLZ0pDbWdXNWRrZ00/view?usp=sharing))
    * Camera Pose Estimation (Chessboard Version): [pose_estimation_chessboard.cpp][]
* __Two-view Geometry__
  * Planar 2D-2D Geometry (Projective Geometry)
    * Perspective Distortion Correction: [perspective_correction.cpp][] (screenshot: [original](https://drive.google.com/file/d/0B_iOV9kV0whLVlFpeFBzYWVadlk/view?usp=sharing), [rectified](https://drive.google.com/file/d/0B_iOV9kV0whLMi1UTjN5QXhnWFk/view?usp=sharing))
    * Planar Image Stitching: [image_stitching.cpp][] ([screenshot](https://drive.google.com/file/d/0B_iOV9kV0whLOEQzVmhGUGVEaW8/view?usp=sharing))
    * 2D Video Stabilization: [video_stabilization.cpp][]
  * General 2D-2D Geometry (Epipolar Geometry)
    * Monocular Visual Odometry (Epipolar Version): [visual_odometry_epipolar.cpp][]
    * Triangulation (Two-view Reconstruction): [triangulation.cpp][]
* __Multi-view Geometry__
  * Bundle Adjustment using cvsba (Multiple-view Reconstruction): [bundle_adjustment.cpp][]
  * Sparse and Dense 3D Reconstruction using VisualSFM
* __Correspondence Problem__
  * Line Fitting with RANSAC: [ransac_line.cpp][]

### Dependencies
* [OpenCV][] (> 3.0.0, 3-clause BSD License)
  * _OpenCV_ is a base of all example codes for linear algebra, vision algorithms, image/video manipulation, and GUI.
* [cvsba][] (GPL): An OpenCV wrapper for sba library
  * _cvsba_ is used by bundle adjustment. It is optional for bundle adjustment but included in EXTERNAL directory in the sake of your convenience.
* [CLAPACK][] (Public Domain): f2c'ed version of LAPACK
  * _CLAPACK_ is used by cvsba. It is optional for bundle adjustment.

### License
* [Beerware](http://en.wikipedia.org/wiki/Beerware)

### Authors
* [Sunglok Choi](http://sites.google.com/site/sunglok/) (sunglok AT hanmail DOT net)

### Acknowledgement
The authors thank the following contributors and projects.

* [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/): We include some of KITTI odometry dataset for demonstrating visual odometry.
* [Russell Hewett](https://courses.engr.illinois.edu/cs498dh3/fa2013/projects/stitching/ComputationalPhotograph_ProjectStitching.html): We include his two hill images for demonstrating image stitching.
* [Kang Li](http://www.cs.cmu.edu/~kangli/code/Image_Stabilizer.html): We include his shaking CCTV video for demonstrating video stabilization.
* [Dr. Jae-Yeong Lee](https://sites.google.com/site/roricljy/): We sincerely thank him for motivating many examples and providing [OpenCV][] binaries for Windows.
* Jaeho Lim: We thank him for his careful review and comment on the tutorial slides.

[OpenCV]: http://opencv.org/
[cvsba]: https://www.uco.es/investiga/grupos/ava/node/39
[CLAPACK]: http://www.netlib.org/clapack/

[image_formation.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/image_formation.cpp
[simple_object_proposal.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/simple_object_proposal.cpp
[distortion_correction.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/distortion_correction.cpp
[camera_calibration.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/camera_calibration.cpp
[pose_estimation_chessboard.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/pose_estimation_chessboard.cpp
[perspective_correction.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/perspective_correction.cpp
[image_stitching.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/image_stitching.cpp
[video_stabilization.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/video_stabilization.cpp
[visual_odometry_epipolar.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/visual_odometry_epipolar.cpp
[triangulation.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/triangulation.cpp
[bundle_adjustment.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/bundle_adjustment.cpp
[ransac_line.cpp]: https://github.com/sunglok/3dv_tutorial/blob/master/src/ransac_line.cpp
