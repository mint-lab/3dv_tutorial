## An Invitation to 3D Vision: A Tutorial for Everyone

_An Invitation to 3D Vision_ is a introductory tutorial on 3D vision (a.k.a. visual geometry or geometric vision). It aims to make beginners understand basic theory of 3D vision and implement their own applications using [OpenCV][]. In addition to tutorial slides, we provide a set of example codes. The example codes are written as short as possible (__less than 100 lines__) to improve readability and involve many interesting and practical applications.

* Download tutorial slides [Link][https://github.com/sunglok/3dv_tutorial/releases/download/misc/3dv_slides.pdf]
* Download example codes in a ZIP file [Link][https://github.com/sunglok/3dv_tutorial/archive/master.zip]
* Download binaries and headers for [OpenCV][] and [CLAPACK][] for Windows [Link][https://github.com/sunglok/3dv_tutorial/releases/download/misc/EXTERNAL4Windows.32bit.zip]

### What does its name come from?
* The tutorial title, _An Invitation to 3D Vision_, came from [a legendary book by Yi Ma, Stefano Soatto, Jana Kosecka, and Shankar S. Sastry][http://vision.ucla.edu/MASKS/]. We wish that this tutorial will be the first gentle invitation card for beginners in 3D vision and explorers from other fields.
* The subtitle, _for everyone_, came from [Prof. Kim's online lecture][https://hunkim.github.io/ml/] (in Korean). This tutorial is also intended not only for students and researchers in academia, but also for hobbyists and developers in industries. We tried to describe important and typical problems and their solutions in [OpenCV][]. We hope readers understand it without serious mathematical background.

### Example descriptions
 * __Single-view Geometry__
  * Camera Projection Model
    * Simple Camera Calibration and Object Localization
    * Image Formation: [image_formation.cpp][]
    * Geometric Distortion Correction: [distortion_correction.cpp][]
   * General 2D-3D Geometry
    * Camera Calibration: [camera_calibration.cpp][]
    * Camera Pose Estimation (Chessboard Version): [pose_estimation_chessboard.cpp][]
 * __Two-view Geometry__
  * Planar 2D-2D Geometry (Projective Geometry)
    * Perspective Distortion Correction: [perspective_correction.cpp][]
    * Planar Image Stitching: [image_stitching.cpp][]
    * 2D Video Stabilization: [video_stabilization.cpp][]
  * General 2D-2D Geometry (Epipolar Geometry)
    * Monocular Visual Odometry (Epipolar Version): [visual_odometry_epipolar.cpp][]
    * Triangulation (Two-view Reconstruction): [triangulation.cpp][]
 * __Multi-view Geometry__
  * Bundle Adjustment using cvsba (Multiple-view Reconstruction): [bundle_adjustment.cpp][]
  * Sparse and Dense 3D Reconstruction using VisualSFM
 * __Correspondence Problem__
  * Line Fitting with RANSAC: [ransac_line.cpp][]

### Dependency
 * [OpenCV][] (> 3.0.0, 3-clause BSD License)
  * _OpenCV_ is a base of all example codes for linear algebra, vision algorithms, image/video manipulation, and GUI.
 * [cvsba][] (GPL): An OpenCV wrapper for sba library
  * _cvsba_ is used for bundle adjustment and already included in EXTERNAL directory.
 * [CLAPACK][] (Public Domain): f2c'ed version of LAPACK
  * _CLAPACK_ is necessary for cvsba.

### Contact
 * [Sunglok Choi](http://sites.google.com/site/sunglok/) (sunglok AT hanmail DOT net)

[OpenCV][http://opencv.org/]
[cvsba][https://www.uco.es/investiga/grupos/ava/node/39]
[CLAPACK][http://www.netlib.org/clapack/]
[image_formation.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/image_formation.cpp]
[distortion_correction.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/distortion_correction.cpp]
[camera_calibration.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/camera_calibration.cpp]
[pose_estimation_chessboard.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/pose_estimation_chessboard.cpp]
[perspective_correction.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/perspective_correction.cpp]
[image_stitching.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/image_stitching.cpp]
[video_stabilization.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/video_stabilization.cpp]
[visual_odometry_epipolar.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/visual_odometry_epipolar.cpp]
[triangulation.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/triangulation.cpp]
[bundle_adjustment.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/bundle_adjustment.cpp]
[ransac_line.cpp][https://github.com/sunglok/3dv_tutorial/blob/master/src/ransac_line.cpp]
