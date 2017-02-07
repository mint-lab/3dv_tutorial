/*
/////////////////////////////////////////////////////////////////////////////////
////
////  OpenCV Wrapper for sba library
////  Copyright (C) 2013
////  Sergio Garrido Jurado (i52gajus at uco es)
////  Rafael Munoz Salinas  (rmsalinas at uco es)
////  AVA Group. University of Cordoba, Spain
////
////  This program is free software; you can redistribute it and/or modify
////  it under the terms of the GNU General Public License as published by
////  the Free Software Foundation; either version 2 of the License, or
////  (at your option) any later version.
////
////  This program is distributed in the hope that it will be useful,
////  but WITHOUT ANY WARRANTY; without even the implied warranty of
////  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
////  GNU General Public License for more details.
////
///////////////////////////////////////////////////////////////////////////////////
*/

#ifndef CVSBA_H
#define CVSBA_H

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
namespace cvsba {

    class Sba {
        public:

        enum TYPE{MOTIONSTRUCTURE=0, MOTION, STRUCTURE };
        /**Params leading the optimization
         */
        struct Params {
            TYPE type; // type of sba: motionstructure(3dpoints+extrinsics), just motion(extrinsics) or just structure(3dpoints)
            int iterations; // max number of iterations, stop criteria
            double minError; // min error, stop criteria
            int fixedIntrinsics; //number of intrinsics parameters that keep fixed [0-5] (fx cx cy fy/fx s)
            int fixedDistortion; // number of distortion parameters that keep fixed [0-5] (k1 k2 p1 p2 k3)
            bool verbose;

            Params ( TYPE t= MOTIONSTRUCTURE, int iters = 150,double minErr = 1e-10,
            int  fixedIntri =5,int  fixedDist = 5,bool Verbose=false ) {

                type = t;
                iterations = iters;
                minError = minErr;
                fixedIntrinsics =fixedIntri;
                fixedDistortion = fixedDist;
                verbose=Verbose;
            }
        };
        /**Empty constructor
         */

        Sba();

        /**Sets params before optimization
         */
        void setParams ( Params params ) {
            _params = params;
        };

        /**
        * Run sba algorithm. We are using the same interface defined in cv::LevMarqSparse.
         *
         * Let N and M denote the number of object points and the number of cameras.
         *
         * @param  points N x 3 object points
         * @param  imagePoints M x N x 2 image points for each camera and each points. The outer  vector has M elements and each element has N elements of Point2d .
         * @param  visibility M x N x 1 visibility matrix, the element [i][j] = 1 when object point i is visible from camera j and 0 if not.
         * @param  cameraMatrix M x 3 x 3 camera matrix (intrinsic parameters) 3 x 3 camera matrix for each image
         * @param  distCoeffs M x   5  x1  distortion coefficient  for each image
         * @param R  M x 3 x 3 rotation matrix  for each image
         * @param T M x 3 x 1 translation matrix  for each image
         *
         * @return average reprojection error of the optimized solution
         *
         * @note matrices can be either  float or double, they are internally changed.
         */
        double run ( std::vector<cv::Point3d>& points, //positions of points in global coordinate system (input and output)
                     const std::vector<std::vector<cv::Point2d> >& imagePoints, //projections of 3d points for every camera
                     const std::vector<std::vector<int> >& visibility, //visibility of 3d points for every camera
                     std::vector<cv::Mat>& cameraMatrix, //intrinsic matrices of all cameras (input and output)
                     std::vector<cv::Mat>& R, //rotation matrices of all cameras (input and output) (Rodrigues format)
                     std::vector<cv::Mat>& T, //translation vector of all cameras (input and output)
                    std::vector<cv::Mat>& distCoeffs //distortion coefficients of all cameras (input and output)
                    )  throw ( cv::Exception );

        /**
         * Same but using float points
         */
        double run ( std::vector<cv::Point3f>& points, //positions of points in global coordinate system (input and output)
                     const std::vector<std::vector<cv::Point2f> >& imagePoints, //projections of 3d points for every camera
                     const std::vector<std::vector<int> >& visibility, //visibility of 3d points for every camera
                     std::vector<cv::Mat>& cameraMatrix, //intrinsic matrices of all cameras (input and output)
                     std::vector<cv::Mat>& R, //rotation matrices of all cameras (input and output) (Rodrigues format)
                     std::vector<cv::Mat>& T, //translation vector of all cameras (input and output)
                     std::vector<cv::Mat>& distCoeffs //distortion coefficients of all cameras (input and output)
                   )  throw ( cv::Exception );



        // Observer
        const Params getParams() {
            return _params;
        };

        /**Used after run, indicates the initial error before running the optimization
         */
        double getInitialReprjError() const{return _initRprjErr;}
        /**Used after run, indicates the  error after running the optimization. This is the same value returned by run
         */
        double getFinalReprjError() const {return _finalRprjErr;}



/// Auxiliar functions for quaternions

        /**
         * Convert 4-items quaternion to normalized 3-items quaternion
         */
        static void quat2normquat ( const cv::Mat& quat, cv::Mat& normquat );

        /**
         * Convert 3-items normalized quaternion to 4-items quaternion
         */
        static void normquat2quat ( const cv::Mat& normquat, cv::Mat& quat );

        /**
         * Convert 4-items quaternion to Rodrigues rotation vector
         */
        static void quat2rod ( const cv::Mat &quat, cv::Mat &rod );

        /**
         * Convert Rodrigues rotation vector to 4-items quaternion
         */
        static void rod2quat ( const cv::Mat& rod, cv::Mat& quat );



        private:

        double _initRprjErr,_finalRprjErr;

        Params _params;


        /// functions and data extracted from eucsbademo
        /// just for sba internal use
        struct globs_{
            double *rot0params;//8
            double *intrcalib;//8
            int nccalib;//4
            int ncdist;//4
            int cnp, pnp, mnp;//12
            double *ptparams;//4
            double *camparams;//4
        };
        static void img_projKDRTS ( int j, int i, double *aj, double *bi, double *xij, void *adata );
        static void img_projKDRT ( int j, int i, double *aj, double *xij, void *adata );
        static void img_projKDS ( int j, int i, double *bi, double *xij, void *adata );
        static void img_projKDRTS_jac ( int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata );
        static void img_projKDRT_jac ( int j, int i, double *aj, double *Aij, void *adata );
        static void img_projKDS_jac ( int j, int i, double *bi, double *Bij, void *adata );
        static void calcDistImgProj ( double a[5],double kc[5],double qr0[4],double v[3],double t[3],double M[3],double n[2] );
        static void calcDistImgProjJacKDRTS ( double a[5],double kc[5],double qr0[4], double v[3],double t[3],double M[3],double jacmKDRT[2][16],double jacmS[2][3] );
        static void calcDistImgProjJacKDRT ( double a[5],double kc[5],double qr0[4],double v[3],double t[3],double M[3],double jacmKDRT[2][16] );
        static void calcDistImgProjJacS ( double a[5],double kc[5],double qr0[4],double v[3],double t[3],double M[3],double jacmS[2][3] );

	void quatMultFast ( double q1[4], double q2[4], double p[4] );


    };








}
#endif // CVSBA_H
