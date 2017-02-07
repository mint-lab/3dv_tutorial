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

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include "cvsba.h"
#include "sba.h"
#include "compiler.h"
using namespace std;
namespace cvsba {


    /**
     *
     */
    Sba::Sba() {
        _initRprjErr=_finalRprjErr=-1;//unkown values yet
    };

    double  Sba::run ( std::vector<cv::Point3f>& points, //positions of points in global coordinate system (input and output)
                       const std::vector<std::vector<cv::Point2f> >& imagePoints, //projections of 3d points for every camera
                       const std::vector<std::vector<int> >& visibility, //visibility of 3d points for every camera
                       std::vector<cv::Mat>& cameraMatrix, //intrinsic matrices of all cameras (input and output)
                       std::vector<cv::Mat>& R, //rotation matrices of all cameras (input and output) (Rodrigues format)
                       std::vector<cv::Mat>& T, //translation vector of all cameras (input and output)
                       std::vector<cv::Mat>& distCoeffs  //distortion coefficients of all cameras (input and output)
                     )  throw ( cv::Exception ) {
        std::vector<cv::Point3d> points_d ( points.size() );
        for ( size_t i=0; i<points_d.size(); i++ )
            points_d[i]=cv::Point3d ( points[i].x,points[i].y,points[i].z );

        std::vector<std::vector<cv::Point2d> >  imagePoints_d ( imagePoints.size() );
        for ( size_t i=0; i<imagePoints.size(); i++ ) {
            imagePoints_d[i].resize ( imagePoints[i].size() );
            for ( size_t j=0; j<imagePoints[i].size(); j++ ) {
                imagePoints_d[i][j]=cv::Point2d ( imagePoints[i][j].x,imagePoints[i][j].y );
            }
        }
        //copy back if STRUCTURE enabled

        double err=run ( points_d,imagePoints_d,visibility,cameraMatrix,R,T,distCoeffs );
        if ( _params.type==MOTION || _params.type==MOTIONSTRUCTURE )
            for ( size_t i=0; i<points_d.size(); i++ )
                points[i]=cv::Point3f ( points_d[i].x,points_d[i].y,points_d[i].z );
        return err;
    }
#define _MK_QUAT_FRM_VEC(q, v){                                     \
  (q)[1]=(v)[0]; (q)[2]=(v)[1]; (q)[3]=(v)[2];                      \
  (q)[0]=sqrt(1.0 - (q)[1]*(q)[1] - (q)[2]*(q)[2]- (q)[3]*(q)[3]);  \
}
    /**
     *
     */
    double Sba::run ( std::vector<cv::Point3d>& points, //positions of points in global coordinate system (input and output)
                      const std::vector<std::vector<cv::Point2d> >& imagePoints, //projections of 3d points for every camera
                      const std::vector<std::vector<int> >& visibility, //visibility of 3d points for every camera
                      std::vector<cv::Mat>& cameraMatrix, //intrinsic matrices of all cameras (input and output)
                      std::vector<cv::Mat>& R, //rotation matrices of all cameras (input and output) (Rodrigues format)
                      std::vector<cv::Mat>& T, //translation vector of all cameras (input and output)
                      std::vector<cv::Mat>& distCoeffs //distortion coefficients of all cameras (input and output)
                    )  throw ( cv::Exception ) {
        if ( cameraMatrix.size() !=R.size() || R.size() !=T.size() || T.size() !=distCoeffs.size()  || distCoeffs.size() !=visibility.size() )
            throw cv::Exception ( 8888,"Error in input size vectors R,T... ","Sba::run",__FILE__,__LINE__ );

        //check further
        for ( size_t i=0; i<cameraMatrix.size(); i++ ) {
            if ( R[i].total() !=3 && R[i].total() !=9 )  throw cv::Exception ( 8888,"Error in R (must be 3x1, 1x3 or 3x3)","Sba::run",__FILE__,__LINE__ );
            if ( T[i].total() !=3 )  throw cv::Exception ( 8888,"Error in T (must be 3x1 or 1x3)","Sba::run",__FILE__,__LINE__ );
            if ( cameraMatrix[i].total() !=9 )  throw cv::Exception ( 8888,"Error in CameraMatrix  (must be 3x3)","Sba::run",__FILE__,__LINE__ );
            if ( distCoeffs[i].total() !=5 )  throw cv::Exception ( 8888,"Error in distortion coeff  (must be 1x5 or 5x1)","Sba::run",__FILE__,__LINE__ );
        }

        globs_ globs;
        int numpts3D = visibility[0].size(); //n
        int nframes = imagePoints.size(); //m
        int nconstframes = 0; //mcon
        int cnp = 16; // 3 rotation + 3 translation + 5 intrinsics + 5 distortion
        int pnp = 3; // 3 coordinates for 3d points
        int mnp = 2; // 2 coordinates for 2d points

        //change types (convert to 64 bits doubles)
        for ( size_t i=0; i<cameraMatrix.size(); i++ ) {
            cv::Mat aux;
            cameraMatrix[i].convertTo ( aux,CV_64F );
            cameraMatrix[i]=aux.clone();
            R[i].convertTo ( aux,CV_64F );
            R[i]=aux.clone();
            T[i].convertTo ( aux,CV_64F );
            T[i]=aux.clone();
            distCoeffs[i].convertTo ( aux,CV_64F );
            distCoeffs[i]=aux.clone();
        }


        // fill visibility mask
        char* vmask = new char[numpts3D*nframes];
        int nvisibles = 0;
        for ( int i=0; i<numpts3D; i++ ) {
            for ( int j=0; j<nframes; j++ ) {
                vmask[i*nframes+j] = visibility[j][i];
                if ( visibility[j][i] ) nvisibles++;
            }
        }

        // convert rot from rodrigues to quaternions
        std::vector<cv::Mat> Rquat;
        Rquat.resize ( R.size() );
        for ( int i=0; i<nframes; i++ ) {
            rod2quat ( R[i], Rquat[i] );
        }


        // fill intrinsics and extrinsics
        double* motstruct = new double[nframes*cnp + numpts3D*pnp]; //p
        for ( int i=0; i<nframes; i++ ) {
            motstruct[16*i+0] = cameraMatrix[i].ptr<double> ( 0 ) [0];
            motstruct[16*i+1] = cameraMatrix[i].ptr<double> ( 0 ) [2];
            motstruct[16*i+2] = cameraMatrix[i].ptr<double> ( 0 ) [5];
            motstruct[16*i+3] = cameraMatrix[i].ptr<double> ( 0 ) [4]/cameraMatrix[i].ptr<double> ( 0 ) [0];
            motstruct[16*i+4] = 0;
            for ( int j=0; j<5; j++ ) motstruct[16*i+5+j] = distCoeffs[i].ptr<double> ( 0 ) [j];
            //rot is set to 0 since what is actually calculated is the concatenation to the initial rotation (stored in globs)
            for ( int j=0; j<3; j++ ) motstruct[16*i+10+j] = 0.;
            for ( int j=0; j<3; j++ ) motstruct[16*i+13+j] = T[i].ptr<double> ( 0 ) [j];
        }

        if ( _params.type==STRUCTURE ) { // in this case, rotation is not estimated, so it is filled with initial values
            for ( int i=0; i<nframes; i++ ) {
                cv::Mat Rnormquat;
                quat2normquat ( Rquat[i], Rnormquat );
                for ( int j=0; j<3; j++ ) motstruct[16*i+10+j] = Rnormquat.ptr<double> ( 0 ) [j];
            }
        }

        // fill initial 3D points
        for ( int i=0; i<numpts3D; i++ ) {
            motstruct[16*nframes+3*i+0] = points[i].x;
            motstruct[16*nframes+3*i+1] = points[i].y;
            motstruct[16*nframes+3*i+2] = points[i].z;
        }


        // fill image points
        double* imgpts = new double[nvisibles*mnp];
        for ( int i=0, idx=0; i<numpts3D; i++ ) {
            for ( int j=0; j<nframes; j++ ) {
                if ( visibility[j][i] ) {
                    imgpts[idx] = imagePoints[j][i].x;
                    imgpts[idx+1] = imagePoints[j][i].y;
                    idx+=2;
                }
            }
        }

        // calculate num of projections
        // other parameters
        double* covimgpts=NULL; //double* covimgpts = new double[nframes*numpts3D*mnp*mnp]; //covx

        double opts[SBA_OPTSSZ], info[SBA_INFOSZ];
        opts[0]=SBA_INIT_MU;
        opts[1]=SBA_STOP_THRESH;
        opts[2]=SBA_STOP_THRESH;
//   opts[3]=SBA_STOP_THRESH;
        opts[3]=_params.minError*nvisibles;
        //opts[3]=0.05*nvisibles; // uncomment to force termination if the average reprojection error drops below 0.05
        opts[4]=0.0;
        //opts[4]=1E-05; // uncomment to force termination if the relative reduction in the RMS reprojection error drops below 1E-05
        int iterations = _params.iterations;
        if ( iterations<1 )
            throw cv::Exception ( 8888,"Invalid number of iterations","Sba::run",__FILE__,__LINE__ );

        int verbose = _params.verbose;
        // fill globs structure
        globs.cnp=cnp;
        globs.pnp=pnp;
        globs.mnp=mnp;

        globs.rot0params = new double[nframes*4];
        for ( int i=0; i<nframes; i++ )
            for ( int j=0; j<4; j++ ) globs.rot0params[i*4+j] = Rquat[i].ptr<double> ( 0 ) [j]; // initial rotation

        globs.intrcalib=NULL;
        globs.nccalib=_params.fixedIntrinsics; // number of intrinsics fixed parameters
        globs.ncdist=_params.fixedDistortion; // number of distortion fixed parameters
        globs.ptparams=NULL;
        globs.camparams=NULL;
        void* adata = NULL;
        adata = ( void * ) ( &globs );

        // call sba function
        int sbaerror;
        switch ( _params.type ) {
        case MOTIONSTRUCTURE:
            sbaerror=sba_motstr_levmar ( numpts3D, 0, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                                         img_projKDRTS,img_projKDRTS_jac,adata, iterations, verbose, opts, info );
            break;

        case MOTION:
            globs.ptparams=motstruct+nframes*cnp;
            sbaerror=sba_mot_levmar ( numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                                      img_projKDRT,img_projKDRT_jac,adata, iterations, verbose, opts, info );
            break;

        case STRUCTURE:
            globs.camparams=motstruct;
            sbaerror=sba_str_levmar ( numpts3D, 0, nframes, vmask, motstruct+nframes*cnp, pnp, imgpts, covimgpts, mnp,
                                      img_projKDS,img_projKDS_jac,adata, iterations, verbose, opts, info );
            break;

        default:
            throw cv::Exception ( 8888,"Invalid Sba mode","Sba::run",__FILE__,__LINE__ );
        }
        if ( sbaerror==SBA_ERROR )
            throw cv::Exception ( 8888,"Error occured during Sba optimization","Sba::run",__FILE__,__LINE__ );

        _finalRprjErr=info[1]/nvisibles ;
        _initRprjErr=info[0]/nvisibles ;
        // print error
//         std::cout << "Error: " << info[1]/nvisibles << " (Initial: " << info[0]/nvisibles << ")" << std::endl;

        //output data
        // extrinsics
        if ( _params.type!=STRUCTURE ) {
            /* combine the local rotation estimates with the initial ones */
            for ( int i=0; i<nframes; ++i ) {
                double *v, qs[4], q0[4], prd[4];

                /* retrieve the vector part */
                v=motstruct + ( i+1 ) *cnp - 6; // note the +1, we access the motion parameters from the right, assuming 3 for translation!
                _MK_QUAT_FRM_VEC ( qs, v );

                for ( int j=0; j<4; j++ )
                    q0[j]=Rquat[i].ptr<double> ( 0 ) [j];
                quatMultFast ( qs, q0, prd ); // prd=qs*q0

                cv::Mat mquat ( 1,4,CV_64F,prd ),mrod;
                quat2rod ( mquat,mrod );
                if ( R[i].total() ==3 ) R[i]=mrod.clone();
                else cv::Rodrigues ( mrod, R[i] ); //output as 3x3
                for ( int j=0; j<3; j++ ) T[i].ptr<double> ( 0 ) [j] = motstruct[16*i+13+j]; // save traslation

//       cout<<"q0="<<q0[0]<<" "<<q0[1]<<" "<<q0[2]<<" "<<q0[3]<<endl;
//       cout<<"qs="<<qs[0]<<" "<<qs[1]<<" "<<qs[2]<<" "<<qs[3]<<endl;
//       cout<<"prd="<<prd[0]<<" "<<prd[1]<<" "<<prd[2]<<" "<<prd[3]<<endl;
//       cout<<"T="<<T[i].ptr<double> ( 0 ) [0]<<" " <<T[i].ptr<double> ( 0 ) [1]<<" " <<T[i].ptr<double> ( 0 ) [2]<<endl;
//       cout<<"v="<< v[0]<<" "<< v[1]<<" "<< v[2]<<endl;
            }
        }/*

        if ( _params.type==MOTIONSTRUCTURE || _params.type==MOTION ) {
            for ( int i=0; i<nframes; i++ ) {
                // total Rot is concatenation of initial rot and obtained rotation
                cv::Mat rinit_mat, rquat3, rquat4, rrod, rrod_mat, totalrot;
                cv::Rodrigues ( R[i], rinit_mat );
                rquat3 = cv::Mat ( 3,1,CV_64FC1,cv::Scalar::all ( 0 ) );
                for ( int j=0; j<3; j++ ) rquat3.ptr<double> ( 0 ) [j] = motstruct[16*i+10+j]; // get normalized quaternion
                normquat2quat ( rquat3, rquat4 ); // transform to 4 item quaternion
                quat2rod ( rquat4, rrod ); // transform to rodrigues
                cv::Rodrigues ( rrod, rrod_mat ); // transform to rot matrix
                totalrot = rrod_mat*rinit_mat; // multiply with initial rotation
                cv::Rodrigues ( totalrot, R[i] ); // transform to rodriguez and save

                for ( int j=0; j<3; j++ ) T[i].ptr<double> ( 0 ) [j] = motstruct[16*i+13+j]; // save traslation
            }
        }*/

        // 3d points
        if ( _params.type==MOTIONSTRUCTURE || _params.type==STRUCTURE ) {
            for ( int i=0; i<numpts3D; i++ ) {
                points[i].x = motstruct[16*nframes+3*i+0];
                points[i].y = motstruct[16*nframes+3*i+1];
                points[i].z = motstruct[16*nframes+3*i+2];
            }
        }

        // camparams
        if ( _params.fixedIntrinsics<5 ) {
            for ( int i=0; i<nframes; i++ ) {
                cameraMatrix[i].ptr<double> ( 0 ) [0] = motstruct[16*i+0];
                cameraMatrix[i].ptr<double> ( 0 ) [2] = motstruct[16*i+1];
                cameraMatrix[i].ptr<double> ( 0 ) [5] = motstruct[16*i+2];
                cameraMatrix[i].ptr<double> ( 0 ) [4] = motstruct[16*i+3]*cameraMatrix[i].ptr<double> ( 0 ) [0];
            }
        }

        // distcoeffs
        for ( int j=_params.fixedDistortion; j<5; j++ )
            for ( int i=0; i<nframes; i++ )
                distCoeffs[i].ptr<double> ( 0 ) [j] = motstruct[16*i+5+j];


        // free memory
        if ( vmask!=NULL ) delete [] vmask;
        if ( motstruct!=NULL ) delete [] motstruct;
        if ( imgpts!=NULL ) delete [] imgpts;
        if ( covimgpts!=NULL ) delete [] covimgpts;
        if ( globs.rot0params!=NULL ) delete [] globs.rot0params;

        return _finalRprjErr;
    };


    /**
     *
     */
    void Sba::quat2normquat ( const cv::Mat& quat, cv::Mat& normquat ) {
        normquat = cv::Mat ( 3,1,CV_64FC1,cv::Scalar::all ( 0 ) );
        double mag, sg;
        mag=sqrt ( quat.ptr<double> ( 0 ) [0]*quat.ptr<double> ( 0 ) [0] + quat.ptr<double> ( 0 ) [1]*quat.ptr<double> ( 0 ) [1] +
                   quat.ptr<double> ( 0 ) [2]*quat.ptr<double> ( 0 ) [2] + quat.ptr<double> ( 0 ) [3]*quat.ptr<double> ( 0 ) [3] );
        sg= ( quat.ptr<double> ( 0 ) [0]>=0.0 ) ? 1.0 : -1.0;
        mag=sg/mag;
        normquat.ptr<double> ( 0 ) [0]  =quat.ptr<double> ( 0 ) [1]*mag;
        normquat.ptr<double> ( 0 ) [1]=quat.ptr<double> ( 0 ) [2]*mag;
        normquat.ptr<double> ( 0 ) [2]=quat.ptr<double> ( 0 ) [3]*mag;
    }


    /**
     *
     */
    void Sba::normquat2quat ( const cv::Mat& normquat, cv::Mat& quat ) {

        quat = cv::Mat ( 4,1,CV_64FC1,cv::Scalar::all ( 0 ) );

        quat.ptr<double> ( 0 ) [1] = normquat.ptr<double> ( 0 ) [0];
        quat.ptr<double> ( 0 ) [2] = normquat.ptr<double> ( 0 ) [1];
        quat.ptr<double> ( 0 ) [3] = normquat.ptr<double> ( 0 ) [2];
        quat.ptr<double> ( 0 ) [0] = sqrt ( 1.0 - quat.ptr<double> ( 0 ) [1]*quat.ptr<double> ( 0 ) [1] -
                                            quat.ptr<double> ( 0 ) [2]*quat.ptr<double> ( 0 ) [2] -
                                            quat.ptr<double> ( 0 ) [3]*quat.ptr<double> ( 0 ) [3] );

    }



    /**
     *
     */
    void Sba::quat2rod ( const cv::Mat &quat, cv::Mat &rod ) {
        cv::Mat mat = cv::Mat ( 3,3,CV_64FC1,cv::Scalar::all ( 0 ) );

        cv::Mat dquat;
        quat.convertTo ( dquat,CV_64F );
        double a=dquat.ptr<double> ( 0 ) [0];
        double b=dquat.ptr<double> ( 0 ) [1];
        double c=dquat.ptr<double> ( 0 ) [2];
        double d=dquat.ptr<double> ( 0 ) [3];

        mat .ptr<double> ( 0 ) [0]= a*a + b*b - c*c -d*d ;
        mat .ptr<double> ( 0 ) [1]= 2* ( b*c - a*d );
        mat .ptr<double> ( 0 ) [2]= 2* ( b*d + a*c );


        mat .ptr<double> ( 0 ) [3]= 2* ( b*c + a*d );
        mat .ptr<double> ( 0 ) [4]=  a*a - b*b + c*c - d*d ;
        mat .ptr<double> ( 0 ) [5]=  2* ( c*d - a*b );

        mat .ptr<double> ( 0 ) [6]= 2* ( b*d - a*c );
        mat .ptr<double> ( 0 ) [7]=  2* ( c*d + a*b );
        mat .ptr<double> ( 0 ) [8]=  a*a - b*b - c*c + d*d ;

        cv::Rodrigues ( mat,rod );
    }

    /**
     *
     */
    inline float SIGN ( float x ) {
        return ( x >= 0.0f ) ? +1.0f : -1.0f;
    }

    /**
     *
     */
    void Sba::rod2quat ( const cv::Mat& rod, cv::Mat& quat ) {
        cv::Mat mat;
        //check that rod is not 3x3 matrix already
        if ( rod.total() ==3 ) {
            cv::Mat aux;
            cv::Rodrigues ( rod, aux );
            aux.convertTo ( mat,CV_64FC1 );
        } else if ( rod.total() ==9 ) rod.convertTo ( mat,CV_64FC1 );
        else {
            cerr<<"rod2quat error in input matrix "<<__FILE__<<" "<<__LINE__<<endl;
        }

        quat = cv::Mat::zeros ( 4,1,CV_64FC1 );

        double r11=mat.ptr<double> ( 0 ) [0];
        double r12=mat.ptr<double> ( 0 ) [1];
        double r13=mat.ptr<double> ( 0 ) [2];
        double r21=mat.ptr<double> ( 0 ) [3];
        double r22=mat.ptr<double> ( 0 ) [4];
        double r23=mat.ptr<double> ( 0 ) [5];
        double r31=mat.ptr<double> ( 0 ) [6];
        double r32=mat.ptr<double> ( 0 ) [7];
        double r33=mat.ptr<double> ( 0 ) [8];

        double q0,q1,q2,q3;

        q0 = ( r11 + r22 + r33 + 1.0f ) / 4.0f;
        q1 = ( r11 - r22 - r33 + 1.0f ) / 4.0f;
        q2 = ( -r11 + r22 - r33 + 1.0f ) / 4.0f;
        q3 = ( -r11 - r22 + r33 + 1.0f ) / 4.0f;
        if ( q0 < 0.0f ) q0 = 0.0f;
        if ( q1 < 0.0f ) q1 = 0.0f;
        if ( q2 < 0.0f ) q2 = 0.0f;
        if ( q3 < 0.0f ) q3 = 0.0f;
        q0 = sqrt ( q0 );
        q1 = sqrt ( q1 );
        q2 = sqrt ( q2 );
        q3 = sqrt ( q3 );
        if ( q0 >= q1 && q0 >= q2 && q0 >= q3 ) {
            q0 *= +1.0f;
            q1 *= SIGN ( r32 - r23 );
            q2 *= SIGN ( r13 - r31 );
            q3 *= SIGN ( r21 - r12 );
        } else if ( q1 >= q0 && q1 >= q2 && q1 >= q3 ) {
            q0 *= SIGN ( r32 - r23 );
            q1 *= +1.0f;
            q2 *= SIGN ( r21 + r12 );
            q3 *= SIGN ( r13 + r31 );
        } else if ( q2 >= q0 && q2 >= q1 && q2 >= q3 ) {
            q0 *= SIGN ( r13 - r31 );
            q1 *= SIGN ( r21 + r12 );
            q2 *= +1.0f;
            q3 *= SIGN ( r32 + r23 );
        } else if ( q3 >= q0 && q3 >= q1 && q3 >= q2 ) {
            q0 *= SIGN ( r21 - r12 );
            q1 *= SIGN ( r31 + r13 );
            q2 *= SIGN ( r32 + r23 );
            q3 *= +1.0f;
        } else {
            std::cerr<<  "coding error" <<endl;
        }
        double r = sqrt ( q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3 );
        q0 /= r;
        q1 /= r;
        q2 /= r;
        q3 /= r;
        quat.ptr<double> ( 0 ) [0]=q0;
        quat.ptr<double> ( 0 ) [1]=q1;
        quat.ptr<double> ( 0 ) [2]=q2;
        quat.ptr<double> ( 0 ) [3]=q3;


    }





/// functions and data extracted from eucsbademo
/// for sba internal use

    /**
     *
     */
    void Sba::img_projKDRTS ( int j, int i, double *aj, double *bi, double *xij, void *adata ) {
        double *pr0;
        struct globs_ *gl;

        gl= ( struct globs_ * ) adata;
        pr0=gl->rot0params+j*4; // full quat for initial rotation estimate

        calcDistImgProj ( aj, aj+5, pr0, aj+5+5, aj+5+5+3, bi, xij ); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
    }


    /**
     *
     */
    void Sba::img_projKDRTS_jac ( int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata ) {
        struct globs_ *gl;
        double *pA, *pr0;
        int nc;

        gl= ( struct globs_ * ) adata;
        pr0=gl->rot0params+j*4; // full quat for initial rotation estimate
        calcDistImgProjJacKDRTS ( aj, aj+5, pr0, aj+5+5, aj+5+5+3, bi, ( double ( * ) [5+5+6] ) Aij, ( double ( * ) [3] ) Bij ); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

        /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
        gl= ( struct globs_ * ) adata;
        nc=gl->nccalib;
        if ( nc ) {
            int cnp, mnp, j0;

            pA=Aij;
            cnp=gl->cnp;
            mnp=gl->mnp;
            j0=5-nc;

            for ( i=0; i<mnp; ++i, pA+=cnp )
                for ( j=j0; j<5; ++j )
                    pA[j]=0.0; // pA[i*cnp+j]=0.0;
        }

        /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
        nc=gl->ncdist;
        if ( nc ) {
            int cnp, mnp, j0;

            pA=Aij;
            cnp=gl->cnp;
            mnp=gl->mnp;
            j0=5-nc;

            for ( i=0; i<mnp; ++i, pA+=cnp )
                for ( j=j0; j<5; ++j )
                    pA[5+j]=0.0; // pA[i*cnp+5+j]=0.0;
        }
    }


    /**
     *
     */
    void Sba::img_projKDRT ( int j, int i, double *aj, double *xij, void *adata ) {
        int pnp;

        double *ptparams, *pr0;
        struct globs_ *gl;

        gl= ( struct globs_ * ) adata;
        pnp=gl->pnp;
        ptparams=gl->ptparams;
        pr0=gl->rot0params+j*4; // full quat for initial rotation estimate

        calcDistImgProj ( aj, aj+5, pr0, aj+5+5, aj+5+5+3, ptparams+i*pnp, xij ); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
    }


    /**
     *
     */
    void Sba::img_projKDRT_jac ( int j, int i, double *aj, double *Aij, void *adata ) {
        struct globs_ *gl;
        double *pA, *ptparams, *pr0;
        int pnp, nc;

        gl= ( struct globs_ * ) adata;
        pnp=gl->pnp;
        ptparams=gl->ptparams;
        pr0=gl->rot0params+j*4; // full quat for initial rotation estimate

        calcDistImgProjJacKDRT ( aj, aj+5, pr0, aj+5+5, aj+5+5+3, ptparams+i*pnp, ( double ( * ) [5+5+6] ) Aij ); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

        /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
        nc=gl->nccalib;
        if ( nc ) {
            int cnp, mnp, j0;

            pA=Aij;
            cnp=gl->cnp;
            mnp=gl->mnp;
            j0=5-nc;

            for ( i=0; i<mnp; ++i, pA+=cnp )
                for ( j=j0; j<5; ++j )
                    pA[j]=0.0; // pA[i*cnp+j]=0.0;
        }
        nc=gl->ncdist;
        if ( nc ) {
            int cnp, mnp, j0;

            pA=Aij;
            cnp=gl->cnp;
            mnp=gl->mnp;
            j0=5-nc;

            for ( i=0; i<mnp; ++i, pA+=cnp )
                for ( j=j0; j<5; ++j )
                    pA[5+j]=0.0; // pA[i*cnp+5+j]=0.0;
        }
    }


    /**
     *
     */
    void Sba::img_projKDS ( int j, int i, double *bi, double *xij, void *adata ) {
        int cnp;

        double *camparams, *aj;
        struct globs_ *gl;

        gl= ( struct globs_ * ) adata;
        cnp=gl->cnp;
        camparams=gl->camparams;
        aj=camparams+j*cnp;


// calcDistImgProjFullR(aj, aj+5, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
        const double zerorotquat[4]= {1.0, 0.0, 0.0, 0.0};
        calcDistImgProj ( aj, aj+5, ( double * ) zerorotquat, aj+5+5, aj+5+5+3, bi, xij ); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
    }


    /**
     *
     */
    void Sba::img_projKDS_jac ( int j, int i, double *bi, double *Bij, void *adata ) {
        int cnp;

        double *camparams, *aj;
        struct globs_ *gl;

        gl= ( struct globs_ * ) adata;
        cnp=gl->cnp;
        camparams=gl->camparams;
        aj=camparams+j*cnp;

        const double zerorotquat[4]= {1.0, 0.0, 0.0, 0.0};
        calcDistImgProjJacS ( aj, aj+5, ( double * ) zerorotquat, aj+5+5, aj+5+5+3, bi, ( double ( * ) [3] ) Bij ); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
    }


    /**
     *
     */
    void Sba::calcDistImgProj ( double a[5],double kc[5],double qr0[4],double v[3], double t[3],double M[3],double n[2] ) {
        double t1;
        double t10;
        double t113;
        double t12;
        double t14;
        double t16;
        double t18;
        double t19;
        double t2;
        double t25;
        double t26;
        double t3;
        double t32;
        double t33;
        double t35;
        double t36;
        double t4;
        double t42;
        double t46;
        double t5;
        double t51;
        double t52;
        double t57;
        double t58;
        double t6;
        double t61;
        double t62;
        double t68;
        double t69;
        double t7;
        double t70;
        double t71;
        double t77;
        double t78;
        double t79;
        double t80;
        double t89;
        double t9;
        double t91;
        double t93;
        double t95;
        double t98;
        {
            t1 = a[0];
            t2 = v[0];
            t3 = t2*t2;
            t4 = v[1];
            t5 = t4*t4;
            t6 = v[2];
            t7 = t6*t6;
            t9 = sqrt ( 1.0-t3-t5-t7 );
            t10 = qr0[1];
            t12 = qr0[0];
            t14 = qr0[3];
            t16 = qr0[2];
            t18 = t9*t10+t12*t2+t4*t14-t6*t16;
            t19 = M[0];
            t25 = t9*t16+t12*t4+t6*t10-t2*t14;
            t26 = M[1];
            t32 = t9*t14+t12*t6+t2*t16-t4*t10;
            t33 = M[2];
            t35 = -t18*t19-t25*t26-t33*t32;
            t36 = -t18;
            t42 = t9*t12-t2*t10-t4*t16-t6*t14;
            t46 = t42*t19+t25*t33-t32*t26;
            t51 = t42*t26+t32*t19-t18*t33;
            t52 = -t32;
            t57 = t42*t33+t18*t26-t25*t19;
            t58 = -t25;
            t61 = t35*t36+t42*t46+t52*t51-t57*t58+t[0];
            t62 = t61*t61;
            t68 = t52*t35+t42*t57+t46*t58-t51*t36+t[2];
            t69 = t68*t68;
            t70 = 1/t69;
            t71 = t62*t70;
            t77 = t35*t58+t42*t51+t57*t36-t52*t46+t[1];
            t78 = t77*t77;
            t79 = t78*t70;
            t80 = t71+t79;
            t89 = 1.0+t80* ( kc[0]+t80* ( kc[1]+t80*kc[4] ) );
            t91 = 1/t68;
            t93 = kc[2];
            t95 = t70*t77;
            t98 = kc[3];
            t113 = t89*t77*t91+t93* ( t71+3.0*t79 ) +2.0*t98*t61*t95;
            n[0] = t1* ( t89*t61*t91+2.0*t93*t61*t95+t98* ( 3.0*t71+t79 ) ) +a[4]*t113+a[1];
            n[1] = t1*a[3]*t113+a[2];

            return;
        }
    }





    /**
     *
     */
    void Sba::calcDistImgProjJacKDRTS ( double a[5],double kc[5],double qr0[4], double v[3],double t[3],double M[3],double jacmKDRT[2][16],double jacmS[2][3] ) {
        double t1;
        double t101;
        double t102;
        double t105;
        double t107;
        double t11;
        double t110;
        double t111;
        double t113;
        double t115;
        double t117;
        double t119;
        double t120;
        double t122;
        double t13;
        double t137;
        double t140;
        double t147;
        double t148;
        double t15;
        double t150;
        double t152;
        double t154;
        double t156;
        double t158;
        double t160;
        double t162;
        double t164;
        double t166;
        double t17;
        double t171;
        double t176;
        double t178;
        double t18;
        double t183;
        double t185;
        double t187;
        double t188;
        double t190;
        double t191;
        double t2;
        double t200;
        double t201;
        double t210;
        double t211;
        double t212;
        double t213;
        double t215;
        double t222;
        double t227;
        double t232;
        double t233;
        double t236;
        double t24;
        double t25;
        double t265;
        double t268;
        double t271;
        double t274;
        double t276;
        double t278;
        double t281;
        double t286;
        double t291;
        double t293;
        double t298;
        double t3;
        double t300;
        double t302;
        double t303;
        double t31;
        double t312;
        double t313;
        double t32;
        double t322;
        double t323;
        double t324;
        double t326;
        double t333;
        double t338;
        double t34;
        double t343;
        double t346;
        double t35;
        double t375;
        double t378;
        double t381;
        double t384;
        double t386;
        double t388;
        double t391;
        double t396;
        double t4;
        double t401;
        double t403;
        double t408;
        double t41;
        double t410;
        double t412;
        double t413;
        double t422;
        double t423;
        double t432;
        double t433;
        double t434;
        double t436;
        double t443;
        double t448;
        double t45;
        double t453;
        double t456;
        double t485;
        double t491;
        double t496;
        double t499;
        double t5;
        double t50;
        double t501;
        double t503;
        double t51;
        double t510;
        double t513;
        double t514;
        double t523;
        double t532;
        double t535;
        double t542;
        double t56;
        double t563;
        double t565;
        double t566;
        double t568;
        double t569;
        double t57;
        double t570;
        double t571;
        double t572;
        double t575;
        double t576;
        double t577;
        double t580;
        double t581;
        double t582;
        double t583;
        double t585;
        double t592;
        double t597;
        double t6;
        double t60;
        double t602;
        double t605;
        double t61;
        double t634;
        double t638;
        double t639;
        double t640;
        double t643;
        double t644;
        double t645;
        double t647;
        double t648;
        double t649;
        double t650;
        double t652;
        double t659;
        double t664;
        double t669;
        double t67;
        double t672;
        double t68;
        double t69;
        double t70;
        double t701;
        double t705;
        double t706;
        double t708;
        double t709;
        double t712;
        double t713;
        double t714;
        double t716;
        double t723;
        double t728;
        double t733;
        double t736;
        double t76;
        double t765;
        double t77;
        double t78;
        double t79;
        double t8;
        double t82;
        double t84;
        double t86;
        double t88;
        double t89;
        double t9;
        double t90;
        double t92;
        double t93;
        double t94;
        double t97;
        double t99;
        {
            t1 = v[0];
            t2 = t1*t1;
            t3 = v[1];
            t4 = t3*t3;
            t5 = v[2];
            t6 = t5*t5;
            t8 = sqrt ( 1.0-t2-t4-t6 );
            t9 = qr0[1];
            t11 = qr0[0];
            t13 = qr0[3];
            t15 = qr0[2];
            t17 = t9*t8+t11*t1+t13*t3-t5*t15;
            t18 = M[0];
            t24 = t8*t15+t11*t3+t5*t9-t1*t13;
            t25 = M[1];
            t31 = t8*t13+t5*t11+t1*t15-t3*t9;
            t32 = M[2];
            t34 = -t17*t18-t24*t25-t31*t32;
            t35 = -t17;
            t41 = t8*t11-t1*t9-t3*t15-t5*t13;
            t45 = t41*t18+t24*t32-t31*t25;
            t50 = t41*t25+t31*t18-t32*t17;
            t51 = -t31;
            t56 = t41*t32+t17*t25-t24*t18;
            t57 = -t24;
            t60 = t34*t35+t41*t45+t50*t51-t56*t57+t[0];
            t61 = t60*t60;
            t67 = t34*t51+t41*t56+t45*t57-t50*t35+t[2];
            t68 = t67*t67;
            t69 = 1/t68;
            t70 = t61*t69;
            t76 = t34*t57+t41*t50+t56*t35-t45*t51+t[1];
            t77 = t76*t76;
            t78 = t77*t69;
            t79 = t70+t78;
            t82 = kc[4];
            t84 = kc[1]+t79*t82;
            t86 = kc[0]+t79*t84;
            t88 = 1.0+t79*t86;
            t89 = t88*t60;
            t90 = 1/t67;
            t92 = kc[2];
            t93 = t92*t60;
            t94 = t69*t76;
            t97 = kc[3];
            t99 = 3.0*t70+t78;
            jacmKDRT[0][0] = t89*t90+2.0*t93*t94+t97*t99;
            t101 = a[3];
            t102 = t88*t76;
            t105 = t70+3.0*t78;
            t107 = t97*t60;
            t110 = t102*t90+t92*t105+2.0*t107*t94;
            jacmKDRT[1][0] = t101*t110;
            jacmKDRT[0][1] = 1.0;
            jacmKDRT[1][1] = 0.0;
            jacmKDRT[0][2] = 0.0;
            jacmKDRT[1][2] = 1.0;
            jacmKDRT[0][3] = 0.0;
            t111 = a[0];
            jacmKDRT[1][3] = t110*t111;
            jacmKDRT[0][4] = t110;
            jacmKDRT[1][4] = 0.0;
            t113 = t60*t90;
            t115 = a[4];
            t117 = t76*t90;
            jacmKDRT[0][5] = t111*t79*t113+t115*t79*t117;
            t119 = t111*t101;
            t120 = t79*t76;
            jacmKDRT[1][5] = t119*t120*t90;
            t122 = t79*t79;
            jacmKDRT[0][6] = t111*t122*t113+t115*t122*t117;
            jacmKDRT[1][6] = t119*t122*t76*t90;
            jacmKDRT[0][7] = 2.0*t111*t60*t94+t115*t105;
            jacmKDRT[1][7] = t119*t105;
            jacmKDRT[0][8] = t111*t99+2.0*t115*t60*t94;
            t137 = t60*t69;
            jacmKDRT[1][8] = 2.0*t119*t137*t76;
            t140 = t122*t79;
            jacmKDRT[0][9] = t111*t140*t113+t115*t140*t117;
            jacmKDRT[1][9] = t119*t140*t76*t90;
            t147 = 1/t8;
            t148 = t147*t9;
            t150 = -t148*t1+t11;
            t152 = t147*t15;
            t154 = -t152*t1-t13;
            t156 = t13*t147;
            t158 = -t156*t1+t15;
            t160 = -t150*t18-t154*t25-t158*t32;
            t162 = -t150;
            t164 = t147*t11;
            t166 = -t164*t1-t9;
            t171 = t166*t18+t154*t32-t158*t25;
            t176 = t166*t25+t158*t18-t150*t32;
            t178 = -t158;
            t183 = t32*t166+t150*t25-t154*t18;
            t185 = -t154;
            t187 = t160*t35+t34*t162+t166*t45+t41*t171+t176*t51+t50*t178-t183*t57-t56*
                   t185;
            t188 = t137*t187;
            t190 = 1/t68/t67;
            t191 = t61*t190;
            t200 = t160*t51+t34*t178+t166*t56+t41*t183+t171*t57+t45*t185-t176*t35-t50*
                   t162;
            t201 = t191*t200;
            t210 = t160*t57+t34*t185+t166*t50+t41*t176+t183*t35+t56*t162-t171*t51-t45*
                   t178;
            t211 = t94*t210;
            t212 = t77*t190;
            t213 = t212*t200;
            t215 = 2.0*t188-2.0*t201+2.0*t211-2.0*t213;
            t222 = t215*t86+t79* ( t215*t84+t79*t215*t82 );
            t227 = t69*t200;
            t232 = t190*t76;
            t233 = t232*t200;
            t236 = t69*t210;
            t265 = t222*t76*t90+t88*t210*t90-t102*t227+t92* ( 2.0*t188-2.0*t201+6.0*t211
                    -6.0*t213 ) +2.0*t97*t187*t94-4.0*t107*t233+2.0*t107*t236;
            jacmKDRT[0][10] = t111* ( t222*t60*t90+t88*t187*t90-t89*t227+2.0*t92*t187*t94
                                      -4.0*t93*t233+2.0*t93*t236+t97* ( 6.0*t188-6.0*t201+2.0*t211-2.0*t213 ) ) +t115*t265
                              ;
            jacmKDRT[1][10] = t119*t265;
            t268 = -t148*t3+t13;
            t271 = -t152*t3+t11;
            t274 = -t156*t3-t9;
            t276 = -t268*t18-t271*t25-t274*t32;
            t278 = -t268;
            t281 = -t164*t3-t15;
            t286 = t281*t18+t271*t32-t274*t25;
            t291 = t281*t25+t274*t18-t32*t268;
            t293 = -t274;
            t298 = t281*t32+t268*t25-t271*t18;
            t300 = -t271;
            t302 = t276*t35+t34*t278+t281*t45+t41*t286+t291*t51+t50*t293-t298*t57-t56*
                   t300;
            t303 = t137*t302;
            t312 = t276*t51+t34*t293+t281*t56+t41*t298+t286*t57+t45*t300-t291*t35-t50*
                   t278;
            t313 = t191*t312;
            t322 = t276*t57+t34*t300+t281*t50+t41*t291+t298*t35+t56*t278-t286*t51-t45*
                   t293;
            t323 = t94*t322;
            t324 = t212*t312;
            t326 = 2.0*t303-2.0*t313+2.0*t323-2.0*t324;
            t333 = t326*t86+t79* ( t326*t84+t79*t326*t82 );
            t338 = t69*t312;
            t343 = t312*t232;
            t346 = t69*t322;
            t375 = t333*t76*t90+t88*t322*t90-t338*t102+t92* ( 2.0*t303-2.0*t313+6.0*t323
                    -6.0*t324 ) +2.0*t97*t302*t94-4.0*t107*t343+2.0*t346*t107;
            jacmKDRT[0][11] = t111* ( t333*t60*t90+t88*t302*t90-t89*t338+2.0*t92*t302*t94
                                      -4.0*t93*t343+2.0*t93*t346+t97* ( 6.0*t303-6.0*t313+2.0*t323-2.0*t324 ) ) +t115*t375
                              ;
            jacmKDRT[1][11] = t119*t375;
            t378 = -t148*t5-t15;
            t381 = -t152*t5+t9;
            t384 = -t156*t5+t11;
            t386 = -t378*t18-t25*t381-t384*t32;
            t388 = -t378;
            t391 = -t164*t5-t13;
            t396 = t391*t18+t381*t32-t384*t25;
            t401 = t25*t391+t384*t18-t378*t32;
            t403 = -t384;
            t408 = t391*t32+t378*t25-t381*t18;
            t410 = -t381;
            t412 = t35*t386+t388*t34+t391*t45+t41*t396+t401*t51+t50*t403-t408*t57-t56*
                   t410;
            t413 = t137*t412;
            t422 = t386*t51+t34*t403+t56*t391+t41*t408+t396*t57+t45*t410-t401*t35-t50*
                   t388;
            t423 = t191*t422;
            t432 = t386*t57+t34*t410+t391*t50+t41*t401+t408*t35+t56*t388-t396*t51-t45*
                   t403;
            t433 = t94*t432;
            t434 = t212*t422;
            t436 = 2.0*t413-2.0*t423+2.0*t433-2.0*t434;
            t443 = t436*t86+t79* ( t436*t84+t79*t436*t82 );
            t448 = t69*t422;
            t453 = t232*t422;
            t456 = t69*t432;
            t485 = t443*t76*t90+t88*t432*t90-t102*t448+t92* ( 2.0*t413-2.0*t423+6.0*t433
                    -6.0*t434 ) +2.0*t97*t412*t94-4.0*t107*t453+2.0*t107*t456;
            jacmKDRT[0][12] = t111* ( t443*t60*t90+t88*t412*t90-t89*t448+2.0*t92*t412*t94
                                      -4.0*t93*t453+2.0*t93*t456+t97* ( 6.0*t413-6.0*t423+2.0*t433-2.0*t434 ) ) +t115*t485
                              ;
            jacmKDRT[1][12] = t119*t485;
            t491 = t69*t82;
            t496 = 2.0*t137*t86+t79* ( 2.0*t137*t84+2.0*t79*t60*t491 );
            t499 = t88*t90;
            t501 = t92*t69*t76;
            t503 = t107*t69;
            t510 = 2.0*t93*t69;
            t513 = 2.0*t97*t69*t76;
            t514 = t496*t76*t90+t510+t513;
            jacmKDRT[0][13] = t111* ( t496*t60*t90+t499+2.0*t501+6.0*t503 ) +t115*t514;
            jacmKDRT[1][13] = t119*t514;
            t523 = 2.0*t94*t86+t79* ( 2.0*t94*t84+2.0*t120*t491 );
            t532 = t523*t76*t90+t499+6.0*t501+2.0*t503;
            jacmKDRT[0][14] = t111* ( t523*t60*t90+t510+t513 ) +t115*t532;
            jacmKDRT[1][14] = t119*t532;
            t535 = -2.0*t191-2.0*t212;
            t542 = t535*t86+t79* ( t535*t84+t79*t535*t82 );
            t563 = t542*t76*t90-t102*t69+t92* ( -2.0*t191-6.0*t212 )-4.0*t107*t232;
            jacmKDRT[0][15] = t111* ( t542*t60*t90-t89*t69-4.0*t93*t232+t97* ( -6.0*t191
                                      -2.0*t212 ) ) +t115*t563;
            jacmKDRT[1][15] = t119*t563;
            t565 = t35*t35;
            t566 = t41*t41;
            t568 = t57*t57;
            t569 = t565+t566+t31*t51-t568;
            t570 = t137*t569;
            t571 = t35*t51;
            t572 = t41*t57;
            t575 = t571+2.0*t572-t31*t35;
            t576 = t191*t575;
            t577 = t35*t57;
            t580 = t41*t51;
            t581 = 2.0*t577+t41*t31-t580;
            t582 = t94*t581;
            t583 = t212*t575;
            t585 = 2.0*t570-2.0*t576+2.0*t582-2.0*t583;
            t592 = t585*t86+t79* ( t84*t585+t79*t585*t82 );
            t597 = t69*t575;
            t602 = t232*t575;
            t605 = t69*t581;
            t634 = t592*t76*t90+t88*t581*t90-t102*t597+t92* ( 2.0*t570-2.0*t576+6.0*t582
                    -6.0*t583 ) +2.0*t97*t569*t94-4.0*t107*t602+2.0*t107*t605;
            jacmS[0][0] = t111* ( t592*t60*t90+t88*t569*t90-t89*t597+2.0*t92*t569*t94-4.0
                                  *t93*t602+2.0*t93*t605+t97* ( 6.0*t570-6.0*t576+2.0*t582-2.0*t583 ) ) +t115*t634;
            jacmS[1][0] = t119*t634;
            t638 = t577+2.0*t580-t17*t57;
            t639 = t137*t638;
            t640 = t57*t51;
            t643 = t41*t35;
            t644 = 2.0*t640+t41*t17-t643;
            t645 = t191*t644;
            t647 = t51*t51;
            t648 = t568+t566+t17*t35-t647;
            t649 = t94*t648;
            t650 = t212*t644;
            t652 = 2.0*t639-2.0*t645+2.0*t649-2.0*t650;
            t659 = t652*t86+t79* ( t652*t84+t79*t652*t82 );
            t664 = t69*t644;
            t669 = t232*t644;
            t672 = t69*t648;
            t701 = t659*t76*t90+t88*t648*t90-t102*t664+t92* ( 2.0*t639-2.0*t645+6.0*t649
                    -6.0*t650 ) +2.0*t97*t638*t94-4.0*t107*t669+2.0*t107*t672;
            jacmS[0][1] = t111* ( t659*t60*t90+t88*t638*t90-t89*t664+2.0*t92*t638*t94-4.0
                                  *t93*t669+2.0*t93*t672+t97* ( 6.0*t639-6.0*t645+2.0*t649-2.0*t650 ) ) +t115*t701;
            jacmS[1][1] = t119*t701;
            t705 = 2.0*t571+t41*t24-t572;
            t706 = t137*t705;
            t708 = t647+t566+t57*t24-t565;
            t709 = t191*t708;
            t712 = t640+2.0*t643-t24*t51;
            t713 = t94*t712;
            t714 = t212*t708;
            t716 = 2.0*t706-2.0*t709+2.0*t713-2.0*t714;
            t723 = t716*t86+t79* ( t716*t84+t79*t716*t82 );
            t728 = t69*t708;
            t733 = t232*t708;
            t736 = t69*t712;
            t765 = t723*t76*t90+t88*t712*t90-t102*t728+t92* ( 2.0*t706-2.0*t709+6.0*t713
                    -6.0*t714 ) +2.0*t97*t705*t94-4.0*t107*t733+2.0*t107*t736;
            jacmS[0][2] = t111* ( t723*t60*t90+t88*t705*t90-t89*t728+2.0*t92*t705*t94-4.0
                                  *t93*t733+2.0*t93*t736+t97* ( 6.0*t706-6.0*t709+2.0*t713-2.0*t714 ) ) +t115*t765;
            jacmS[1][2] = t119*t765;
            return;
        }
    }


    /**
     *
     */
    void Sba::calcDistImgProjJacKDRT ( double a[5],double kc[5],double qr0[4],double v[3],
                                       double t[3],double M[3],double jacmKDRT[2][16] ) {
        double t1;
        double t101;
        double t102;
        double t105;
        double t107;
        double t11;
        double t110;
        double t111;
        double t113;
        double t115;
        double t117;
        double t119;
        double t120;
        double t122;
        double t13;
        double t137;
        double t140;
        double t147;
        double t148;
        double t15;
        double t150;
        double t152;
        double t154;
        double t156;
        double t158;
        double t160;
        double t162;
        double t164;
        double t166;
        double t17;
        double t171;
        double t176;
        double t178;
        double t18;
        double t183;
        double t185;
        double t187;
        double t188;
        double t190;
        double t191;
        double t2;
        double t200;
        double t201;
        double t210;
        double t211;
        double t212;
        double t213;
        double t215;
        double t222;
        double t227;
        double t232;
        double t233;
        double t236;
        double t24;
        double t25;
        double t265;
        double t268;
        double t271;
        double t274;
        double t276;
        double t278;
        double t281;
        double t286;
        double t291;
        double t293;
        double t298;
        double t3;
        double t300;
        double t302;
        double t303;
        double t31;
        double t312;
        double t313;
        double t32;
        double t322;
        double t323;
        double t324;
        double t326;
        double t333;
        double t338;
        double t34;
        double t343;
        double t346;
        double t35;
        double t375;
        double t378;
        double t381;
        double t384;
        double t386;
        double t388;
        double t391;
        double t396;
        double t4;
        double t401;
        double t403;
        double t408;
        double t41;
        double t410;
        double t412;
        double t413;
        double t422;
        double t423;
        double t432;
        double t433;
        double t434;
        double t436;
        double t443;
        double t448;
        double t45;
        double t453;
        double t456;
        double t485;
        double t491;
        double t496;
        double t499;
        double t5;
        double t50;
        double t501;
        double t503;
        double t51;
        double t510;
        double t513;
        double t514;
        double t523;
        double t532;
        double t535;
        double t542;
        double t56;
        double t563;
        double t57;
        double t6;
        double t60;
        double t61;
        double t67;
        double t68;
        double t69;
        double t70;
        double t76;
        double t77;
        double t78;
        double t79;
        double t8;
        double t82;
        double t84;
        double t86;
        double t88;
        double t89;
        double t9;
        double t90;
        double t92;
        double t93;
        double t94;
        double t97;
        double t99;
        {
            t1 = v[0];
            t2 = t1*t1;
            t3 = v[1];
            t4 = t3*t3;
            t5 = v[2];
            t6 = t5*t5;
            t8 = sqrt ( 1.0-t2-t4-t6 );
            t9 = qr0[1];
            t11 = qr0[0];
            t13 = qr0[3];
            t15 = qr0[2];
            t17 = t9*t8+t11*t1+t13*t3-t5*t15;
            t18 = M[0];
            t24 = t8*t15+t11*t3+t5*t9-t1*t13;
            t25 = M[1];
            t31 = t8*t13+t5*t11+t1*t15-t3*t9;
            t32 = M[2];
            t34 = -t17*t18-t24*t25-t31*t32;
            t35 = -t17;
            t41 = t8*t11-t1*t9-t3*t15-t5*t13;
            t45 = t41*t18+t24*t32-t31*t25;
            t50 = t41*t25+t31*t18-t32*t17;
            t51 = -t31;
            t56 = t41*t32+t17*t25-t24*t18;
            t57 = -t24;
            t60 = t34*t35+t41*t45+t50*t51-t56*t57+t[0];
            t61 = t60*t60;
            t67 = t34*t51+t41*t56+t45*t57-t50*t35+t[2];
            t68 = t67*t67;
            t69 = 1/t68;
            t70 = t61*t69;
            t76 = t34*t57+t41*t50+t56*t35-t45*t51+t[1];
            t77 = t76*t76;
            t78 = t77*t69;
            t79 = t70+t78;
            t82 = kc[4];
            t84 = kc[1]+t79*t82;
            t86 = kc[0]+t79*t84;
            t88 = 1.0+t79*t86;
            t89 = t88*t60;
            t90 = 1/t67;
            t92 = kc[2];
            t93 = t92*t60;
            t94 = t69*t76;
            t97 = kc[3];
            t99 = 3.0*t70+t78;
            jacmKDRT[0][0] = t89*t90+2.0*t93*t94+t97*t99;
            t101 = a[3];
            t102 = t88*t76;
            t105 = t70+3.0*t78;
            t107 = t97*t60;
            t110 = t102*t90+t92*t105+2.0*t107*t94;
            jacmKDRT[1][0] = t101*t110;
            jacmKDRT[0][1] = 1.0;
            jacmKDRT[1][1] = 0.0;
            jacmKDRT[0][2] = 0.0;
            jacmKDRT[1][2] = 1.0;
            jacmKDRT[0][3] = 0.0;
            t111 = a[0];
            jacmKDRT[1][3] = t110*t111;
            jacmKDRT[0][4] = t110;
            jacmKDRT[1][4] = 0.0;
            t113 = t60*t90;
            t115 = a[4];
            t117 = t76*t90;
            jacmKDRT[0][5] = t111*t79*t113+t115*t79*t117;
            t119 = t111*t101;
            t120 = t79*t76;
            jacmKDRT[1][5] = t119*t120*t90;
            t122 = t79*t79;
            jacmKDRT[0][6] = t111*t122*t113+t115*t122*t117;
            jacmKDRT[1][6] = t119*t122*t76*t90;
            jacmKDRT[0][7] = 2.0*t111*t60*t94+t115*t105;
            jacmKDRT[1][7] = t119*t105;
            jacmKDRT[0][8] = t111*t99+2.0*t115*t60*t94;
            t137 = t60*t69;
            jacmKDRT[1][8] = 2.0*t119*t137*t76;
            t140 = t122*t79;
            jacmKDRT[0][9] = t111*t140*t113+t115*t140*t117;
            jacmKDRT[1][9] = t119*t140*t76*t90;
            t147 = 1/t8;
            t148 = t147*t9;
            t150 = -t148*t1+t11;
            t152 = t147*t15;
            t154 = -t152*t1-t13;
            t156 = t13*t147;
            t158 = -t156*t1+t15;
            t160 = -t150*t18-t154*t25-t158*t32;
            t162 = -t150;
            t164 = t147*t11;
            t166 = -t164*t1-t9;
            t171 = t166*t18+t154*t32-t158*t25;
            t176 = t166*t25+t158*t18-t150*t32;
            t178 = -t158;
            t183 = t32*t166+t150*t25-t154*t18;
            t185 = -t154;
            t187 = t160*t35+t34*t162+t166*t45+t41*t171+t176*t51+t50*t178-t183*t57-t56*
                   t185;
            t188 = t137*t187;
            t190 = 1/t68/t67;
            t191 = t61*t190;
            t200 = t160*t51+t34*t178+t166*t56+t41*t183+t171*t57+t45*t185-t176*t35-t50*
                   t162;
            t201 = t191*t200;
            t210 = t160*t57+t34*t185+t166*t50+t41*t176+t183*t35+t56*t162-t171*t51-t45*
                   t178;
            t211 = t94*t210;
            t212 = t77*t190;
            t213 = t212*t200;
            t215 = 2.0*t188-2.0*t201+2.0*t211-2.0*t213;
            t222 = t215*t86+t79* ( t215*t84+t79*t215*t82 );
            t227 = t69*t200;
            t232 = t190*t76;
            t233 = t232*t200;
            t236 = t69*t210;
            t265 = t222*t76*t90+t88*t210*t90-t102*t227+t92* ( 2.0*t188-2.0*t201+6.0*t211
                    -6.0*t213 ) +2.0*t97*t187*t94-4.0*t107*t233+2.0*t107*t236;
            jacmKDRT[0][10] = t111* ( t222*t60*t90+t88*t187*t90-t89*t227+2.0*t92*t187*t94
                                      -4.0*t93*t233+2.0*t93*t236+t97* ( 6.0*t188-6.0*t201+2.0*t211-2.0*t213 ) ) +t115*t265
                              ;
            jacmKDRT[1][10] = t119*t265;
            t268 = -t148*t3+t13;
            t271 = -t152*t3+t11;
            t274 = -t156*t3-t9;
            t276 = -t268*t18-t271*t25-t274*t32;
            t278 = -t268;
            t281 = -t164*t3-t15;
            t286 = t281*t18+t271*t32-t274*t25;
            t291 = t281*t25+t274*t18-t32*t268;
            t293 = -t274;
            t298 = t281*t32+t268*t25-t271*t18;
            t300 = -t271;
            t302 = t276*t35+t34*t278+t281*t45+t41*t286+t291*t51+t50*t293-t298*t57-t56*
                   t300;
            t303 = t137*t302;
            t312 = t276*t51+t34*t293+t281*t56+t41*t298+t286*t57+t45*t300-t291*t35-t50*
                   t278;
            t313 = t191*t312;
            t322 = t276*t57+t34*t300+t281*t50+t41*t291+t298*t35+t56*t278-t286*t51-t45*
                   t293;
            t323 = t94*t322;
            t324 = t212*t312;
            t326 = 2.0*t303-2.0*t313+2.0*t323-2.0*t324;
            t333 = t326*t86+t79* ( t326*t84+t79*t326*t82 );
            t338 = t69*t312;
            t343 = t312*t232;
            t346 = t69*t322;
            t375 = t333*t76*t90+t88*t322*t90-t338*t102+t92* ( 2.0*t303-2.0*t313+6.0*t323
                    -6.0*t324 ) +2.0*t97*t302*t94-4.0*t107*t343+2.0*t346*t107;
            jacmKDRT[0][11] = t111* ( t333*t60*t90+t88*t302*t90-t89*t338+2.0*t92*t302*t94
                                      -4.0*t93*t343+2.0*t93*t346+t97* ( 6.0*t303-6.0*t313+2.0*t323-2.0*t324 ) ) +t115*t375
                              ;
            jacmKDRT[1][11] = t119*t375;
            t378 = -t148*t5-t15;
            t381 = -t152*t5+t9;
            t384 = -t156*t5+t11;
            t386 = -t378*t18-t25*t381-t384*t32;
            t388 = -t378;
            t391 = -t164*t5-t13;
            t396 = t391*t18+t381*t32-t384*t25;
            t401 = t25*t391+t384*t18-t378*t32;
            t403 = -t384;
            t408 = t391*t32+t378*t25-t381*t18;
            t410 = -t381;
            t412 = t35*t386+t388*t34+t391*t45+t41*t396+t401*t51+t50*t403-t408*t57-t56*
                   t410;
            t413 = t137*t412;
            t422 = t386*t51+t34*t403+t56*t391+t41*t408+t396*t57+t45*t410-t401*t35-t50*
                   t388;
            t423 = t191*t422;
            t432 = t386*t57+t34*t410+t391*t50+t41*t401+t408*t35+t56*t388-t396*t51-t45*
                   t403;
            t433 = t94*t432;
            t434 = t212*t422;
            t436 = 2.0*t413-2.0*t423+2.0*t433-2.0*t434;
            t443 = t436*t86+t79* ( t436*t84+t79*t436*t82 );
            t448 = t69*t422;
            t453 = t232*t422;
            t456 = t69*t432;
            t485 = t443*t76*t90+t88*t432*t90-t102*t448+t92* ( 2.0*t413-2.0*t423+6.0*t433
                    -6.0*t434 ) +2.0*t97*t412*t94-4.0*t107*t453+2.0*t107*t456;
            jacmKDRT[0][12] = t111* ( t443*t60*t90+t88*t412*t90-t89*t448+2.0*t92*t412*t94
                                      -4.0*t93*t453+2.0*t93*t456+t97* ( 6.0*t413-6.0*t423+2.0*t433-2.0*t434 ) ) +t115*t485
                              ;
            jacmKDRT[1][12] = t119*t485;
            t491 = t69*t82;
            t496 = 2.0*t137*t86+t79* ( 2.0*t137*t84+2.0*t79*t60*t491 );
            t499 = t88*t90;
            t501 = t92*t69*t76;
            t503 = t107*t69;
            t510 = 2.0*t93*t69;
            t513 = 2.0*t97*t69*t76;
            t514 = t496*t76*t90+t510+t513;
            jacmKDRT[0][13] = t111* ( t496*t60*t90+t499+2.0*t501+6.0*t503 ) +t115*t514;
            jacmKDRT[1][13] = t119*t514;
            t523 = 2.0*t94*t86+t79* ( 2.0*t94*t84+2.0*t120*t491 );
            t532 = t523*t76*t90+t499+6.0*t501+2.0*t503;
            jacmKDRT[0][14] = t111* ( t523*t60*t90+t510+t513 ) +t115*t532;
            jacmKDRT[1][14] = t119*t532;
            t535 = -2.0*t191-2.0*t212;
            t542 = t535*t86+t79* ( t535*t84+t79*t535*t82 );
            t563 = t542*t76*t90-t102*t69+t92* ( -2.0*t191-6.0*t212 )-4.0*t107*t232;
            jacmKDRT[0][15] = t111* ( t542*t60*t90-t89*t69-4.0*t93*t232+t97* ( -6.0*t191
                                      -2.0*t212 ) ) +t115*t563;
            jacmKDRT[1][15] = t119*t563;
            return;
        }
    }


    /**
     *
     */
    void Sba::calcDistImgProjJacS ( double a[5],double kc[5],double qr0[4],double v[3],
                                    double t[3],double M[3],double jacmS[2][3] ) {
        double t1;
        double t10;
        double t100;
        double t101;
        double t102;
        double t104;
        double t108;
        double t110;
        double t112;
        double t114;
        double t12;
        double t121;
        double t123;
        double t126;
        double t129;
        double t130;
        double t132;
        double t136;
        double t137;
        double t138;
        double t14;
        double t141;
        double t144;
        double t153;
        double t158;
        double t16;
        double t169;
        double t174;
        double t177;
        double t18;
        double t180;
        double t181;
        double t182;
        double t185;
        double t186;
        double t187;
        double t189;
        double t19;
        double t190;
        double t191;
        double t192;
        double t194;
        double t2;
        double t201;
        double t206;
        double t211;
        double t214;
        double t243;
        double t247;
        double t248;
        double t25;
        double t250;
        double t251;
        double t254;
        double t255;
        double t256;
        double t258;
        double t26;
        double t265;
        double t270;
        double t275;
        double t278;
        double t3;
        double t307;
        double t32;
        double t33;
        double t35;
        double t36;
        double t4;
        double t42;
        double t46;
        double t5;
        double t51;
        double t52;
        double t57;
        double t58;
        double t6;
        double t61;
        double t67;
        double t68;
        double t69;
        double t7;
        double t70;
        double t71;
        double t72;
        double t74;
        double t75;
        double t76;
        double t77;
        double t79;
        double t80;
        double t81;
        double t82;
        double t85;
        double t86;
        double t9;
        double t92;
        double t93;
        double t94;
        double t97;
        double t98;
        double t99;
        {
            t1 = a[0];
            t2 = v[0];
            t3 = t2*t2;
            t4 = v[1];
            t5 = t4*t4;
            t6 = v[2];
            t7 = t6*t6;
            t9 = sqrt ( 1.0-t3-t5-t7 );
            t10 = qr0[1];
            t12 = qr0[0];
            t14 = qr0[3];
            t16 = qr0[2];
            t18 = t9*t10+t12*t2+t4*t14-t6*t16;
            t19 = M[0];
            t25 = t9*t16+t12*t4+t10*t6-t2*t14;
            t26 = M[1];
            t32 = t9*t14+t12*t6+t2*t16-t4*t10;
            t33 = M[2];
            t35 = -t18*t19-t25*t26-t32*t33;
            t36 = -t18;
            t42 = t9*t12-t2*t10-t4*t16-t6*t14;
            t46 = t42*t19+t25*t33-t32*t26;
            t51 = t42*t26+t32*t19-t18*t33;
            t52 = -t32;
            t57 = t42*t33+t26*t18-t25*t19;
            t58 = -t25;
            t61 = t35*t36+t42*t46+t51*t52-t57*t58+t[0];
            t67 = t35*t52+t42*t57+t46*t58-t51*t36+t[2];
            t68 = t67*t67;
            t69 = 1/t68;
            t70 = t61*t69;
            t71 = t36*t36;
            t72 = t42*t42;
            t74 = t58*t58;
            t75 = t71+t72+t32*t52-t74;
            t76 = t70*t75;
            t77 = t61*t61;
            t79 = 1/t68/t67;
            t80 = t79*t77;
            t81 = t36*t52;
            t82 = t58*t42;
            t85 = t81+2.0*t82-t32*t36;
            t86 = t80*t85;
            t92 = t35*t58+t51*t42+t57*t36-t46*t52+t[1];
            t93 = t92*t69;
            t94 = t36*t58;
            t97 = t42*t52;
            t98 = 2.0*t94+t42*t32-t97;
            t99 = t93*t98;
            t100 = t92*t92;
            t101 = t100*t79;
            t102 = t101*t85;
            t104 = 2.0*t76-2.0*t86+2.0*t99-2.0*t102;
            t108 = t77*t69+t69*t100;
            t110 = kc[4];
            t112 = kc[1]+t108*t110;
            t114 = kc[0]+t112*t108;
            t121 = t104*t114+t108* ( t112*t104+t108*t104*t110 );
            t123 = 1/t67;
            t126 = 1.0+t108*t114;
            t129 = t126*t61;
            t130 = t69*t85;
            t132 = kc[2];
            t136 = t132*t61;
            t137 = t79*t92;
            t138 = t137*t85;
            t141 = t69*t98;
            t144 = kc[3];
            t153 = a[4];
            t158 = t126*t92;
            t169 = t144*t61;
            t174 = t121*t92*t123+t126*t98*t123-t158*t130+t132* ( 2.0*t76-2.0*t86+6.0*t99
                    -6.0*t102 ) +2.0*t144*t75*t93-4.0*t138*t169+2.0*t141*t169;
            jacmS[0][0] = t1* ( t121*t61*t123+t126*t75*t123-t129*t130+2.0*t132*t75*t93
                                -4.0*t136*t138+2.0*t136*t141+t144* ( 6.0*t76-6.0*t86+2.0*t99-2.0*t102 ) ) +t153*t174
                          ;
            t177 = t1*a[3];
            jacmS[1][0] = t177*t174;
            t180 = t94+2.0*t97-t58*t18;
            t181 = t70*t180;
            t182 = t52*t58;
            t185 = t36*t42;
            t186 = 2.0*t182+t42*t18-t185;
            t187 = t80*t186;
            t189 = t52*t52;
            t190 = t74+t72+t36*t18-t189;
            t191 = t93*t190;
            t192 = t101*t186;
            t194 = 2.0*t181-2.0*t187+2.0*t191-2.0*t192;
            t201 = t114*t194+t108* ( t112*t194+t108*t194*t110 );
            t206 = t69*t186;
            t211 = t137*t186;
            t214 = t69*t190;
            t243 = t201*t92*t123+t126*t190*t123-t158*t206+t132* ( 2.0*t181-2.0*t187+6.0*
                    t191-6.0*t192 ) +2.0*t144*t180*t93-4.0*t211*t169+2.0*t169*t214;
            jacmS[0][1] = t1* ( t201*t61*t123+t126*t180*t123-t129*t206+2.0*t132*t180*t93
                                -4.0*t136*t211+2.0*t136*t214+t144* ( 6.0*t181-6.0*t187+2.0*t191-2.0*t192 ) ) +t153*
                          t243;
            jacmS[1][1] = t177*t243;
            t247 = 2.0*t81+t42*t25-t82;
            t248 = t70*t247;
            t250 = t189+t72+t25*t58-t71;
            t251 = t80*t250;
            t254 = t182+2.0*t185-t25*t52;
            t255 = t93*t254;
            t256 = t101*t250;
            t258 = 2.0*t248-2.0*t251+2.0*t255-2.0*t256;
            t265 = t258*t114+t108* ( t258*t112+t108*t258*t110 );
            t270 = t250*t69;
            t275 = t137*t250;
            t278 = t254*t69;
            t307 = t265*t92*t123+t126*t254*t123-t270*t158+t132* ( 2.0*t248-2.0*t251+6.0*
                    t255-6.0*t256 ) +2.0*t144*t247*t93-4.0*t169*t275+2.0*t169*t278;
            jacmS[0][2] = t1* ( t265*t61*t123+t126*t247*t123-t129*t270+2.0*t132*t247*t93
                                -4.0*t136*t275+2.0*t136*t278+t144* ( 6.0*t248-6.0*t251+2.0*t255-2.0*t256 ) ) +t153*
                          t307;
            jacmS[1][2] = t307*t177;
            return;
        }
    }


    void Sba::quatMultFast ( double q1[4], double q2[4], double p[4] ) {
        double t1, t2, t3, t4, t5, t6, t7, t8, t9;
//double t10, t11, t12;

        t1= ( q1[0]+q1[1] ) * ( q2[0]+q2[1] );
        t2= ( q1[3]-q1[2] ) * ( q2[2]-q2[3] );
        t3= ( q1[1]-q1[0] ) * ( q2[2]+q2[3] );
        t4= ( q1[2]+q1[3] ) * ( q2[1]-q2[0] );
        t5= ( q1[1]+q1[3] ) * ( q2[1]+q2[2] );
        t6= ( q1[1]-q1[3] ) * ( q2[1]-q2[2] );
        t7= ( q1[0]+q1[2] ) * ( q2[0]-q2[3] );
        t8= ( q1[0]-q1[2] ) * ( q2[0]+q2[3] );
 

        /* following fragment it equivalent to the one above */
        t9=0.5* ( t5-t6+t7+t8 );
        p[0]= t2 + t9-t5;
        p[1]= t1 - t9-t6;
        p[2]=-t3 + t9-t8;
        p[3]=-t4 + t9-t7;
    }

}
