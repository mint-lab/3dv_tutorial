/*
/////////////////////////////////////////////////////////////////////////////////
//// 
////  test_cvsba.cpp
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

#include "cvsba.h"
#include "readparams.h"


void readDataCVFormat(char* camfile, char* ptsfile, std::vector<cv::Point3f>& points, std::vector<std::vector<cv::Point2f> >& imagePoints, 
		std::vector<std::vector<int> >& visibility, std::vector<cv::Mat>& cameraMatrix, std::vector<cv::Mat>& distCoeffs,
		std::vector<cv::Mat>& R, std::vector<cv::Mat>& T);


/**
 * test opencv wrapper for sba. Receive camera data and point data files
 * e.g. "test_cvsba 54camsvarKD.txt 54pts.txt" using sba sample data files
 */
int main(int argc, char** argv) {
  if(argc<3) {
    std::cerr << "Usage:   inputcamsfile inputpointsfile" << std::endl;
    return 0;
  }
  
  std::vector<cv::Point3f> points;
  std::vector<std::vector<cv::Point2f> >  imagePoints;
  std::vector<std::vector<int> > visibility;
  std::vector<cv::Mat> cameraMatrix;
  std::vector<cv::Mat> R;
  std::vector<cv::Mat> T;
  std::vector<cv::Mat> distCoeffs;
  cv::TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 150, 1e-10);
  
  // read data from sba file format. All obtained data is in opencv format (including rotation in rodrigues format)
  readDataCVFormat(argv[1], argv[2], points, imagePoints, visibility, cameraMatrix, distCoeffs, R, T);

  // run sba optimization
  cvsba::Sba sba;
  
  // change params if desired
  cvsba::Sba::Params params ;
  params.type = cvsba::Sba::MOTIONSTRUCTURE;
  params.iterations = 150;
  params.minError = 1e-10;
  params.fixedIntrinsics = 5;
  params.fixedDistortion = 5;
  params.verbose=true;
  sba.setParams(params);
  
  
  sba.run( points, imagePoints, visibility, cameraMatrix,  R, T,distCoeffs);
  
  std::cout<<"Optimization. Initial error="<<sba.getInitialReprjError()<<" and Final error="<<sba.getFinalReprjError()<<std::endl;
  return 0;
}


/**
 * auxiliar function from eucsbademo to read file parameters
 */
void quat2vecRead(double *inp, int nin, double *outp, int nout)
{
double mag, sg;
register int i;

  /* intrinsics & distortion */
  if(nin>7) // are they present?
    for(i=0; i<nin-7; ++i)
      outp[i]=inp[i];
  else
    i=0;

  /* rotation */
  /* normalize and ensure that the quaternion's scalar component is non-negative;
   * if not, negate the quaternion since two quaternions q and -q represent the
   * same rotation
   */
  mag=sqrt(inp[i]*inp[i] + inp[i+1]*inp[i+1] + inp[i+2]*inp[i+2] + inp[i+3]*inp[i+3]);
  sg=(inp[i]>=0.0)? 1.0 : -1.0;
  mag=sg/mag;
  outp[i]  =inp[i+1]*mag;
  outp[i+1]=inp[i+2]*mag;
  outp[i+2]=inp[i+3]*mag;
  i+=3;

  /* translation*/
  for( ; i<nout; ++i)
    outp[i]=inp[i+1];
}



/**
 * 
 */
void readDataCVFormat(char* camfile, char* ptsfile, std::vector< cv::Point3f >& points, std::vector< std::vector< cv::Point2f > >& imagePoints, std::vector< std::vector< int > >& visibility, std::vector< cv::Mat >& cameraMatrix, std::vector< cv::Mat >& distCoeffs, std::vector< cv::Mat >& R, std::vector< cv::Mat >& T)
{
  // read data from file to test
  int nframes, numpts3D, numprojs;
  double *motstruct, *initrot, *imgpts, *covimgpts;
  char *vmask;
  int cnp=16, pnp=3, mnp=2;
  readInitialSBAEstimate(camfile, ptsfile, cnp, pnp, mnp, quat2vecRead, cnp+1, //NULL, 0, 
			  &nframes, &numpts3D, &numprojs, &motstruct, &initrot, &imgpts, &covimgpts, &vmask);  
  
  //motstruct
  // fu, u0, v0, ar, s   kc(1:5)   quaternion(x3) translation(x3) 
  
//   visibility.resize(nframes);
//   for(int i=0; i<nframes; i++) {
//     visibility[i].resize(numpts3D);
//     for(int j=0; j<numpts3D; j++) visibility[i][j] = vmask[i*numpts3D+j];
//   }
  
  visibility.resize(nframes);
   for(int i=0; i<nframes; i++) 
     visibility[i].resize(numpts3D);

   for(int i=0; i<numpts3D; i++) 
    for(int j=0; j<nframes; j++) 
      visibility[j][i] = vmask[i*nframes+j]; 
  
  imagePoints.resize(nframes);
  for(int i=0; i<nframes; i++) {
    imagePoints[i].resize(numpts3D);
    for(int j=0; j<numpts3D; j++)  
      imagePoints[i][j].x=imagePoints[i][j].y=std::numeric_limits<float>::quiet_NaN();
  }
    
  for(int i=0, idx=0; i<numpts3D; i++) {
    for(int j=0; j<nframes; j++) {
      if(visibility[j][i]) {
	imagePoints[j][i].x=imgpts[idx];
	imagePoints[j][i].y=imgpts[idx+1];
	idx+=2;
      }
    }
  }    
    
//   imagePoints.resize(nframes);
//   for(int i=0, idx=0; i<nframes; i++) {
//     for(int j=0; j<numpts3D; j++) {
//       if(visibility[i][j]) {
// 	cv::Point2f auxPt;
// 	auxPt.x = imgpts[idx];
// 	auxPt.y = imgpts[idx+1];
// 	imagePoints[i].push_back(auxPt);
// 	idx+=2;
//       }
//     }
//   }
  
  // Rs have to be converted from quaternions
  R.resize(nframes);
  for(int i=0; i<nframes; i++) {
    cv::Mat aux = cv::Mat(4,1,CV_64FC1,cv::Scalar::all(0));
    for(int j=0; j<4; j++) aux.ptr<double>(0)[j] = initrot[i*4+j];
    cvsba::Sba::quat2rod(aux,R[i]);
    //std::cout<<"Read "<<aux<<" "<<R[i]<<std::endl;
  }
  
  T.resize(nframes);
  cameraMatrix.resize(nframes);
  distCoeffs.resize(nframes);
  for(int i=0; i<nframes; i++) {

    T[i] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
    for(int j=0; j<3; j++) T[i].ptr<double>(0)[j] = motstruct[i*16+13+j];
    cameraMatrix[i] = cv::Mat(3,3,CV_64FC1,cv::Scalar::all(0));
    cameraMatrix[i].ptr<double>(0)[8] = 1.;
    cameraMatrix[i].ptr<double>(0)[0] = motstruct[i*16+0]; //fx
    cameraMatrix[i].ptr<double>(0)[2] = motstruct[i*16+1]; //cx
    cameraMatrix[i].ptr<double>(0)[4] = motstruct[i*16+0]*motstruct[i*16+3]; //fy
    cameraMatrix[i].ptr<double>(0)[5] = motstruct[i*16+2]; //cy
    distCoeffs[i] = cv::Mat::zeros(5,1,CV_64FC1);
    for(int j=0; j<5; j++) distCoeffs[i].ptr<double>(0)[j] = motstruct[i*16+5+j];
  }
  
  points.resize(numpts3D);
  for(int i=0; i<numpts3D; i++) {
    points[i].x = motstruct[nframes*16 + 3*i + 0];
    points[i].y = motstruct[nframes*16 + 3*i + 1];
    points[i].z = motstruct[nframes*16 + 3*i + 2];
  }
  
  
}

