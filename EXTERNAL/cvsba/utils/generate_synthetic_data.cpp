/*
/////////////////////////////////////////////////////////////////////////////////
//// 
////  generate_synthethic_data.cpp
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
#include <cstdio>
#include <opencv2/calib3d/calib3d.hpp>

// generate artificial noisy data for 2 cameras to test sba optimization
void generateSyntheticData(std::vector<cv::Point3d>& points, std::vector<std::vector<cv::Point2d> >& imagePoints, 
		std::vector<std::vector<int> >& visibility, std::vector<cv::Mat>& cameraMatrix,
		std::vector<cv::Mat>& distCoeffs, std::vector<cv::Mat>& R, std::vector<cv::Mat>& T );

// save output to sba files format (first file is cameras info, second file is points info)
void saveToFile(char* file1, char* file2, std::vector<cv::Point3d>& points, std::vector<std::vector<cv::Point2d> >& imagePoints, 
		std::vector<std::vector<int> >& visibility, std::vector<cv::Mat>& cameraMatrix,
		std::vector<cv::Mat>& distCoeffs, std::vector<cv::Mat>& R, std::vector<cv::Mat>& T );



/**
 * Generate artificial noisy files to test sba optimization
 * Output files can be used as input for test_cvsba program
 */
int main(int argc, char** argv) {
  if(argc<3) {
    std::cerr << "Use: generate_synthetic_data outcamsfile outpointsfile" << std::endl;
    return 0;
  }
  
  std::vector<cv::Point3d> points;
  std::vector<std::vector<cv::Point2d> >  imagePoints;
  std::vector<std::vector<int> > visibility;
  std::vector<cv::Mat> cameraMatrix;
  std::vector<cv::Mat> R;
  std::vector<cv::Mat> T;
  std::vector<cv::Mat> distCoeffs;
  
  generateSyntheticData(points, imagePoints, visibility, cameraMatrix, distCoeffs, R, T)  ;
  saveToFile(argv[1], argv[2], points, imagePoints, visibility, cameraMatrix, distCoeffs, R, T); 
  
  return 0;
}


/**
 * 
 */
void generateSyntheticData(std::vector< cv::Point3d >& points, std::vector< std::vector< cv::Point2d > >& imagePoints, std::vector< std::vector< int > >& visibility, std::vector< cv::Mat >& cameraMatrix, std::vector< cv::Mat >& distCoeffs, std::vector< cv::Mat >& R, std::vector< cv::Mat >& T)
{
  int npoints = 240;
  int ncams = 15;
  
  // cam intrinsics
  cv::Mat synCamMat, synDist;
  synCamMat = cv::Mat(3,3,CV_64FC1, cv::Scalar::all(0));
  synCamMat.ptr<double>(0)[0]=1000;
  synCamMat.ptr<double>(0)[2]=1200/2;
  synCamMat.ptr<double>(0)[4]=1000;
  synCamMat.ptr<double>(0)[5]=800/2;
  synCamMat.ptr<double>(0)[8]=1;
  synDist = cv::Mat(5,1,CV_64FC1, cv::Scalar::all(0));
  for(int i=0; i<ncams; i++) {
    cameraMatrix.push_back(synCamMat.clone());
    distCoeffs.push_back(synDist.clone());
  }
  
  // 3d points (random points)
  srand(time(NULL));
  for(int i=0; i<npoints; i++) {
    cv::Point3d p;
    p.x = (rand()%2000-1000)/1000.;
    p.y = (rand()%2000-1000)/1000.;
    p.z = (rand()%2000-1000)/1000.;
    points.push_back(p);
  }  
  
  // R and T
  cv::Mat synT = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
  synT.ptr<double>(0)[2] = 3.;
  for(int i=0; i<ncams; i++)
    T.push_back(synT.clone());
  
  cv::Mat synR = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
  synR.ptr<double>(0)[0] = 0.;
  synR.ptr<double>(0)[1] = 0.;
  double offsetangle = 2*3.14159265359/double(ncams);
  for(int i=0; i<ncams; i++) {
    synR.ptr<double>(0)[2] = double(i)*offsetangle;
    R.push_back(synR.clone());    
  }  
  
  // 2d points
  imagePoints.resize(ncams);
  for(int i=0; i<ncams; i++)
    cv::projectPoints(points,R[i],T[i],synCamMat,synDist,imagePoints[i]);
  
  visibility.resize(ncams);
  for(int i=0; i<ncams; i++) 
    for(int j=0; j<npoints; j++) 
      visibility[i].push_back(1);  
  
    
  // add some noise
 for(int i=0; i<imagePoints.size(); i++) {
    for(int j=0; j<imagePoints[i].size(); j++) {
     imagePoints[i][j].x += (rand()%200-100)/10000.;
     imagePoints[i][j].y += (rand()%200-100)/10000.;
    }  
  }  
  // noise for structure (comment for testing just MOTION sba)
  for(int j=0; j<points.size(); j++) {
    points[j].x += (rand()%200-100)/10000.;
    points[j].y += (rand()%200-100)/10000.;
    points[j].z += (rand()%200-100)/10000.; 
  }
  // noise for motion (comment for testing just STRUCTURE sba)
  for(int j=0; j<ncams; j++) {
    for(int i=0; i<3; i++) {
      R[j].ptr<double>(0)[i] += (rand()%200-100)/1000.;
      T[j].ptr<double>(0)[i] += (rand()%200-100)/10000.;
    }
  }
  
  
}


/**
 * 
 */
void saveToFile(char* file1, char* file2, std::vector<cv::Point3d>& points, std::vector<std::vector<cv::Point2d> >& imagePoints, 
		std::vector<std::vector<int> >& visibility, std::vector<cv::Mat>& cameraMatrix,
		std::vector<cv::Mat>& distCoeffs, std::vector<cv::Mat>& R, std::vector<cv::Mat>& T ) {
  
  int ncams = cameraMatrix.size();
  
  // cams file
  FILE* f1 = fopen(file1, "w");
  for(int i=0; i<ncams; i++) {
    double fx, cx, cy, rel, s;
    double d[5];
    double r[4];
    double t[3];
    fx = cameraMatrix[i].ptr<double>(0)[0];
    cx = cameraMatrix[i].ptr<double>(0)[2];
    cy = cameraMatrix[i].ptr<double>(0)[5];
    rel = cameraMatrix[i].ptr<double>(0)[4]/cameraMatrix[i].ptr<double>(0)[0];
    s = 0;
    for(int j=0; j<5; j++) d[j] = distCoeffs[i].ptr<double>(0)[j];
    cv::Mat quat;
    cvsba::Sba::rod2quat(R[i], quat);
    for(int j=0; j<4; j++) r[j] = quat.ptr<double>(0)[j];
    for(int j=0; j<3; j++) t[j] = T[i].ptr<double>(0)[j];
    fprintf(f1, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", fx, cx, cy, rel, s, d[0], d[1], d[2], d[3], d[4], r[0], r[1], r[2], r[3], t[0], t[1], t[2]);
  }
  fclose(f1);
  
  // points file
  FILE* f2 = fopen(file2, "w");
  std::vector<int> idxs;
  idxs.resize(ncams);
  for(int i=0; i<ncams; i++) idxs[i] = 0;
  for(int i=0; i<points.size(); i++) {
    int ntimes = 0;
    for(int j=0; j<ncams; j++) if(visibility[j][i]) ntimes++;
    fprintf(f2, "%f %f %f %d", points[i].x, points[i].y, points[i].z, ntimes);
    for(int j=0; j<ncams; j++) {
      if(visibility[j][i]) {
	double x,y;
	x = imagePoints[j][idxs[j]].x;
	y = imagePoints[j][idxs[j]].y;
	idxs[j]++;
	fprintf(f2, " %d %f %f", j, x, y);
      }
    }
    fprintf(f2, "\n");
  }
  fclose(f2);
}