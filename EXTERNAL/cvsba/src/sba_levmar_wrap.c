/////////////////////////////////////////////////////////////////////////////////
//// 
////  Simple drivers for sparse bundle adjustment based on the
////  Levenberg - Marquardt minimization algorithm
////  This file provides simple wrappers to the functions defined in sba_levmar.c
////  Copyright (C) 2004-2009 Manolis Lourakis (lourakis at ics forth gr)
////  Institute of Computer Science, Foundation for Research & Technology - Hellas
////  Heraklion, Crete, Greece.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "sba.h"


#define FABS(x)           (((x)>=0)? (x) : -(x))

struct wrap_motstr_data_ {
  void   (*proj)(int j, int i, double *aj, double *bi, double *xij, void *adata); // Q
  void (*projac)(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata); // dQ/da, dQ/db
  int cnp, pnp, mnp; /* parameter numbers */
  void *adata;
};

struct wrap_mot_data_ {
  void   (*proj)(int j, int i, double *aj, double *xij, void *adata); // Q
  void (*projac)(int j, int i, double *aj, double *Aij, void *adata); // dQ/da
  int cnp, mnp; /* parameter numbers */
  void *adata;
};

struct wrap_str_data_ {
  void   (*proj)(int j, int i, double *bi, double *xij, void *adata); // Q
  void (*projac)(int j, int i, double *bi, double *Bij, void *adata); // dQ/db
  int pnp, mnp; /* parameter numbers */
  void *adata;
};

/* Routines to estimate the estimated measurement vector (i.e. "func") and
 * its sparse jacobian (i.e. "fjac") needed by BA expert drivers. Code below
 * makes use of user-supplied functions computing "Q", "dQ/da", d"Q/db",
 * i.e. predicted projection and associated jacobians for a SINGLE image measurement.
 * Notice also that what follows is two pairs of "func" and corresponding "fjac" routines.
 * The first is to be used in full (i.e. motion + structure) BA, the second in 
 * motion only BA.
 */

/* FULL BUNDLE ADJUSTMENT */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Caller supplies rcidxs and rcsubs which can be used as working memory.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void sba_motstr_Qs(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp; 
  double *pa, *pb, *paj, *pbi, *pxij;
  int n, m, nnz;
  struct wrap_motstr_data_ *wdata;
  void (*proj)(int j, int i, double *aj, double *bi, double *xij, void *proj_adata);
  void *proj_adata;

  wdata=(struct wrap_motstr_data_ *)adata;
  cnp=wdata->cnp; pnp=wdata->pnp; mnp=wdata->mnp;
  proj=wdata->proj;
  proj_adata=wdata->adata;

  n=idxij->nr; m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    paj=pa+j*cnp;

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      pbi=pb + rcsubs[i]*pnp;
      pxij=hx + idxij->val[rcidxs[i]]*mnp; // set pxij to point to hx_ij

      (*proj)(j, rcsubs[i], paj, pbi, pxij, proj_adata); // evaluate Q in pxij
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, B_11, ..., A_1m, B_1m, ..., A_n1, B_n1, ..., A_nm, B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Caller supplies rcidxs and rcsubs which can be used as working memory.
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void sba_motstr_Qs_jac(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *paj, *pbi, *pAij, *pBij;
  int n, m, nnz, Asz, Bsz, ABsz, idx;
  struct wrap_motstr_data_ *wdata;
  void (*projac)(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *projac_adata);
  void *projac_adata;

  
  wdata=(struct wrap_motstr_data_ *)adata;
  cnp=wdata->cnp; pnp=wdata->pnp; mnp=wdata->mnp;
  projac=wdata->projac;
  projac_adata=wdata->adata;

  n=idxij->nr; m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    paj=pa+j*cnp;

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      pbi=pb + rcsubs[i]*pnp;
      idx=idxij->val[rcidxs[i]];
      pAij=jac  + idx*ABsz; // set pAij to point to A_ij
      pBij=pAij + Asz; // set pBij to point to B_ij

      (*projac)(j, rcsubs[i], paj, pbi, pAij, pBij, projac_adata); // evaluate dQ/da, dQ/db in pAij, pBij
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is approximated with the aid of finite differences and is returned in the order
 * (A_11, B_11, ..., A_1m, B_1m, ..., A_n1, B_n1, ..., A_nm, B_nm),
 * where A_ij=dx_ij/da_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 * Problem-specific information is assumed to be stored in a structure pointed to by "dat".
 *
 * NOTE: This function is provided mainly for illustration purposes; in case that execution time is a concern,
 * the jacobian should be computed analytically
 */
static void sba_motstr_Qs_fdjac(
    double *p,                /* I: current parameter estimate, (m*cnp+n*pnp)x1 */
    struct sba_crsm *idxij,   /* I: sparse matrix containing the location of x_ij in hx */
    int    *rcidxs,           /* work array for the indexes of nonzero elements of a single sparse matrix row/column */
    int    *rcsubs,           /* work array for the subscripts of nonzero elements in a single sparse matrix row/column */
    double *jac,              /* O: array for storing the approximated jacobian */
    void   *dat)              /* I: points to a "wrap_motstr_data_" structure */
{
  register int i, j, ii, jj;
  double *pa, *pb, *paj, *pbi;
  register double *pAB;
  int n, m, nnz, Asz, Bsz, ABsz;

  double tmp;
  register double d, d1;

  struct wrap_motstr_data_ *fdjd;
  void (*proj)(int j, int i, double *aj, double *bi, double *xij, void *adata);
  double *hxij, *hxxij;
  int cnp, pnp, mnp;
  void *adata;

  /* retrieve problem-specific information passed in *dat */
  fdjd=(struct wrap_motstr_data_ *)dat;
  proj=fdjd->proj;
  cnp=fdjd->cnp; pnp=fdjd->pnp; mnp=fdjd->mnp;
  adata=fdjd->adata;

  n=idxij->nr; m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  /* allocate memory for hxij, hxxij */
  if((hxij=malloc(2*mnp*sizeof(double)))==NULL){
    fprintf(stderr, "memory allocation request failed in sba_motstr_Qs_fdjac()!\n");
    exit(1);
  }
  hxxij=hxij+mnp;

    /* compute A_ij */
    for(j=0; j<m; ++j){
      paj=pa+j*cnp; // j-th camera parameters

      nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero A_ij, i=0...n-1 */
      for(jj=0; jj<cnp; ++jj){
        /* determine d=max(SBA_DELTA_SCALE*|paj[jj]|, SBA_MIN_DELTA), see HZ */
        d=(double)(SBA_DELTA_SCALE)*paj[jj]; // force evaluation
        d=FABS(d);
        if(d<SBA_MIN_DELTA) d=SBA_MIN_DELTA;
        d1=1.0/d; /* invert so that divisions can be carried out faster as multiplications */

        for(i=0; i<nnz; ++i){
          pbi=pb + rcsubs[i]*pnp; // i-th point parameters
          (*proj)(j, rcsubs[i], paj, pbi, hxij, adata); // evaluate supplied function on current solution

          tmp=paj[jj];
          paj[jj]+=d;
          (*proj)(j, rcsubs[i], paj, pbi, hxxij, adata);
          paj[jj]=tmp; /* restore */

          pAB=jac + idxij->val[rcidxs[i]]*ABsz; // set pAB to point to A_ij
          for(ii=0; ii<mnp; ++ii)
            pAB[ii*cnp+jj]=(hxxij[ii]-hxij[ii])*d1;
        }
      }
    }

    /* compute B_ij */
    for(i=0; i<n; ++i){
      pbi=pb+i*pnp; // i-th point parameters

      nnz=sba_crsm_row_elmidxs(idxij, i, rcidxs, rcsubs); /* find nonzero B_ij, j=0...m-1 */
      for(jj=0; jj<pnp; ++jj){
        /* determine d=max(SBA_DELTA_SCALE*|pbi[jj]|, SBA_MIN_DELTA), see HZ */
        d=(double)(SBA_DELTA_SCALE)*pbi[jj]; // force evaluation
        d=FABS(d);
        if(d<SBA_MIN_DELTA) d=SBA_MIN_DELTA;
        d1=1.0/d; /* invert so that divisions can be carried out faster as multiplications */

        for(j=0; j<nnz; ++j){
          paj=pa + rcsubs[j]*cnp; // j-th camera parameters
          (*proj)(rcsubs[j], i, paj, pbi, hxij, adata); // evaluate supplied function on current solution

          tmp=pbi[jj];
          pbi[jj]+=d;
          (*proj)(rcsubs[j], i, paj, pbi, hxxij, adata);
          pbi[jj]=tmp; /* restore */

          pAB=jac + idxij->val[rcidxs[j]]*ABsz + Asz; // set pAB to point to B_ij
          for(ii=0; ii<mnp; ++ii)
            pAB[ii*pnp+jj]=(hxxij[ii]-hxij[ii])*d1;
        }
      }
    }

  free(hxij);
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Caller supplies rcidxs and rcsubs which can be used as working memory.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void sba_mot_Qs(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, mnp; 
  double *paj, *pxij;
  //int n;
  int m, nnz;
  struct wrap_mot_data_ *wdata;
  void (*proj)(int j, int i, double *aj, double *xij, void *proj_adata);
  void *proj_adata;

  wdata=(struct wrap_mot_data_ *)adata;
  cnp=wdata->cnp; mnp=wdata->mnp;
  proj=wdata->proj;
  proj_adata=wdata->adata;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    paj=p+j*cnp;

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      pxij=hx + idxij->val[rcidxs[i]]*mnp; // set pxij to point to hx_ij

      (*proj)(j, rcsubs[i], paj, pxij, proj_adata); // evaluate Q in pxij
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Caller supplies rcidxs and rcsubs which can be used as working memory.
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void sba_mot_Qs_jac(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, mnp;
  double *paj, *pAij;
  //int n;
  int m, nnz, Asz, idx;
  struct wrap_mot_data_ *wdata;
  void (*projac)(int j, int i, double *aj, double *Aij, void *projac_adata);
  void *projac_adata;

  wdata=(struct wrap_mot_data_ *)adata;
  cnp=wdata->cnp; mnp=wdata->mnp;
  projac=wdata->projac;
  projac_adata=wdata->adata;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    paj=p+j*cnp;

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      idx=idxij->val[rcidxs[i]];
      pAij=jac + idx*Asz; // set pAij to point to A_ij

      (*projac)(j, rcsubs[i], paj, pAij, projac_adata); // evaluate dQ/da in pAij
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac the jacobian
 * of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is approximated with the aid of finite differences and is returned in the order
 * (A_11, ..., A_1m, ..., A_n1, ..., A_nm), where A_ij=dx_ij/da_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 * Problem-specific information is assumed to be stored in a structure pointed to by "dat".
 *
 * NOTE: This function is provided mainly for illustration purposes; in case that execution time is a concern,
 * the jacobian should be computed analytically
 */
static void sba_mot_Qs_fdjac(
    double *p,                /* I: current parameter estimate, (m*cnp)x1 */
    struct sba_crsm *idxij,   /* I: sparse matrix containing the location of x_ij in hx */
    int    *rcidxs,           /* work array for the indexes of nonzero elements of a single sparse matrix row/column */
    int    *rcsubs,           /* work array for the subscripts of nonzero elements in a single sparse matrix row/column */
    double *jac,              /* O: array for storing the approximated jacobian */
    void   *dat)              /* I: points to a "wrap_mot_data_" structure */
{
  register int i, j, ii, jj;
  double *paj;
  register double *pA;
  //int n; 
  int m, nnz, Asz;

  double tmp;
  register double d, d1;

  struct wrap_mot_data_ *fdjd;
  void (*proj)(int j, int i, double *aj, double *xij, void *adata);
  double *hxij, *hxxij;
  int cnp, mnp;
  void *adata;

  /* retrieve problem-specific information passed in *dat */
  fdjd=(struct wrap_mot_data_ *)dat;
  proj=fdjd->proj;
  cnp=fdjd->cnp; mnp=fdjd->mnp;
  adata=fdjd->adata;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  /* allocate memory for hxij, hxxij */
  if((hxij=malloc(2*mnp*sizeof(double)))==NULL){
    fprintf(stderr, "memory allocation request failed in sba_mot_Qs_fdjac()!\n");
    exit(1);
  }
  hxxij=hxij+mnp;

  /* compute A_ij */
  for(j=0; j<m; ++j){
    paj=p+j*cnp; // j-th camera parameters

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero A_ij, i=0...n-1 */
    for(jj=0; jj<cnp; ++jj){
      /* determine d=max(SBA_DELTA_SCALE*|paj[jj]|, SBA_MIN_DELTA), see HZ */
      d=(double)(SBA_DELTA_SCALE)*paj[jj]; // force evaluation
      d=FABS(d);
      if(d<SBA_MIN_DELTA) d=SBA_MIN_DELTA;
      d1=1.0/d; /* invert so that divisions can be carried out faster as multiplications */

      for(i=0; i<nnz; ++i){
        (*proj)(j, rcsubs[i], paj, hxij, adata); // evaluate supplied function on current solution

        tmp=paj[jj];
        paj[jj]+=d;
        (*proj)(j, rcsubs[i], paj, hxxij, adata);
        paj[jj]=tmp; /* restore */

        pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij
        for(ii=0; ii<mnp; ++ii)
          pA[ii*cnp+jj]=(hxxij[ii]-hxij[ii])*d1;
      }
    }
  }

  free(hxij);
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Caller supplies rcidxs and rcsubs which can be used as working memory.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void sba_str_Qs(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int pnp, mnp; 
  double *pbi, *pxij;
  //int n;
  int m, nnz;
  struct wrap_str_data_ *wdata;
  void (*proj)(int j, int i, double *bi, double *xij, void *proj_adata);
  void *proj_adata;

  wdata=(struct wrap_str_data_ *)adata;
  pnp=wdata->pnp; mnp=wdata->mnp;
  proj=wdata->proj;
  proj_adata=wdata->adata;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      pbi=p + rcsubs[i]*pnp;
      pxij=hx + idxij->val[rcidxs[i]]*mnp; // set pxij to point to hx_ij

      (*proj)(j, rcsubs[i], pbi, pxij, proj_adata); // evaluate Q in pxij
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm), where B_ij=dx_ij/db_i (see HZ).
 * Caller supplies rcidxs and rcsubs which can be used as working memory.
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void sba_str_Qs_jac(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int pnp, mnp;
  double *pbi, *pBij;
  //int n;
  int m, nnz, Bsz, idx;
  struct wrap_str_data_ *wdata;
  void (*projac)(int j, int i, double *bi, double *Bij, void *projac_adata);
  void *projac_adata;

  
  wdata=(struct wrap_str_data_ *)adata;
  pnp=wdata->pnp; mnp=wdata->mnp;
  projac=wdata->projac;
  projac_adata=wdata->adata;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      pbi=p + rcsubs[i]*pnp;
      idx=idxij->val[rcidxs[i]];
      pBij=jac + idx*Bsz; // set pBij to point to B_ij

      (*projac)(j, rcsubs[i], pbi, pBij, projac_adata); // evaluate dQ/db in pBij
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is approximated with the aid of finite differences and is returned in the order
 * (B_11, ..., B_1m, ..., B_n1, ..., B_nm), where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 * Problem-specific information is assumed to be stored in a structure pointed to by "dat".
 *
 * NOTE: This function is provided mainly for illustration purposes; in case that execution time is a concern,
 * the jacobian should be computed analytically
 */
static void sba_str_Qs_fdjac(
    double *p,                /* I: current parameter estimate, (n*pnp)x1 */
    struct sba_crsm *idxij,   /* I: sparse matrix containing the location of x_ij in hx */
    int    *rcidxs,           /* work array for the indexes of nonzero elements of a single sparse matrix row/column */
    int    *rcsubs,           /* work array for the subscripts of nonzero elements in a single sparse matrix row/column */
    double *jac,              /* O: array for storing the approximated jacobian */
    void   *dat)              /* I: points to a "wrap_str_data_" structure */
{
  register int i, j, ii, jj;
  double *pbi;
  register double *pB;
  //int m;
  int n, nnz, Bsz;

  double tmp;
  register double d, d1;

  struct wrap_str_data_ *fdjd;
  void (*proj)(int j, int i, double *bi, double *xij, void *adata);
  double *hxij, *hxxij;
  int pnp, mnp;
  void *adata;

  /* retrieve problem-specific information passed in *dat */
  fdjd=(struct wrap_str_data_ *)dat;
  proj=fdjd->proj;
  pnp=fdjd->pnp; mnp=fdjd->mnp;
  adata=fdjd->adata;

  n=idxij->nr;
  //m=idxij->nc;
  Bsz=mnp*pnp;

  /* allocate memory for hxij, hxxij */
  if((hxij=malloc(2*mnp*sizeof(double)))==NULL){
    fprintf(stderr, "memory allocation request failed in sba_str_Qs_fdjac()!\n");
    exit(1);
  }
  hxxij=hxij+mnp;

  /* compute B_ij */
  for(i=0; i<n; ++i){
    pbi=p+i*pnp; // i-th point parameters

    nnz=sba_crsm_row_elmidxs(idxij, i, rcidxs, rcsubs); /* find nonzero B_ij, j=0...m-1 */
    for(jj=0; jj<pnp; ++jj){
      /* determine d=max(SBA_DELTA_SCALE*|pbi[jj]|, SBA_MIN_DELTA), see HZ */
      d=(double)(SBA_DELTA_SCALE)*pbi[jj]; // force evaluation
      d=FABS(d);
      if(d<SBA_MIN_DELTA) d=SBA_MIN_DELTA;
      d1=1.0/d; /* invert so that divisions can be carried out faster as multiplications */

      for(j=0; j<nnz; ++j){
        (*proj)(rcsubs[j], i, pbi, hxij, adata); // evaluate supplied function on current solution

        tmp=pbi[jj];
        pbi[jj]+=d;
        (*proj)(rcsubs[j], i, pbi, hxxij, adata);
        pbi[jj]=tmp; /* restore */

        pB=jac + idxij->val[rcidxs[j]]*Bsz; // set pB to point to B_ij
        for(ii=0; ii<mnp; ++ii)
          pB[ii*pnp+jj]=(hxxij[ii]-hxij[ii])*d1;
      }
    }
  }

  free(hxij);
}


/* 
 * Simple driver to sba_motstr_levmar_x for bundle adjustment on camera and structure parameters.
 *
 * Returns the number of iterations (>=0) if successfull, SBA_ERROR if failed
 */

int sba_motstr_levmar(
    const int n,   /* number of points */
    const int ncon,/* number of points (starting from the 1st) whose parameters should not be modified.
                    * All B_ij (see below) with i<ncon are assumed to be zero
                    */
    const int m,   /* number of images */
    const int mcon,/* number of images (starting from the 1st) whose parameters should not be modified.
					          * All A_ij (see below) with j<mcon are assumed to be zero
					          */
    char *vmask,  /* visibility mask: vmask[i, j]=1 if point i visible in image j, 0 otherwise. nxm */
    double *p,    /* initial parameter vector p0: (a1, ..., am, b1, ..., bn).
                   * aj are the image j parameters, bi are the i-th point parameters,
                   * size m*cnp + n*pnp
                   */
    const int cnp,/* number of parameters for ONE camera; e.g. 6 for Euclidean cameras */
    const int pnp,/* number of parameters for ONE point; e.g. 3 for Euclidean points */
    double *x,    /* measurements vector: (x_11^T, .. x_1m^T, ..., x_n1^T, .. x_nm^T)^T where
                   * x_ij is the projection of the i-th point on the j-th image.
                   * NOTE: some of the x_ij might be missing, if point i is not visible in image j;
                   * see vmask[i, j], max. size n*m*mnp
                   */
    double *covx, /* measurements covariance matrices: (Sigma_x_11, .. Sigma_x_1m, ..., Sigma_x_n1, .. Sigma_x_nm),
                   * where Sigma_x_ij is the mnp x mnp covariance of x_ij stored row-by-row. Set to NULL if no
                   * covariance estimates are available (identity matrices are implicitly used in this case).
                   * NOTE: a certain Sigma_x_ij is missing if the corresponding x_ij is also missing;
                   * see vmask[i, j], max. size n*m*mnp*mnp
                   */
    const int mnp,/* number of parameters for EACH measurement; usually 2 */
    void (*proj)(int j, int i, double *aj, double *bi, double *xij, void *adata),
                                              /* functional relation computing a SINGLE image measurement. Assuming that
                                               * the parameters of point i are bi and the parameters of camera j aj,
                                               * computes a prediction of \hat{x}_{ij}. aj is cnp x 1, bi is pnp x 1 and
                                               * xij is mnp x 1. This function is called only if point i is visible in
                                               * image j (i.e. vmask[i, j]==1)
                                               */
    void (*projac)(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata),
                                              /* functional relation to evaluate d x_ij / d a_j and
                                               * d x_ij / d b_i in Aij and Bij resp.
                                               * This function is called only if point i is visible in * image j
                                               * (i.e. vmask[i, j]==1). Also, A_ij and B_ij are mnp x cnp and mnp x pnp
                                               * matrices resp. and they should be stored in row-major order.
                                               *
                                               * If NULL, the jacobians are approximated by repetitive proj calls
                                               * and finite differences. 
                                               */
    void *adata,       /* pointer to possibly additional data, passed uninterpreted to proj, projac */ 

    const int itmax,   /* I: maximum number of iterations. itmax==0 signals jacobian verification followed by immediate return */
    const int verbose, /* I: verbosity */
    const double opts[SBA_OPTSSZ],
	                     /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \epsilon4]. Respectively the scale factor for initial \mu,
                        * stoping thresholds for ||J^T e||_inf, ||dp||_2, ||e||_2 and (||e||_2-||e_new||_2)/||e||_2
                        */
    double info[SBA_INFOSZ]
	                     /* O: information regarding the minimization. Set to NULL if don't care
                        * info[0]=||e||_2 at initial p.
                        * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                        * info[5]= # iterations,
                        * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                        *                                 2 - stopped by small dp
                        *                                 3 - stopped by itmax
                        *                                 4 - stopped by small relative reduction in ||e||_2
                        *                                 5 - too many attempts to increase damping. Restart with increased mu
                        *                                 6 - stopped by small ||e||_2
                        *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                        * info[7]= # function evaluations
                        * info[8]= # jacobian evaluations
			                  * info[9]= # number of linear systems solved, i.e. number of attempts	for reducing error
                        */
)
{
int retval;
struct wrap_motstr_data_ wdata;
static void (*fjac)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata);

  wdata.proj=proj;
  wdata.projac=projac;
  wdata.cnp=cnp;
  wdata.pnp=pnp;
  wdata.mnp=mnp;
  wdata.adata=adata;

  fjac=(projac)? sba_motstr_Qs_jac : sba_motstr_Qs_fdjac;
  retval=sba_motstr_levmar_x(n, ncon, m, mcon, vmask, p, cnp, pnp, x, covx, mnp, sba_motstr_Qs, fjac, &wdata, itmax, verbose, opts, info);

  if(info){
    register int i;
    int nvis;

    /* count visible image points */
    for(i=nvis=0; i<n*m; ++i)
      nvis+=(vmask[i]!=0);

    /* each "func" & "fjac" evaluation requires nvis "proj" & "projac" evaluations */
    info[7]*=nvis;
    info[8]*=nvis;
  }

  return retval;
}


/* 
 * Simple driver to sba_mot_levmar_x for bundle adjustment on camera parameters.
 *
 * Returns the number of iterations (>=0) if successfull, SBA_ERROR if failed
 */

int sba_mot_levmar(
    const int n,   /* number of points */
    const int m,   /* number of images */
    const int mcon,/* number of images (starting from the 1st) whose parameters should not be modified.
					          * All A_ij (see below) with j<mcon are assumed to be zero
					          */
    char *vmask,  /* visibility mask: vmask[i, j]=1 if point i visible in image j, 0 otherwise. nxm */
    double *p,    /* initial parameter vector p0: (a1, ..., am).
                   * aj are the image j parameters, size m*cnp */
    const int cnp,/* number of parameters for ONE camera; e.g. 6 for Euclidean cameras */
    double *x,    /* measurements vector: (x_11^T, .. x_1m^T, ..., x_n1^T, .. x_nm^T)^T where
                   * x_ij is the projection of the i-th point on the j-th image.
                   * NOTE: some of the x_ij might be missing, if point i is not visible in image j;
                   * see vmask[i, j], max. size n*m*mnp
                   */
    double *covx, /* measurements covariance matrices: (Sigma_x_11, .. Sigma_x_1m, ..., Sigma_x_n1, .. Sigma_x_nm),
                   * where Sigma_x_ij is the mnp x mnp covariance of x_ij stored row-by-row. Set to NULL if no
                   * covariance estimates are available (identity matrices are implicitly used in this case).
                   * NOTE: a certain Sigma_x_ij is missing if the corresponding x_ij is also missing;
                   * see vmask[i, j], max. size n*m*mnp*mnp
                   */
    const int mnp,/* number of parameters for EACH measurement; usually 2 */
    void (*proj)(int j, int i, double *aj, double *xij, void *adata),
                                              /* functional relation computing a SINGLE image measurement. Assuming that
                                               * the parameters of camera j are aj, computes a prediction of \hat{x}_{ij}
                                               * for point i. aj is cnp x 1 and xij is mnp x 1.
                                               * This function is called only if point i is visible in  image j (i.e. vmask[i, j]==1)
                                               */
    void (*projac)(int j, int i, double *aj, double *Aij, void *adata),
                                              /* functional relation to evaluate d x_ij / d a_j in Aij 
                                               * This function is called only if point i is visible in image j
                                               * (i.e. vmask[i, j]==1). Also, A_ij are a mnp x cnp matrices
                                               * and should be stored in row-major order.
                                               *
                                               * If NULL, the jacobian is approximated by repetitive proj calls
                                               * and finite differences. 
                                               */
    void *adata,       /* pointer to possibly additional data, passed uninterpreted to proj, projac */ 

    const int itmax,   /* I: maximum number of iterations. itmax==0 signals jacobian verification followed by immediate return */
    const int verbose, /* I: verbosity */
    const double opts[SBA_OPTSSZ],
	                     /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \epsilon]. Respectively the scale factor for initial \mu,
                        * stoping thresholds for ||J^T e||_inf, ||dp||_2, ||e||_2 and (||e||_2-||e_new||_2)/||e||_2
                        */
    double info[SBA_INFOSZ]
	                     /* O: information regarding the minimization. Set to NULL if don't care
                        * info[0]=||e||_2 at initial p.
                        * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                        * info[5]= # iterations,
                        * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                        *                                 2 - stopped by small dp
                        *                                 3 - stopped by itmax
                        *                                 4 - stopped by small relative reduction in ||e||_2
                        *                                 5 - too many attempts to increase damping. Restart with increased mu
                        *                                 6 - stopped by small ||e||_2
                        *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                        * info[7]= # function evaluations
                        * info[8]= # jacobian evaluations
			                  * info[9]= # number of linear systems solved, i.e. number of attempts	for reducing error
                        */
)
{
int retval;
struct wrap_mot_data_ wdata;
void (*fjac)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata);

  wdata.proj=proj;
  wdata.projac=projac;
  wdata.cnp=cnp;
  wdata.mnp=mnp;
  wdata.adata=adata;

  fjac=(projac)? sba_mot_Qs_jac : sba_mot_Qs_fdjac;
  retval=sba_mot_levmar_x(n, m, mcon, vmask, p, cnp, x, covx, mnp, sba_mot_Qs, fjac, &wdata, itmax, verbose, opts, info);

  if(info){
    register int i;
    int nvis;

    /* count visible image points */
    for(i=nvis=0; i<n*m; ++i)
      nvis+=(vmask[i]!=0);

    /* each "func" & "fjac" evaluation requires nvis "proj" & "projac" evaluations */
    info[7]*=nvis;
    info[8]*=nvis;
  }

  return retval;
}

/* 
 * Simple driver to sba_str_levmar_x for bundle adjustment on structure parameters.
 *
 * Returns the number of iterations (>=0) if successfull, SBA_ERROR if failed
 */

int sba_str_levmar(
    const int n,   /* number of points */
    const int ncon,/* number of points (starting from the 1st) whose parameters should not be modified.
                    * All B_ij (see below) with i<ncon are assumed to be zero
                    */
    const int m,   /* number of images */
    char *vmask,  /* visibility mask: vmask[i, j]=1 if point i visible in image j, 0 otherwise. nxm */
    double *p,    /* initial parameter vector p0: (b1, ..., bn).
                   * bi are the i-th point parameters, size n*pnp
                   */
    const int pnp,/* number of parameters for ONE point; e.g. 3 for Euclidean points */
    double *x,    /* measurements vector: (x_11^T, .. x_1m^T, ..., x_n1^T, .. x_nm^T)^T where
                   * x_ij is the projection of the i-th point on the j-th image.
                   * NOTE: some of the x_ij might be missing, if point i is not visible in image j;
                   * see vmask[i, j], max. size n*m*mnp
                   */
    double *covx, /* measurements covariance matrices: (Sigma_x_11, .. Sigma_x_1m, ..., Sigma_x_n1, .. Sigma_x_nm),
                   * where Sigma_x_ij is the mnp x mnp covariance of x_ij stored row-by-row. Set to NULL if no
                   * covariance estimates are available (identity matrices are implicitly used in this case).
                   * NOTE: a certain Sigma_x_ij is missing if the corresponding x_ij is also missing;
                   * see vmask[i, j], max. size n*m*mnp*mnp
                   */
    const int mnp,/* number of parameters for EACH measurement; usually 2 */
    void (*proj)(int j, int i, double *bi, double *xij, void *adata),
                                              /* functional relation computing a SINGLE image measurement. Assuming that
                                               * the parameters of point i are bi, computes a prediction of \hat{x}_{ij}.
                                               * bi is pnp x 1 and  xij is mnp x 1. This function is called only if point
                                               * i is visible in image j (i.e. vmask[i, j]==1)
                                               */
    void (*projac)(int j, int i, double *bi, double *Bij, void *adata),
                                              /* functional relation to evaluate d x_ij / d b_i in Bij.
                                               * This function is called only if point i is visible in image j
                                               * (i.e. vmask[i, j]==1). Also, B_ij are mnp x pnp matrices
                                               * and they should be stored in row-major order.
                                               *
                                               * If NULL, the jacobians are approximated by repetitive proj calls
                                               * and finite differences. 
                                               */
    void *adata,       /* pointer to possibly additional data, passed uninterpreted to proj, projac */ 

    const int itmax,   /* I: maximum number of iterations. itmax==0 signals jacobian verification followed by immediate return */
    const int verbose, /* I: verbosity */
    const double opts[SBA_OPTSSZ],
	                     /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \epsilon4]. Respectively the scale factor for initial \mu,
                        * stoping thresholds for ||J^T e||_inf, ||dp||_2, ||e||_2 and (||e||_2-||e_new||_2)/||e||_2
                        */
    double info[SBA_INFOSZ]
	                     /* O: information regarding the minimization. Set to NULL if don't care
                        * info[0]=||e||_2 at initial p.
                        * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                        * info[5]= # iterations,
                        * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                        *                                 2 - stopped by small dp
                        *                                 3 - stopped by itmax
                        *                                 4 - stopped by small relative reduction in ||e||_2
                        *                                 5 - too many attempts to increase damping. Restart with increased mu
                        *                                 6 - stopped by small ||e||_2
                        *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                        * info[7]= # function evaluations
                        * info[8]= # jacobian evaluations
			                  * info[9]= # number of linear systems solved, i.e. number of attempts	for reducing error
                        */
)
{
int retval;
struct wrap_str_data_ wdata;
static void (*fjac)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata);

  wdata.proj=proj;
  wdata.projac=projac;
  wdata.pnp=pnp;
  wdata.mnp=mnp;
  wdata.adata=adata;

  fjac=(projac)? sba_str_Qs_jac : sba_str_Qs_fdjac;
  retval=sba_str_levmar_x(n, ncon, m, vmask, p, pnp, x, covx, mnp, sba_str_Qs, fjac, &wdata, itmax, verbose, opts, info);

  if(info){
    register int i;
    int nvis;

    /* count visible image points */
    for(i=nvis=0; i<n*m; ++i)
      nvis+=(vmask[i]!=0);

    /* each "func" & "fjac" evaluation requires nvis "proj" & "projac" evaluations */
    info[7]*=nvis;
    info[8]*=nvis;
  }

  return retval;
}
