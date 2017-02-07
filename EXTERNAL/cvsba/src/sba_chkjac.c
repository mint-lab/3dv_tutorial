/////////////////////////////////////////////////////////////////////////////////
//// 
////  Verification routines for the jacobians employed in the expert & simple drivers
////  for sparse bundle adjustment based on the Levenberg - Marquardt minimization algorithm
////  Copyright (C) 2005-2008 Manolis Lourakis (lourakis at ics forth gr)
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
#include <math.h>
#include <float.h>

#include "compiler.h"
#include "sba.h"

#define emalloc(sz)       emalloc_(__FILE__, __LINE__, sz)

#define FABS(x)           (((x)>=0)? (x) : -(x))


/* auxiliary memory allocation routine with error checking */
inline static void *emalloc_(char *file, int line, size_t sz)
{
void *ptr;

  ptr=(void *)malloc(sz);
  if(ptr==NULL){
    fprintf(stderr, "SBA: memory allocation request for %u bytes failed in file %s, line %d, exiting", sz, file, line);
    exit(1);
  }

  return ptr;
}

/* 
 * Check the jacobian of a projection function in nvars variables
 * evaluated at a point p, for consistency with the function itself.
 * Expert version
 *
 * Based on fortran77 subroutine CHKDER by
 * Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
 * Argonne National Laboratory. MINPACK project. March 1980.
 *
 *
 * func points to a function from R^{nvars} --> R^{nobs}: Given a p in R^{nvars}
 *      it yields hx in R^{nobs}
 * jacf points to a function implementing the jacobian of func, whose consistency with
 *     func is to be tested. Given a p in R^{nvars}, jacf computes into the nvis*(Asz+Bsz)
 *     matrix jac the jacobian of func at p. Note the jacobian is sparse, consisting of
 *     all A_ij, B_ij and that row i of jac corresponds to the gradient of the i-th
 *     component of func, evaluated at p.
 * p is an input array of length nvars containing the point of evaluation.
 * idxij, rcidxs, rcsubs, ncon, mcon, cnp, pnp, mnp are as usual. Note that if cnp=0 or
 *     pnp=0 a jacobian corresponding resp. to motion or camera parameters
 *     only is assumed.
 * func_adata, jac_adata point to possible additional data and are passed
 *     uninterpreted to func, jacf respectively.
 * err is an array of length nobs. On output, err contains measures
 *     of correctness of the respective gradients. if there is
 *     no severe loss of significance, then if err[i] is 1.0 the
 *     i-th gradient is correct, while if err[i] is 0.0 the i-th
 *     gradient is incorrect. For values of err between 0.0 and 1.0,
 *     the categorization is less certain. In general, a value of
 *     err[i] greater than 0.5 indicates that the i-th gradient is
 *     probably correct, while a value of err[i] less than 0.5
 *     indicates that the i-th gradient is probably incorrect.
 *
 * CAUTION: THIS FUNCTION IS NOT 100% FOOLPROOF. The
 * following excerpt comes from CHKDER's documentation:
 *
 *     "The function does not perform reliably if cancellation or
 *     rounding errors cause a severe loss of significance in the
 *     evaluation of a function. therefore, none of the components
 *     of p should be unusually small (in particular, zero) or any
 *     other value which may cause loss of significance."
 */

void sba_motstr_chkjac_x(
    void (*func)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata),
    void (*jacf)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata),
    double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, int ncon, int mcon, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata)
{
const double factor=100.0, one=1.0, zero=0.0;
double *fvec, *fjac, *pp, *fvecp, *buf, *err;

int nvars, nobs, m, n, Asz, Bsz, ABsz, nnz;
register int i, j, ii, jj;
double eps, epsf, temp, epsmch, epslog;
register double *ptr1, *ptr2, *pab;
double *pa, *pb;
int fvec_sz, pp_sz, fvecp_sz, numerr=0;

  nobs=idxij->nnz*mnp;
  n=idxij->nr; m=idxij->nc;
  nvars=m*cnp + n*pnp;
  epsmch=DBL_EPSILON;
  eps=sqrt(epsmch);

  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;
  fjac=(double *)emalloc(idxij->nnz*ABsz*sizeof(double));

  fvec_sz=fvecp_sz=nobs;
  pp_sz=nvars;
  buf=(double *)emalloc((fvec_sz + pp_sz + fvecp_sz)*sizeof(double));
  fvec=buf;
  pp=fvec+fvec_sz;
  fvecp=pp+pp_sz;

  err=(double *)emalloc(nobs*sizeof(double));

  /* compute fvec=func(p) */
  (*func)(p, idxij, rcidxs, rcsubs, fvec, func_adata);

  /* compute the jacobian at p */
  (*jacf)(p, idxij, rcidxs, rcsubs, fjac, jac_adata);

  /* compute pp */
  for(j=0; j<nvars; ++j){
    temp=eps*FABS(p[j]);
    if(temp==zero) temp=eps;
    pp[j]=p[j]+temp;
  }

  /* compute fvecp=func(pp) */
  (*func)(pp, idxij, rcidxs, rcsubs, fvecp, func_adata);

  epsf=factor*epsmch;
  epslog=log10(eps);

  for(i=0; i<nobs; ++i)
    err[i]=zero;

  pa=p;
  pb=p + m*cnp;
  for(i=0; i<n; ++i){
    nnz=sba_crsm_row_elmidxs(idxij, i, rcidxs, rcsubs); /* find nonzero A_ij, B_ij, j=0...m-1, actual column numbers in rcsubs */
    for(j=0; j<nnz; ++j){
      ptr2=err + idxij->val[rcidxs[j]]*mnp; // set ptr2 to point into err

      if(cnp && rcsubs[j]>=mcon){ // A_ij is nonzero
        ptr1=fjac + idxij->val[rcidxs[j]]*ABsz; // set ptr1 to point to A_ij
        pab=pa + rcsubs[j]*cnp;
        for(jj=0; jj<cnp; ++jj){
          temp=FABS(pab[jj]);
          if(temp==zero) temp=one;

          for(ii=0; ii<mnp; ++ii)
            ptr2[ii]+=temp*ptr1[ii*cnp+jj];
        }
      }

      if(pnp && i>=ncon){ // B_ij is nonzero
        ptr1=fjac + idxij->val[rcidxs[j]]*ABsz + Asz; // set ptr1 to point to B_ij
        pab=pb + i*pnp;
        for(jj=0; jj<pnp; ++jj){
          temp=FABS(pab[jj]);
          if(temp==zero) temp=one;

          for(ii=0; ii<mnp; ++ii)
            ptr2[ii]+=temp*ptr1[ii*pnp+jj];
        }
      }
    }
  }

  for(i=0; i<nobs; ++i){
    temp=one;
    if(fvec[i]!=zero && fvecp[i]!=zero && FABS(fvecp[i]-fvec[i])>=epsf*FABS(fvec[i]))
        temp=eps*FABS((fvecp[i]-fvec[i])/eps - err[i])/(FABS(fvec[i])+FABS(fvecp[i]));
    err[i]=one;
    if(temp>epsmch && temp<eps)
        err[i]=(log10(temp) - epslog)/epslog;
    if(temp>=eps) err[i]=zero;
  }

  free(fjac);
  free(buf);

  for(i=0; i<n; ++i){
    nnz=sba_crsm_row_elmidxs(idxij, i, rcidxs, rcsubs); /* find nonzero err_ij, j=0...m-1 */
    for(j=0; j<nnz; ++j){
      if(i<ncon && rcsubs[j]<mcon) continue; // corresponding gradients are taken to be zero

      ptr1=err + idxij->val[rcidxs[j]]*mnp; // set ptr1 to point into err
      for(ii=0; ii<mnp; ++ii)
        if(ptr1[ii]<=0.5){
          fprintf(stderr, "SBA: gradient %d (corresponding to element %d of the projection of point %d on camera %d) is %s (err=%g)\n",
                  idxij->val[rcidxs[j]]*mnp+ii, ii, i, rcsubs[j], (ptr1[ii]==0.0)? "wrong" : "probably wrong", ptr1[ii]);
          ++numerr;
        }
    }
  }
  if(numerr) fprintf(stderr, "SBA: found %d suspicious gradients out of %d\n\n", numerr, nobs);

  free(err);

  return;
}

void sba_mot_chkjac_x(
    void (*func)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata),
    void (*jacf)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata),
    double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, int mcon, int cnp, int mnp, void *func_adata, void *jac_adata)
{
  sba_motstr_chkjac_x(func, jacf, p, idxij, rcidxs, rcsubs, 0, mcon, cnp, 0, mnp, func_adata, jac_adata);
}

void sba_str_chkjac_x(
    void (*func)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata),
    void (*jacf)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata),
    double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, int ncon, int pnp, int mnp, void *func_adata, void *jac_adata)
{
  sba_motstr_chkjac_x(func, jacf, p, idxij, rcidxs, rcsubs, ncon, 0, 0, pnp, mnp, func_adata, jac_adata);
}

#if 0
/* Routines for directly checking the jacobians supplied to the simple drivers.
 * They shouldn't be necessary since these jacobians can be verified indirectly
 * through the expert sba_XXX_chkjac_x() routines.
 */

/*****************************************************************************************/
// Sample code for using sba_motstr_chkjac():

  for(i=ncon; i<n; ++i)
    for(j=mcon; j<m; ++j){
      if(!vmask[i*m+j]) continue; // point i does not appear in image j

    sba_motstr_chkjac(proj, projac, p+j*cnp, p+m*cnp+i*pnp, j, i, cnp, pnp, mnp, adata, adata);
  }


/*****************************************************************************************/


/* union used for passing pointers to the user-supplied functions for the motstr/mot/str simple drivers */
union proj_projac{
  struct{
    void (*proj)(int j, int i, double *aj, double *bi, double *xij, void *adata);
    void (*projac)(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata);
  } motstr;

  struct{
    void (*proj)(int j, int i, double *aj, double *xij, void *adata);
    void (*projac)(int j, int i, double *aj, double *Aij, void *adata);
  } mot;

  struct{
    void (*proj)(int j, int i, double *bi, double *xij, void *adata);
    void (*projac)(int j, int i, double *bi, double *Bij, void *adata);
  } str;
};


/* 
 * Check the jacobian of a projection function in cnp+pnp variables
 * evaluated at a point p, for consistency with the function itself.
 * Simple version of the above, NOT to be called directly
 *
 * Based on fortran77 subroutine CHKDER by
 * Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
 * Argonne National Laboratory. MINPACK project. March 1980.
 *
 *
 * proj points to a function from R^{cnp+pnp} --> R^{mnp}: Given a p=(aj, bi) in R^{cnp+pnp}
 *      it yields hx in R^{mnp}
 * projac points to a function implementing the jacobian of func, whose consistency with proj
 *     is to be tested. Given a p in R^{cnp+pnp}, jacf computes into the matrix jac=[Aij | Bij]
 *     jacobian of proj at p. Note that row i of jac corresponds to the gradient of the i-th
 *     component of proj, evaluated at p.
 * aj, bi are input arrays of lengths cnp, pnp containing the parameters for the point of
 *     evaluation, i.e. j-th camera and i-th point
 * jj, ii specify the point (ii) whose projection jacobian in image (jj) is being checked
 * cnp, pnp, mnp are as usual. Note that if cnp=0 or
 *     pnp=0 a jacobian corresponding resp. to motion or camera parameters
 *     only is assumed.
 * func_adata, jac_adata point to possible additional data and are passed
 *     uninterpreted to func, jacf respectively.
 * err is an array of length mnp. On output, err contains measures
 *     of correctness of the respective gradients. if there is
 *     no severe loss of significance, then if err[i] is 1.0 the
 *     i-th gradient is correct, while if err[i] is 0.0 the i-th
 *     gradient is incorrect. For values of err between 0.0 and 1.0,
 *     the categorization is less certain. In general, a value of
 *     err[i] greater than 0.5 indicates that the i-th gradient is
 *     probably correct, while a value of err[i] less than 0.5
 *     indicates that the i-th gradient is probably incorrect.
 *
 * CAUTION: THIS FUNCTION IS NOT 100% FOOLPROOF. The
 * following excerpt comes from CHKDER's documentation:
 *
 *     "The function does not perform reliably if cancellation or
 *     rounding errors cause a severe loss of significance in the
 *     evaluation of a function. therefore, none of the components
 *     of p should be unusually small (in particular, zero) or any
 *     other value which may cause loss of significance."
 */

static void sba_chkjac(
    union proj_projac *funcs, double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata)
{
const double factor=100.0, one=1.0, zero=0.0;
double *fvec, *fjac, *Aij, *Bij, *ajp, *bip, *fvecp, *buf, *err;

int Asz, Bsz;
register int i, j;
double eps, epsf, temp, epsmch, epslog;
int fvec_sz, ajp_sz, bip_sz, fvecp_sz, err_sz, numerr=0;

  epsmch=DBL_EPSILON;
  eps=sqrt(epsmch);

  Asz=mnp*cnp; Bsz=mnp*pnp;
  fjac=(double *)emalloc((Asz+Bsz)*sizeof(double));
  Aij=fjac;
  Bij=Aij+Asz;

  fvec_sz=fvecp_sz=mnp;
  ajp_sz=cnp; bip_sz=pnp;
  err_sz=mnp;
  buf=(double *)emalloc((fvec_sz + ajp_sz + bip_sz + fvecp_sz + err_sz)*sizeof(double));
  fvec=buf;
  ajp=fvec+fvec_sz;
  bip=ajp+ajp_sz;
  fvecp=bip+bip_sz;
  err=fvecp+fvecp_sz;

  /* compute fvec=proj(p), p=(aj, bi) & the jacobian at p */
  if(cnp && pnp){
    (*(funcs->motstr.proj))(jj, ii, aj, bi, fvec, func_adata);
    (*(funcs->motstr.projac))(jj, ii, aj, bi, Aij, Bij, jac_adata);
  }
  else if(cnp){
    (*(funcs->mot.proj))(jj, ii, aj, fvec, func_adata);
    (*(funcs->mot.projac))(jj, ii, aj, Aij, jac_adata);
  }
  else{
    (*(funcs->str.proj))(jj, ii, bi, fvec, func_adata);
    (*(funcs->str.projac))(jj, ii, bi, Bij, jac_adata);
  }

  /* compute pp, pp=(ajp, bip) */
  for(j=0; j<cnp; ++j){
    temp=eps*FABS(aj[j]);
    if(temp==zero) temp=eps;
    ajp[j]=aj[j]+temp;
  }
  for(j=0; j<pnp; ++j){
    temp=eps*FABS(bi[j]);
    if(temp==zero) temp=eps;
    bip[j]=bi[j]+temp;
  }

  /* compute fvecp=proj(pp) */
  if(cnp && pnp)
    (*(funcs->motstr.proj))(jj, ii, ajp, bip, fvecp, func_adata);
  else if(cnp)
    (*(funcs->mot.proj))(jj, ii, ajp, fvecp, func_adata);
  else
    (*(funcs->str.proj))(jj, ii, bip, fvecp, func_adata);

  epsf=factor*epsmch;
  epslog=log10(eps);

  for(i=0; i<mnp; ++i)
    err[i]=zero;

  for(j=0; j<cnp; ++j){
    temp=FABS(aj[j]);
    if(temp==zero) temp=one;

    for(i=0; i<mnp; ++i)
      err[i]+=temp*Aij[i*cnp+j];
  }
  for(j=0; j<pnp; ++j){
    temp=FABS(bi[j]);
    if(temp==zero) temp=one;

    for(i=0; i<mnp; ++i)
      err[i]+=temp*Bij[i*pnp+j];
  }

  for(i=0; i<mnp; ++i){
    temp=one;
    if(fvec[i]!=zero && fvecp[i]!=zero && FABS(fvecp[i]-fvec[i])>=epsf*FABS(fvec[i]))
        temp=eps*FABS((fvecp[i]-fvec[i])/eps - err[i])/(FABS(fvec[i])+FABS(fvecp[i]));
    err[i]=one;
    if(temp>epsmch && temp<eps)
        err[i]=(log10(temp) - epslog)/epslog;
    if(temp>=eps) err[i]=zero;
  }

  for(i=0; i<mnp; ++i)
    if(err[i]<=0.5){
      fprintf(stderr, "SBA: gradient %d (corresponding to element %d of the projection of point %d on camera %d) is %s (err=%g)\n",
                i, i, ii, jj, (err[i]==0.0)? "wrong" : "probably wrong", err[i]);
      ++numerr;
  }
  if(numerr) fprintf(stderr, "SBA: found %d suspicious gradients out of %d\n\n", numerr, mnp);

  free(fjac);
  free(buf);

  return;
}

void sba_motstr_chkjac(
    void (*proj)(int jj, int ii, double *aj, double *bi, double *xij, void *adata),
    void (*projac)(int jj, int ii, double *aj, double *bi, double *Aij, double *Bij, void *adata),
    double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata)
{
union proj_projac funcs;

  funcs.motstr.proj=proj;
  funcs.motstr.projac=projac;

  sba_chkjac(&funcs, aj, bi, jj, ii, cnp, pnp, mnp, func_adata, jac_adata);
}

void sba_mot_chkjac(
    void (*proj)(int jj, int ii, double *aj, double *xij, void *adata),
    void (*projac)(int jj, int ii, double *aj, double *Aij, void *adata),
    double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata)
{
union proj_projac funcs;

  funcs.mot.proj=proj;
  funcs.mot.projac=projac;

  sba_chkjac(&funcs, aj, NULL, jj, ii, cnp, 0, mnp, func_adata, jac_adata);
}

void sba_str_chkjac(
    void (*proj)(int jj, int ii, double *bi, double *xij, void *adata),
    void (*projac)(int jj, int ii, double *bi, double *Bij, void *adata),
    double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata)
{
union proj_projac funcs;

  funcs.str.proj=proj;
  funcs.str.projac=projac;

  sba_chkjac(&funcs, NULL, bi, jj, ii, 0, pnp, mnp, func_adata, jac_adata);
}
#endif /* 0 */
