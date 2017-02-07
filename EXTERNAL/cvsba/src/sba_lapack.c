/////////////////////////////////////////////////////////////////////////////////
//// 
////  Linear algebra operations for the sba package
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

/* A note on memory alignment:
 *
 * Several of the functions below use a piece of dynamically allocated memory
 * to store variables of different size (i.e., ints and doubles). To avoid
 * alignment problems, care must be taken so that elements that are larger
 * (doubles) are stored before smaller ones (ints). This ensures proper
 * alignment under different alignment choices made by different CPUs:
 * For instance, a double variable is aligned on x86 to 4 bytes but
 * aligned to 8 bytes on AMD64 despite having the same size of 8 bytes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "compiler.h"
#include "sba.h"

#ifdef SBA_APPEND_UNDERSCORE_SUFFIX
#define F77_FUNC(func)    func ## _
#else
#define F77_FUNC(func)    func 
#endif /* SBA_APPEND_UNDERSCORE_SUFFIX */


/* declarations of LAPACK routines employed */

/* QR decomposition */
extern int F77_FUNC(dgeqrf)(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern int F77_FUNC(dorgqr)(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);

/* solution of triangular system */
extern int F77_FUNC(dtrtrs)(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);

/* Cholesky decomposition, linear system solution and matrix inversion */
extern int F77_FUNC(dpotf2)(char *uplo, int *n, double *a, int *lda, int *info); /* unblocked Cholesky */
extern int F77_FUNC(dpotrf)(char *uplo, int *n, double *a, int *lda, int *info); /* block version of dpotf2 */
extern int F77_FUNC(dpotrs)(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern int F77_FUNC(dpotri)(char *uplo, int *n, double *a, int *lda, int *info);

/* LU decomposition, linear system solution and matrix inversion */
extern int F77_FUNC(dgetrf)(int *m, int *n, double *a, int *lda, int *ipiv, int *info); /* blocked LU */
extern int F77_FUNC(dgetf2)(int *m, int *n, double *a, int *lda, int *ipiv, int *info); /* unblocked LU */
extern int F77_FUNC(dgetrs)(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern int F77_FUNC(dgetri)(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);

/* SVD */
extern int F77_FUNC(dgesvd)(char *jobu, char *jobvt, int *m, int *n,
           double *a, int *lda, double *s, double *u, int *ldu,
           double *vt, int *ldvt, double *work, int *lwork,
           int *info);

/* lapack 3.0 routine, faster than dgesvd() */
extern int F77_FUNC(dgesdd)(char *jobz, int *m, int *n, double *a, int *lda,
           double *s, double *u, int *ldu, double *vt, int *ldvt,
           double *work, int *lwork, int *iwork, int *info);


/* Bunch-Kaufman factorization of a real symmetric matrix A, solution of linear systems and matrix inverse */
extern int F77_FUNC(dsytrf)(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info); /* blocked ver. */
extern int F77_FUNC(dsytrs)(char *uplo, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern int F77_FUNC(dsytri)(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *info);


/*
 * This function returns the solution of Ax = b
 *
 * The function is based on QR decomposition with explicit computation of Q:
 * If A=Q R with Q orthogonal and R upper triangular, the linear system becomes
 * Q R x = b or R x = Q^T b.
 *
 * A is mxm, b is mx1. Argument iscolmaj specifies whether A is
 * stored in column or row major order. Note that if iscolmaj==1
 * this function modifies A!
 *
 * The function returns 0 in case of error, 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_QR(double *A, double *B, double *x, int m, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0, nb=0;

double *a, *r, *tau, *work;
int a_sz, r_sz, tau_sz, tot_sz;
register int i, j;
int info, worksz, nrhs=1;
register double sum;
   
    if(A==NULL){
      if(buf) free(buf);
      buf=NULL;
      buf_sz=0;

      return 1;
    }

    /* calculate required memory size */
    a_sz=(iscolmaj)? 0 : m*m;
    r_sz=m*m; /* only the upper triangular part really needed */
    tau_sz=m;
    if(!nb){
#ifndef SBA_LS_SCARCE_MEMORY
      double tmp;

      worksz=-1; // workspace query; optimal size is returned in tmp
      F77_FUNC(dgeqrf)((int *)&m, (int *)&m, NULL, (int *)&m, NULL, (double *)&tmp, (int *)&worksz, (int *)&info);
      nb=((int)tmp)/m; // optimal worksize is m*nb
#else
      nb=1; // min worksize is m
#endif /* SBA_LS_SCARCE_MEMORY */
    }
    worksz=nb*m;
    tot_sz=a_sz + r_sz + tau_sz + worksz;

    if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
      if(buf) free(buf); /* free previously allocated memory */

      buf_sz=tot_sz;
      buf=(double *)malloc(buf_sz*sizeof(double));
      if(!buf){
        fprintf(stderr, "memory allocation in sba_Axb_QR() failed!\n");
        exit(1);
      }
    }

    if(!iscolmaj){
    	a=buf;
	    /* store A (column major!) into a */
	    for(i=0; i<m; ++i)
		    for(j=0; j<m; ++j)
			    a[i+j*m]=A[i*m+j];
    }
    else a=A; /* no copying required */

    r=buf+a_sz;
    tau=r+r_sz;
    work=tau+tau_sz;

  /* QR decomposition of A */
  F77_FUNC(dgeqrf)((int *)&m, (int *)&m, a, (int *)&m, tau, work, (int *)&worksz, (int *)&info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dgeqrf in sba_Axb_QR()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "Unknown LAPACK error %d for dgeqrf in sba_Axb_QR()\n", info);
      return 0;
    }
  }

  /* R is now stored in the upper triangular part of a; copy it in r so that dorgqr() below won't destroy it */
  for(i=0; i<r_sz; ++i)
    r[i]=a[i];

  /* compute Q using the elementary reflectors computed by the above decomposition */
  F77_FUNC(dorgqr)((int *)&m, (int *)&m, (int *)&m, a, (int *)&m, tau, work, (int *)&worksz, (int *)&info);
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dorgqr in sba_Axb_QR()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "Unknown LAPACK error (%d) in sba_Axb_QR()\n", info);
      return 0;
    }
  }

  /* Q is now in a; compute Q^T b in x */
  for(i=0; i<m; ++i){
    for(j=0, sum=0.0; j<m; ++j)
      sum+=a[i*m+j]*B[j];
    x[i]=sum;
  }

  /* solve the linear system R x = Q^t b */
  F77_FUNC(dtrtrs)("U", "N", "N", (int *)&m, (int *)&nrhs, r, (int *)&m, x, (int *)&m, &info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dtrtrs in sba_Axb_QR()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the %d-th diagonal element of A is zero (singular matrix) in sba_Axb_QR()\n", info);
      return 0;
    }
  }

	return 1;
}

/*
 * This function returns the solution of Ax = b
 *
 * The function is based on QR decomposition without computation of Q:
 * If A=Q R with Q orthogonal and R upper triangular, the linear system becomes
 * (A^T A) x = A^T b or (R^T Q^T Q R) x = A^T b or (R^T R) x = A^T b.
 * This amounts to solving R^T y = A^T b for y and then R x = y for x
 * Note that Q does not need to be explicitly computed
 *
 * A is mxm, b is mx1. Argument iscolmaj specifies whether A is
 * stored in column or row major order. Note that if iscolmaj==1
 * this function modifies A!
 *
 * The function returns 0 in case of error, 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_QRnoQ(double *A, double *B, double *x, int m, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0, nb=0;

double *a, *tau, *work;
int a_sz, tau_sz, tot_sz;
register int i, j;
int info, worksz, nrhs=1;
register double sum;
   
    if(A==NULL){
      if(buf) free(buf);
      buf=NULL;
      buf_sz=0;

      return 1;
    }

    /* calculate required memory size */
    a_sz=(iscolmaj)? 0 : m*m;
    tau_sz=m;
    if(!nb){
#ifndef SBA_LS_SCARCE_MEMORY
      double tmp;

      worksz=-1; // workspace query; optimal size is returned in tmp
      F77_FUNC(dgeqrf)((int *)&m, (int *)&m, NULL, (int *)&m, NULL, (double *)&tmp, (int *)&worksz, (int *)&info);
      nb=((int)tmp)/m; // optimal worksize is m*nb
#else
      nb=1; // min worksize is m
#endif /* SBA_LS_SCARCE_MEMORY */
    }
    worksz=nb*m;
    tot_sz=a_sz + tau_sz + worksz;

    if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
      if(buf) free(buf); /* free previously allocated memory */

      buf_sz=tot_sz;
      buf=(double *)malloc(buf_sz*sizeof(double));
      if(!buf){
        fprintf(stderr, "memory allocation in sba_Axb_QRnoQ() failed!\n");
        exit(1);
      }
    }

    if(!iscolmaj){
    	a=buf;
	/* store A (column major!) into a */
	for(i=0; i<m; ++i)
		for(j=0; j<m; ++j)
			a[i+j*m]=A[i*m+j];
    }
    else a=A; /* no copying required */

    tau=buf+a_sz;
    work=tau+tau_sz;

  /* compute A^T b in x */
  for(i=0; i<m; ++i){
    for(j=0, sum=0.0; j<m; ++j)
      sum+=a[i*m+j]*B[j];
    x[i]=sum;
  }

  /* QR decomposition of A */
  F77_FUNC(dgeqrf)((int *)&m, (int *)&m, a, (int *)&m, tau, work, (int *)&worksz, (int *)&info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dgeqrf in sba_Axb_QRnoQ()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "Unknown LAPACK error %d for dgeqrf in sba_Axb_QRnoQ()\n", info);
      return 0;
    }
  }

  /* R is stored in the upper triangular part of a */

  /* solve the linear system R^T y = A^t b */
  F77_FUNC(dtrtrs)("U", "T", "N", (int *)&m, (int *)&nrhs, a, (int *)&m, x, (int *)&m, &info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dtrtrs in sba_Axb_QRnoQ()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the %d-th diagonal element of A is zero (singular matrix) in sba_Axb_QRnoQ()\n", info);
      return 0;
    }
  }

  /* solve the linear system R x = y */
  F77_FUNC(dtrtrs)("U", "N", "N", (int *)&m, (int *)&nrhs, a, (int *)&m, x, (int *)&m, &info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dtrtrs in sba_Axb_QRnoQ()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the %d-th diagonal element of A is zero (singular matrix) in sba_Axb_QRnoQ()\n", info);
      return 0;
    }
  }

	return 1;
}

/*
 * This function returns the solution of Ax=b
 *
 * The function assumes that A is symmetric & positive definite and employs
 * the Cholesky decomposition:
 * If A=U^T U with U upper triangular, the system to be solved becomes
 * (U^T U) x = b
 * This amounts to solving U^T y = b for y and then U x = y for x
 *
 * A is mxm, b is mx1. Argument iscolmaj specifies whether A is
 * stored in column or row major order. Note that if iscolmaj==1
 * this function modifies A!
 *
 * The function returns 0 in case of error, 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_Chol(double *A, double *B, double *x, int m, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0;

double *a;
int a_sz, tot_sz;
register int i, j;
int info, nrhs=1;
   
    if(A==NULL){
      if(buf) free(buf);
      buf=NULL;
      buf_sz=0;

      return 1;
    }

    /* calculate required memory size */
    a_sz=(iscolmaj)? 0 : m*m;
    tot_sz=a_sz;

    if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
      if(buf) free(buf); /* free previously allocated memory */

      buf_sz=tot_sz;
      buf=(double *)malloc(buf_sz*sizeof(double));
      if(!buf){
        fprintf(stderr, "memory allocation in sba_Axb_Chol() failed!\n");
        exit(1);
      }
    }

    if(!iscolmaj){
    	a=buf;

      /* store A into a and B into x; A is assumed to be symmetric, hence
       * the column and row major order representations are the same
       */
      for(i=0; i<m; ++i){
        a[i]=A[i];
        x[i]=B[i];
      }
      for(j=m*m; i<j; ++i) // copy remaining rows; note that i is not re-initialized
        a[i]=A[i];
    }
    else{ /* no copying is necessary for A */
      a=A;
      for(i=0; i<m; ++i)
        x[i]=B[i];
    }

  /* Cholesky decomposition of A */
  //F77_FUNC(dpotf2)("U", (int *)&m, a, (int *)&m, (int *)&info);
  F77_FUNC(dpotrf)("U", (int *)&m, a, (int *)&m, (int *)&info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dpotf2/dpotrf in sba_Axb_Chol()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the leading minor of order %d is not positive definite,\nthe factorization could not be completed for dpotf2/dpotrf in sba_Axb_Chol()\n", info);
      return 0;
    }
  }

  /* below are two alternative ways for solving the linear system: */
#if 1
  /* use the computed Cholesky in one lapack call */
  F77_FUNC(dpotrs)("U", (int *)&m, (int *)&nrhs, a, (int *)&m, x, (int *)&m, &info);
  if(info<0){
    fprintf(stderr, "LAPACK error: illegal value for argument %d of dpotrs in sba_Axb_Chol()\n", -info);
    exit(1);
  }
#else
  /* solve the linear systems U^T y = b, U x = y */
  F77_FUNC(dtrtrs)("U", "T", "N", (int *)&m, (int *)&nrhs, a, (int *)&m, x, (int *)&m, &info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dtrtrs in sba_Axb_Chol()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the %d-th diagonal element of A is zero (singular matrix) in sba_Axb_Chol()\n", info);
      return 0;
    }
  }

  /* solve U x = y */
  F77_FUNC(dtrtrs)("U", "N", "N", (int *)&m, (int *)&nrhs, a, (int *)&m, x, (int *)&m, &info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dtrtrs in sba_Axb_Chol()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the %d-th diagonal element of A is zero (singular matrix) in sba_Axb_Chol()\n", info);
      return 0;
    }
  }
#endif /* 1 */

	return 1;
}

/*
 * This function returns the solution of Ax = b
 *
 * The function employs LU decomposition:
 * If A=L U with L lower and U upper triangular, then the original system
 * amounts to solving
 * L y = b, U x = y
 *
 * A is mxm, b is mx1. Argument iscolmaj specifies whether A is
 * stored in column or row major order. Note that if iscolmaj==1
 * this function modifies A!
 *
 * The function returns 0 in case of error,
 * 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_LU(double *A, double *B, double *x, int m, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0;

int a_sz, ipiv_sz, tot_sz;
register int i, j;
int info, *ipiv, nrhs=1;
double *a;
   
    if(A==NULL){
      if(buf) free(buf);
      buf=NULL;
      buf_sz=0;

      return 1;
    }

    /* calculate required memory size */
    ipiv_sz=m;
    a_sz=(iscolmaj)? 0 : m*m;
    tot_sz=a_sz*sizeof(double) + ipiv_sz*sizeof(int); /* should be arranged in that order for proper doubles alignment */

    if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
      if(buf) free(buf); /* free previously allocated memory */

      buf_sz=tot_sz;
      buf=(double *)malloc(buf_sz);
      if(!buf){
        fprintf(stderr, "memory allocation in sba_Axb_LU() failed!\n");
        exit(1);
      }
    }

    if(!iscolmaj){
      a=buf;
      ipiv=(int *)(a+a_sz);

      /* store A (column major!) into a and B into x */
	    for(i=0; i<m; ++i){
		    for(j=0; j<m; ++j)
          a[i+j*m]=A[i*m+j];

        x[i]=B[i];
      }
    }
    else{ /* no copying is necessary for A */
      a=A;
      for(i=0; i<m; ++i)
        x[i]=B[i];
      ipiv=(int *)buf;
    }

  /* LU decomposition for A */
	F77_FUNC(dgetrf)((int *)&m, (int *)&m, a, (int *)&m, ipiv, (int *)&info);  
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dgetrf illegal in sba_Axb_LU()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular matrix A for dgetrf in sba_Axb_LU()\n");
			return 0;
		}
	}

  /* solve the system with the computed LU */
  F77_FUNC(dgetrs)("N", (int *)&m, (int *)&nrhs, a, (int *)&m, ipiv, x, (int *)&m, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dgetrs illegal in sba_Axb_LU()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "unknown error for dgetrs in sba_Axb_LU()\n");
			return 0;
		}
	}

	return 1;
}

/*
 * This function returns the solution of Ax = b
 *
 * The function is based on SVD decomposition:
 * If A=U D V^T with U, V orthogonal and D diagonal, the linear system becomes
 * (U D V^T) x = b or x=V D^{-1} U^T b
 * Note that V D^{-1} U^T is the pseudoinverse A^+
 *
 * A is mxm, b is mx1. Argument iscolmaj specifies whether A is
 * stored in column or row major order. Note that if iscolmaj==1
 * this function modifies A!
 *
 * The function returns 0 in case of error, 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_SVD(double *A, double *B, double *x, int m, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0;
static double eps=-1.0;

register int i, j;
double *a, *u, *s, *vt, *work;
int a_sz, u_sz, s_sz, vt_sz, tot_sz;
double thresh, one_over_denom;
register double sum;
int info, rank, worksz, *iwork, iworksz;
   
  if(A==NULL){
    if(buf) free(buf);
    buf=NULL;
    buf_sz=0;

    return 1;
  }

  /* calculate required memory size */
#ifndef SBA_LS_SCARCE_MEMORY
  worksz=-1; // workspace query. Keep in mind that dgesdd requires more memory than dgesvd
  /* note that optimal work size is returned in thresh */
  F77_FUNC(dgesdd)("A", (int *)&m, (int *)&m, NULL, (int *)&m, NULL, NULL, (int *)&m, NULL, (int *)&m,
          (double *)&thresh, (int *)&worksz, NULL, &info);
  /* F77_FUNC(dgesvd)("A", "A", (int *)&m, (int *)&m, NULL, (int *)&m, NULL, NULL, (int *)&m, NULL, (int *)&m,
          (double *)&thresh, (int *)&worksz, &info); */
  worksz=(int)thresh;
#else
  worksz=m*(7*m+4); // min worksize for dgesdd
  //worksz=5*m; // min worksize for dgesvd
#endif /* SBA_LS_SCARCE_MEMORY */
  iworksz=8*m;
  a_sz=(!iscolmaj)? m*m : 0;
  u_sz=m*m; s_sz=m; vt_sz=m*m;

  tot_sz=(a_sz + u_sz + s_sz + vt_sz + worksz)*sizeof(double) + iworksz*sizeof(int); /* should be arranged in that order for proper doubles alignment */

  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(double *)malloc(buf_sz);
    if(!buf){
      fprintf(stderr, "memory allocation in sba_Axb_SVD() failed!\n");
      exit(1);
    }
  }

  if(!iscolmaj){
    a=buf;
    u=a+a_sz;
    /* store A (column major!) into a */
    for(i=0; i<m; ++i)
      for(j=0; j<m; ++j)
        a[i+j*m]=A[i*m+j];
  }
  else{
    a=A; /* no copying required */
    u=buf;
  }

  s=u+u_sz;
  vt=s+s_sz;
  work=vt+vt_sz;
  iwork=(int *)(work+worksz);

  /* SVD decomposition of A */
  F77_FUNC(dgesdd)("A", (int *)&m, (int *)&m, a, (int *)&m, s, u, (int *)&m, vt, (int *)&m, work, (int *)&worksz, iwork, &info);
  //F77_FUNC(dgesvd)("A", "A", (int *)&m, (int *)&m, a, (int *)&m, s, u, (int *)&m, vt, (int *)&m, work, (int *)&worksz, &info);

  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dgesdd/dgesvd in sba_Axb_SVD()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: dgesdd (dbdsdc)/dgesvd (dbdsqr) failed to converge in sba_Axb_SVD() [info=%d]\n", info);

      return 0;
    }
  }

  if(eps<0.0){
    double aux;

    /* compute machine epsilon. DBL_EPSILON should do also */
    for(eps=1.0; aux=eps+1.0, aux-1.0>0.0; eps*=0.5)
                              ;
    eps*=2.0;
  }

  /* compute the pseudoinverse in a */
  memset(a, 0, m*m*sizeof(double)); /* initialize to zero */
  for(rank=0, thresh=eps*s[0]; rank<m && s[rank]>thresh; ++rank){
    one_over_denom=1.0/s[rank];

    for(j=0; j<m; ++j)
      for(i=0; i<m; ++i)
        a[i*m+j]+=vt[rank+i*m]*u[j+rank*m]*one_over_denom;
  }

	/* compute A^+ b in x */
	for(i=0; i<m; ++i){
	  for(j=0, sum=0.0; j<m; ++j)
      sum+=a[i*m+j]*B[j];
    x[i]=sum;
  }

	return 1;
}

/*
 * This function returns the solution of Ax = b for a real symmetric matrix A
 *
 * The function is based on UDUT factorization with the pivoting
 * strategy of Bunch and Kaufman:
 * A is factored as U*D*U^T where U is upper triangular and
 * D symmetric and block diagonal (aka spectral decomposition,
 * Banachiewicz factorization, modified Cholesky factorization)
 *
 * A is mxm, b is mx1. Argument iscolmaj specifies whether A is
 * stored in column or row major order. Note that if iscolmaj==1
 * this function modifies A!
 *
 * The function returns 0 in case of error,
 * 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_BK(double *A, double *B, double *x, int m, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0, nb=0;

int a_sz, ipiv_sz, work_sz, tot_sz;
register int i, j;
int info, *ipiv, nrhs=1;
double *a, *work;
   
    if(A==NULL){
      if(buf) free(buf);
      buf=NULL;
      buf_sz=0;

      return 1;
    }

    /* calculate required memory size */
    ipiv_sz=m;
    a_sz=(iscolmaj)? 0 : m*m;
    if(!nb){
#ifndef SBA_LS_SCARCE_MEMORY
      double tmp;

      work_sz=-1; // workspace query; optimal size is returned in tmp
      F77_FUNC(dsytrf)("U", (int *)&m, NULL, (int *)&m, NULL, (double *)&tmp, (int *)&work_sz, (int *)&info);
      nb=((int)tmp)/m; // optimal worksize is m*nb
#else
      nb=-1; // min worksize is 1
#endif /* SBA_LS_SCARCE_MEMORY */
    }
    work_sz=(nb!=-1)? nb*m : 1;
    tot_sz=(a_sz + work_sz)*sizeof(double) + ipiv_sz*sizeof(int); /* should be arranged in that order for proper doubles alignment */

    if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
      if(buf) free(buf); /* free previously allocated memory */

      buf_sz=tot_sz;
      buf=(double *)malloc(buf_sz);
      if(!buf){
        fprintf(stderr, "memory allocation in sba_Axb_BK() failed!\n");
        exit(1);
      }
    }

    if(!iscolmaj){
      a=buf;
    	work=a+a_sz;

      /* store A into a and B into x; A is assumed to be symmetric, hence
       * the column and row major order representations are the same
       */
      for(i=0; i<m; ++i){
        a[i]=A[i];
        x[i]=B[i];
      }
      for(j=m*m; i<j; ++i) // copy remaining rows; note that i is not re-initialized
        a[i]=A[i];
    }
    else{ /* no copying is necessary for A */
      a=A;
      for(i=0; i<m; ++i)
        x[i]=B[i];
      work=buf;
    }
    ipiv=(int *)(work+work_sz);

  /* factorization for A */
	F77_FUNC(dsytrf)("U", (int *)&m, a, (int *)&m, ipiv, work, (int *)&work_sz, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dsytrf illegal in sba_Axb_BK()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular block diagonal matrix D for dsytrf in sba_Axb_BK() [D(%d, %d) is zero]\n", info, info);
			return 0;
		}
	}

  /* solve the system with the computed factorization */
  F77_FUNC(dsytrs)("U", (int *)&m, (int *)&nrhs, a, (int *)&m, ipiv, x, (int *)&m, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dsytrs illegal in sba_Axb_BK()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "unknown error for dsytrs in sba_Axb_BK()\n");
			return 0;
		}
	}

	return 1;
}

/*
 * This function computes the inverse of a square matrix whose upper triangle
 * is stored in A into its lower triangle using LU decomposition
 *
 * The function returns 0 in case of error (e.g. A is singular),
 * 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_symat_invert_LU(double *A, int m)
{
static double *buf=NULL;
static int buf_sz=0, nb=0;

int a_sz, ipiv_sz, work_sz, tot_sz;
register int i, j;
int info, *ipiv;
double *a, *work;
   
  if(A==NULL){
    if(buf) free(buf);
    buf=NULL;
    buf_sz=0;

    return 1;
  }

  /* calculate required memory size */
  ipiv_sz=m;
  a_sz=m*m;
  if(!nb){
#ifndef SBA_LS_SCARCE_MEMORY
    double tmp;

    work_sz=-1; // workspace query; optimal size is returned in tmp
    F77_FUNC(dgetri)((int *)&m, NULL, (int *)&m, NULL, (double *)&tmp, (int *)&work_sz, (int *)&info);
    nb=((int)tmp)/m; // optimal worksize is m*nb
#else
    nb=1; // min worksize is m
#endif /* SBA_LS_SCARCE_MEMORY */
  }
  work_sz=nb*m;
  tot_sz=(a_sz + work_sz)*sizeof(double) + ipiv_sz*sizeof(int); /* should be arranged in that order for proper doubles alignment */

  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(double *)malloc(buf_sz);
    if(!buf){
      fprintf(stderr, "memory allocation in sba_symat_invert_LU() failed!\n");
      exit(1);
    }
  }

  a=buf;
  work=a+a_sz;
  ipiv=(int *)(work+work_sz);

  /* store A (column major!) into a */
	for(i=0; i<m; ++i)
		for(j=i; j<m; ++j)
			a[i+j*m]=a[j+i*m]=A[i*m+j]; // copy A's upper part to a's upper & lower

  /* LU decomposition for A */
	F77_FUNC(dgetrf)((int *)&m, (int *)&m, a, (int *)&m, ipiv, (int *)&info);  
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dgetrf illegal in sba_symat_invert_LU()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular matrix A for dgetrf in sba_symat_invert_LU()\n");
			return 0;
		}
	}

  /* (A)^{-1} from LU */
	F77_FUNC(dgetri)((int *)&m, a, (int *)&m, ipiv, work, (int *)&work_sz, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dgetri illegal in sba_symat_invert_LU()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular matrix A for dgetri in sba_symat_invert_LU()\n");
			return 0;
		}
	}

	/* store (A)^{-1} in A's lower triangle */
	for(i=0; i<m; ++i)
		for(j=0; j<=i; ++j)
      A[i*m+j]=a[i+j*m];

	return 1;
}

/*
 * This function computes the inverse of a square symmetric positive definite 
 * matrix whose upper triangle is stored in A into its lower triangle using
 * Cholesky factorization
 *
 * The function returns 0 in case of error (e.g. A is not positive definite or singular),
 * 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_symat_invert_Chol(double *A, int m)
{
static double *buf=NULL;
static int buf_sz=0;

int a_sz, tot_sz;
register int i, j;
int info;
double *a;
   
  if(A==NULL){
    if(buf) free(buf);
    buf=NULL;
    buf_sz=0;

    return 1;
  }

  /* calculate required memory size */
  a_sz=m*m;
  tot_sz=a_sz; 

  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(double *)malloc(buf_sz*sizeof(double));
    if(!buf){
      fprintf(stderr, "memory allocation in sba_symat_invert_Chol() failed!\n");
      exit(1);
    }
  }

  a=(double *)buf;

  /* store A into a; A is assumed symmetric, hence no transposition is needed */
  for(i=0, j=a_sz; i<j; ++i)
    a[i]=A[i];

  /* Cholesky factorization for A; a's lower part corresponds to A's upper */
  //F77_FUNC(dpotrf)("L", (int *)&m, a, (int *)&m, (int *)&info);
  F77_FUNC(dpotf2)("L", (int *)&m, a, (int *)&m, (int *)&info);
  /* error treatment */
  if(info!=0){
    if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dpotrf in sba_symat_invert_Chol()\n", -info);
      exit(1);
    }
    else{
      fprintf(stderr, "LAPACK error: the leading minor of order %d is not positive definite,\nthe factorization could not be completed for dpotrf in sba_symat_invert_Chol()\n", info);
      return 0;
    }
  }

  /* (A)^{-1} from Cholesky */
  F77_FUNC(dpotri)("L", (int *)&m, a, (int *)&m, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dpotri illegal in sba_symat_invert_Chol()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "the (%d, %d) element of the factor U or L is zero, singular matrix A for dpotri in sba_symat_invert_Chol()\n", info, info);
			return 0;
		}
	}

	/* store (A)^{-1} in A's lower triangle. The lower triangle of the symmetric A^{-1} is in the lower triangle of a */
	for(i=0; i<m; ++i)
		for(j=0; j<=i; ++j)
      A[i*m+j]=a[i+j*m];

	return 1;
}

/*
 * This function computes the inverse of a symmetric indefinite 
 * matrix whose upper triangle is stored in A into its lower triangle
 * using LDLT factorization with the pivoting strategy of Bunch and Kaufman
 *
 * The function returns 0 in case of error (e.g. A is singular),
 * 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_symat_invert_BK(double *A, int m)
{
static double *buf=NULL;
static int buf_sz=0, nb=0;

int a_sz, ipiv_sz, work_sz, tot_sz;
register int i, j;
int info, *ipiv;
double *a, *work;
   
  if(A==NULL){
    if(buf) free(buf);
    buf=NULL;
    buf_sz=0;

    return 1;
  }

  /* calculate required memory size */
  ipiv_sz=m;
  a_sz=m*m;
  if(!nb){
#ifndef SBA_LS_SCARCE_MEMORY
    double tmp;

    work_sz=-1; // workspace query; optimal size is returned in tmp
    F77_FUNC(dsytrf)("L", (int *)&m, NULL, (int *)&m, NULL, (double *)&tmp, (int *)&work_sz, (int *)&info);
    nb=((int)tmp)/m; // optimal worksize is m*nb
#else
    nb=-1; // min worksize is 1
#endif /* SBA_LS_SCARCE_MEMORY */
  }
  work_sz=(nb!=-1)? nb*m : 1;
  work_sz=(work_sz>=m)? work_sz : m; /* ensure that work is at least m elements long, as required by dsytri */
  tot_sz=(a_sz + work_sz)*sizeof(double) + ipiv_sz*sizeof(int); /* should be arranged in that order for proper doubles alignment */

  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(double *)malloc(buf_sz);
    if(!buf){
      fprintf(stderr, "memory allocation in sba_symat_invert_BK() failed!\n");
      exit(1);
    }
  }

  a=buf;
  work=a+a_sz;
  ipiv=(int *)(work+work_sz);

  /* store A into a; A is assumed symmetric, hence no transposition is needed */
  for(i=0, j=a_sz; i<j; ++i)
    a[i]=A[i];

  /* LDLT factorization for A; a's lower part corresponds to A's upper */
	F77_FUNC(dsytrf)("L", (int *)&m, a, (int *)&m, ipiv, work, (int *)&work_sz, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dsytrf illegal in sba_symat_invert_BK()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular block diagonal matrix D for dsytrf in sba_symat_invert_BK() [D(%d, %d) is zero]\n", info, info);
			return 0;
		}
	}

  /* (A)^{-1} from LDLT */
  F77_FUNC(dsytri)("L", (int *)&m, a, (int *)&m, ipiv, work, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dsytri illegal in sba_symat_invert_BK()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "D(%d, %d)=0, matrix is singular and its inverse could not be computed in sba_symat_invert_BK()\n", info, info);
			return 0;
		}
	}

	/* store (A)^{-1} in A's lower triangle. The lower triangle of the symmetric A^{-1} is in the lower triangle of a */
	for(i=0; i<m; ++i)
		for(j=0; j<=i; ++j)
      A[i*m+j]=a[i+j*m];

	return 1;
}


#define __CG_LINALG_BLOCKSIZE           8

/* Dot product of two vectors x and y using loop unrolling and blocking.
 * see http://www.abarnett.demon.co.uk/tutorial.html
 */

inline static double dprod(const int n, const double *const x, const double *const y)
{
register int i, j1, j2, j3, j4, j5, j6, j7; 
int blockn;
register double sum0=0.0, sum1=0.0, sum2=0.0, sum3=0.0,
                sum4=0.0, sum5=0.0, sum6=0.0, sum7=0.0;

  /* n may not be divisible by __CG_LINALG_BLOCKSIZE, 
  * go as near as we can first, then tidy up.
  */ 
  blockn = (n / __CG_LINALG_BLOCKSIZE) * __CG_LINALG_BLOCKSIZE; 

  /* unroll the loop in blocks of `__CG_LINALG_BLOCKSIZE' */ 
  for(i=0; i<blockn; i+=__CG_LINALG_BLOCKSIZE){
            sum0+=x[i]*y[i];
    j1=i+1; sum1+=x[j1]*y[j1];
    j2=i+2; sum2+=x[j2]*y[j2];
    j3=i+3; sum3+=x[j3]*y[j3];
    j4=i+4; sum4+=x[j4]*y[j4];
    j5=i+5; sum5+=x[j5]*y[j5];
    j6=i+6; sum6+=x[j6]*y[j6];
    j7=i+7; sum7+=x[j7]*y[j7];
  } 

 /* 
  * There may be some left to do.
  * This could be done as a simple for() loop, 
  * but a switch is faster (and more interesting) 
  */ 

  if(i<n){ 
    /* Jump into the case at the place that will allow
    * us to finish off the appropriate number of items. 
    */ 

    switch(n - i){ 
      case 7 : sum0+=x[i]*y[i]; ++i;
      case 6 : sum1+=x[i]*y[i]; ++i;
      case 5 : sum2+=x[i]*y[i]; ++i;
      case 4 : sum3+=x[i]*y[i]; ++i;
      case 3 : sum4+=x[i]*y[i]; ++i;
      case 2 : sum5+=x[i]*y[i]; ++i;
      case 1 : sum6+=x[i]*y[i]; ++i;
    }
  } 

  return sum0+sum1+sum2+sum3+sum4+sum5+sum6+sum7;
}


/* Compute z=x+a*y for two vectors x and y and a scalar a; z can be one of x, y.
 * Similarly to the dot product routine, this one uses loop unrolling and blocking
 */

inline static void daxpy(const int n, double *const z, const double *const x, const double a, const double *const y)
{ 
register int i, j1, j2, j3, j4, j5, j6, j7; 
int blockn;

  /* n may not be divisible by __CG_LINALG_BLOCKSIZE, 
  * go as near as we can first, then tidy up.
  */ 
  blockn = (n / __CG_LINALG_BLOCKSIZE) * __CG_LINALG_BLOCKSIZE; 

  /* unroll the loop in blocks of `__CG_LINALG_BLOCKSIZE' */ 
  for(i=0; i<blockn; i+=__CG_LINALG_BLOCKSIZE){
            z[i]=x[i]+a*y[i];
    j1=i+1; z[j1]=x[j1]+a*y[j1];
    j2=i+2; z[j2]=x[j2]+a*y[j2];
    j3=i+3; z[j3]=x[j3]+a*y[j3];
    j4=i+4; z[j4]=x[j4]+a*y[j4];
    j5=i+5; z[j5]=x[j5]+a*y[j5];
    j6=i+6; z[j6]=x[j6]+a*y[j6];
    j7=i+7; z[j7]=x[j7]+a*y[j7];
  } 

 /* 
  * There may be some left to do.
  * This could be done as a simple for() loop, 
  * but a switch is faster (and more interesting) 
  */ 

  if(i<n){ 
    /* Jump into the case at the place that will allow
    * us to finish off the appropriate number of items. 
    */ 

    switch(n - i){ 
      case 7 : z[i]=x[i]+a*y[i]; ++i;
      case 6 : z[i]=x[i]+a*y[i]; ++i;
      case 5 : z[i]=x[i]+a*y[i]; ++i;
      case 4 : z[i]=x[i]+a*y[i]; ++i;
      case 3 : z[i]=x[i]+a*y[i]; ++i;
      case 2 : z[i]=x[i]+a*y[i]; ++i;
      case 1 : z[i]=x[i]+a*y[i]; ++i;
    }
  } 
}

/*
 * This function returns the solution of Ax = b where A is posititive definite,
 * based on the conjugate gradients method; see "An intro to the CG method" by J.R. Shewchuk, p. 50-51
 *
 * A is mxm, b, x are is mx1. Argument niter specifies the maximum number of 
 * iterations and eps is the desired solution accuracy. niter<0 signals that
 * x contains a valid initial approximation to the solution; if niter>0 then 
 * the starting point is taken to be zero. Argument prec selects the desired
 * preconditioning method as follows:
 * 0: no preconditioning
 * 1: jacobi (diagonal) preconditioning
 * 2: SSOR preconditioning
 * Argument iscolmaj specifies whether A is stored in column or row major order.
 *
 * The function returns 0 in case of error,
 * the number of iterations performed if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int sba_Axb_CG(double *A, double *B, double *x, int m, int niter, double eps, int prec, int iscolmaj)
{
static double *buf=NULL;
static int buf_sz=0;

register int i, j;
register double *aim;
int iter, a_sz, res_sz, d_sz, q_sz, s_sz, wk_sz, z_sz, tot_sz;
double *a, *res, *d, *q, *s, *wk, *z;
double delta0, deltaold, deltanew, alpha, beta, eps_sq=eps*eps;
register double sum;
int rec_res;

  if(A==NULL){
    if(buf) free(buf);
    buf=NULL;
    buf_sz=0;

    return 1;
  }

  /* calculate required memory size */
  a_sz=(iscolmaj)? m*m : 0;
	res_sz=m; d_sz=m; q_sz=m;
  if(prec!=SBA_CG_NOPREC){
    s_sz=m; wk_sz=m;
    z_sz=(prec==SBA_CG_SSOR)? m : 0;
  }
  else
    s_sz=wk_sz=z_sz=0;
 
	tot_sz=a_sz+res_sz+d_sz+q_sz+s_sz+wk_sz+z_sz;

  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(double *)malloc(buf_sz*sizeof(double));
    if(!buf){
		  fprintf(stderr, "memory allocation request failed in sba_Axb_CG()\n");
		  exit(1);
	  }
  }

  if(iscolmaj){ 
    a=buf;
    /* store A (row major!) into a */
    for(i=0; i<m; ++i)
      for(j=0, aim=a+i*m; j<m; ++j)
        aim[j]=A[i+j*m];
  }
  else a=A; /* no copying required */

	res=buf+a_sz;
	d=res+res_sz;
	q=d+d_sz;
  if(prec!=SBA_CG_NOPREC){
	  s=q+q_sz;
    wk=s+s_sz;
    z=(prec==SBA_CG_SSOR)? wk+wk_sz : NULL;

    for(i=0; i<m; ++i){ // compute jacobi (i.e. diagonal) preconditioners and save them in wk
      sum=a[i*m+i];
      if(sum>DBL_EPSILON || -sum<-DBL_EPSILON) // != 0.0
        wk[i]=1.0/sum;
      else
        wk[i]=1.0/DBL_EPSILON;
    }
  }
  else{
    s=res;
    wk=z=NULL;
  }

  if(niter>0){
	  for(i=0; i<m; ++i){ // clear solution and initialize residual vector:  res <-- B
		  x[i]=0.0;
      res[i]=B[i];
    }
  }
  else{
    niter=-niter;

	  for(i=0; i<m; ++i){ // initialize residual vector:  res <-- B - A*x
      for(j=0, aim=a+i*m, sum=0.0; j<m; ++j)
        sum+=aim[j]*x[j];
      res[i]=B[i]-sum;
    }
  }

  switch(prec){
    case SBA_CG_NOPREC:
      for(i=0, deltanew=0.0; i<m; ++i){
        d[i]=res[i];
        deltanew+=res[i]*res[i];
      }
      break;
    case SBA_CG_JACOBI: // jacobi preconditioning
      for(i=0, deltanew=0.0; i<m; ++i){
        d[i]=res[i]*wk[i];
        deltanew+=res[i]*d[i];
      }
      break;
    case SBA_CG_SSOR: // SSOR preconditioning; see the "templates" book, fig. 3.2, p. 44
      for(i=0; i<m; ++i){
        for(j=0, sum=0.0, aim=a+i*m; j<i; ++j)
          sum+=aim[j]*z[j];
        z[i]=wk[i]*(res[i]-sum);
      }

      for(i=m-1; i>=0; --i){
        for(j=i+1, sum=0.0, aim=a+i*m; j<m; ++j)
          sum+=aim[j]*d[j];
        d[i]=z[i]-wk[i]*sum;
      }
      deltanew=dprod(m, res, d);
      break;
    default:
      fprintf(stderr, "unknown preconditioning option %d in sba_Axb_CG\n", prec);
      exit(1);
  }

  delta0=deltanew;

	for(iter=1; deltanew>eps_sq*delta0 && iter<=niter; ++iter){
    for(i=0; i<m; ++i){ // q <-- A d
      aim=a+i*m;
/***
      for(j=0, sum=0.0; j<m; ++j)
        sum+=aim[j]*d[j];
***/
      q[i]=dprod(m, aim, d); //sum;
    }

/***
    for(i=0, sum=0.0; i<m; ++i)
      sum+=d[i]*q[i];
***/
    alpha=deltanew/dprod(m, d, q); // deltanew/sum;

/***
    for(i=0; i<m; ++i)
      x[i]+=alpha*d[i];
***/
    daxpy(m, x, x, alpha, d);

    if(!(iter%50)){
	    for(i=0; i<m; ++i){ // accurate computation of the residual vector
        aim=a+i*m;
/***
        for(j=0, sum=0.0; j<m; ++j)
          sum+=aim[j]*x[j];
***/
        res[i]=B[i]-dprod(m, aim, x); //B[i]-sum;
      }
      rec_res=0;
    }
    else{
/***
	    for(i=0; i<m; ++i) // approximate computation of the residual vector
        res[i]-=alpha*q[i];
***/
      daxpy(m, res, res, -alpha, q);
      rec_res=1;
    }

    if(prec){
      switch(prec){
      case SBA_CG_JACOBI: // jacobi
        for(i=0; i<m; ++i)
          s[i]=res[i]*wk[i];
        break;
      case SBA_CG_SSOR: // SSOR
        for(i=0; i<m; ++i){
          for(j=0, sum=0.0, aim=a+i*m; j<i; ++j)
            sum+=aim[j]*z[j];
          z[i]=wk[i]*(res[i]-sum);
        }

        for(i=m-1; i>=0; --i){
          for(j=i+1, sum=0.0, aim=a+i*m; j<m; ++j)
            sum+=aim[j]*s[j];
          s[i]=z[i]-wk[i]*sum;
        }
        break;
      }
    }

    deltaold=deltanew;
/***
	  for(i=0, sum=0.0; i<m; ++i)
      sum+=res[i]*s[i];
***/
    deltanew=dprod(m, res, s); //sum;

    /* make sure that we get around small delta that are due to
     * accumulated floating point roundoff errors
     */
    if(rec_res && deltanew<=eps_sq*delta0){
      /* analytically recompute delta */
	    for(i=0; i<m; ++i){
        for(j=0, aim=a+i*m, sum=0.0; j<m; ++j)
          sum+=aim[j]*x[j];
        res[i]=B[i]-sum;
      }
      deltanew=dprod(m, res, s);
    }

    beta=deltanew/deltaold;

/***
	  for(i=0; i<m; ++i)
      d[i]=s[i]+beta*d[i];
***/
    daxpy(m, d, s, beta, d);
  }

	return iter;
}

/*
 * This function computes the Cholesky decomposition of the inverse of a symmetric
 * (covariance) matrix A into B, i.e. B is s.t. A^-1=B^t*B and B upper triangular.
 * A and B can coincide
 *
 * The function returns 0 in case of error (e.g. A is singular),
 * 1 if successfull
 *
 * This function is often called repetitively to operate on matrices of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 *
 */
#if 0
int sba_mat_cholinv(double *A, double *B, int m)
{
static double *buf=NULL;
static int buf_sz=0, nb=0;

int a_sz, ipiv_sz, work_sz, tot_sz;
register int i, j;
int info, *ipiv;
double *a, *work;
   
  if(A==NULL){
    if(buf) free(buf);
    buf=NULL;
    buf_sz=0;

    return 1;
  }

  /* calculate the required memory size */
  ipiv_sz=m;
  a_sz=m*m;
  if(!nb){
#ifndef SBA_LS_SCARCE_MEMORY
    double tmp;

    work_sz=-1; // workspace query; optimal size is returned in tmp
    F77_FUNC(dgetri)((int *)&m, NULL, (int *)&m, NULL, (double *)&tmp, (int *)&work_sz, (int *)&info);
    nb=((int)tmp)/m; // optimal worksize is m*nb
#else
    nb=1; // min worksize is m
#endif /* SBA_LS_SCARCE_MEMORY */
  }
  work_sz=nb*m;
  tot_sz=(a_sz + work_sz)*sizeof(double) + ipiv_sz*sizeof(int); /* should be arranged in that order for proper doubles alignment */

  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(double *)malloc(buf_sz);
    if(!buf){
      fprintf(stderr, "memory allocation in sba_mat_cholinv() failed!\n");
      exit(1);
    }
  }

  a=buf;
  work=a+a_sz;
  ipiv=(int *)(work+work_sz);

  /* step 1: invert A */
  /* store A into a; A is assumed symmetric, hence no transposition is needed */
  for(i=0; i<m*m; ++i)
    a[i]=A[i];

  /* LU decomposition for A (Cholesky should also do) */
	F77_FUNC(dgetf2)((int *)&m, (int *)&m, a, (int *)&m, ipiv, (int *)&info);  
	//F77_FUNC(dgetrf)((int *)&m, (int *)&m, a, (int *)&m, ipiv, (int *)&info);  
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dgetf2/dgetrf illegal in sba_mat_cholinv()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular matrix A for dgetf2/dgetrf in sba_mat_cholinv()\n");
			return 0;
		}
	}

  /* (A)^{-1} from LU */
	F77_FUNC(dgetri)((int *)&m, a, (int *)&m, ipiv, work, (int *)&work_sz, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dgetri illegal in sba_mat_cholinv()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "singular matrix A for dgetri in sba_mat_cholinv()\n");
			return 0;
		}
	}

  /* (A)^{-1} is now in a (in column major!) */

  /* step 2: Cholesky decomposition of a: A^-1=B^t B, B upper triangular */
  F77_FUNC(dpotf2)("U", (int *)&m, a, (int *)&m, (int *)&info);
  /* error treatment */
  if(info!=0){
		if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dpotf2 in sba_mat_cholinv()\n", -info);
		  exit(1);
		}
		else{
			fprintf(stderr, "LAPACK error: the leading minor of order %d is not positive definite,\n%s\n", info,
						  "and the Cholesky factorization could not be completed in sba_mat_cholinv()");
			return 0;
		}
  }

  /* the decomposition is in the upper part of a (in column-major order!).
   * copying it to the lower part and zeroing the upper transposes
   * a in row-major order
   */
  for(i=0; i<m; ++i)
    for(j=0; j<i; ++j){
      a[i+j*m]=a[j+i*m];
      a[j+i*m]=0.0;
    }
  for(i=0; i<m*m; ++i)
    B[i]=a[i];

	return 1;
}
#endif

int sba_mat_cholinv(double *A, double *B, int m)
{
int a_sz;
register int i, j;
int info;
double *a;
   
  if(A==NULL){
    return 1;
  }

  a_sz=m*m;
  a=B; /* use B as working memory, result is produced in it */

  /* step 1: invert A */
  /* store A into a; A is assumed symmetric, hence no transposition is needed */
  for(i=0; i<a_sz; ++i)
    a[i]=A[i];

  /* Cholesky decomposition for A */
  F77_FUNC(dpotf2)("U", (int *)&m, a, (int *)&m, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dpotf2 illegal in sba_mat_cholinv()\n", -info);
			exit(1);
		}
		else{
			fprintf(stderr, "LAPACK error: the leading minor of order %d is not positive definite,\n%s\n", info,
						  "and the Cholesky factorization could not be completed in sba_mat_cholinv()");
			return 0;
		}
	}

  /* (A)^{-1} from Cholesky */
	F77_FUNC(dpotri)("U", (int *)&m, a, (int *)&m, (int *)&info);
	if(info!=0){
		if(info<0){
			fprintf(stderr, "argument %d of dpotri illegal in sba_mat_cholinv()\n", -info);
			exit(1);
		}
		else{
      fprintf(stderr, "the (%d, %d) element of the factor U or L is zero, singular matrix A for dpotri in sba_mat_cholinv()\n", info, info);
			return 0;
		}
	}

  /* (A)^{-1} is now in a (in column major!) */

  /* step 2: Cholesky decomposition of a: A^-1=B^t B, B upper triangular */
  F77_FUNC(dpotf2)("U", (int *)&m, a, (int *)&m, (int *)&info);
  /* error treatment */
  if(info!=0){
		if(info<0){
      fprintf(stderr, "LAPACK error: illegal value for argument %d of dpotf2 in sba_mat_cholinv()\n", -info);
		  exit(1);
		}
		else{
			fprintf(stderr, "LAPACK error: the leading minor of order %d is not positive definite,\n%s\n", info,
						  "and the Cholesky factorization could not be completed in sba_mat_cholinv()");
			return 0;
		}
  }

  /* the decomposition is in the upper part of a (in column-major order!).
   * copying it to the lower part and zeroing the upper transposes
   * a in row-major order
   */
  for(i=0; i<m; ++i)
    for(j=0; j<i; ++j){
      a[i+j*m]=a[j+i*m];
      a[j+i*m]=0.0;
    }

	return 1;
}
