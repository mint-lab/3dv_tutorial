/////////////////////////////////////////////////////////////////////////////////
//// 
////  CRS sparse matrices manipulation routines
////  Copyright (C) 2004-2008 Manolis Lourakis (lourakis at ics forth gr)
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

#include "sba.h"

static void sba_crsm_print(struct sba_crsm *sm, FILE *fp);
static void sba_crsm_build(struct sba_crsm *sm, int *m, int nr, int nc);

/* allocate a sparse CRS matrix */
void sba_crsm_alloc(struct sba_crsm *sm, int nr, int nc, int nnz)
{
int msz;

  sm->nr=nr;
  sm->nc=nc;
  sm->nnz=nnz;
  msz=2*nnz+nr+1;
  sm->val=(int *)malloc(msz*sizeof(int));  /* required memory is allocated in a single step */
  if(!sm->val){
    fprintf(stderr, "SBA: memory allocation request failed in sba_crsm_alloc() [nr=%d, nc=%d, nnz=%d]\n", nr, nc, nnz);
    exit(1);
  }
  sm->colidx=sm->val+nnz;
  sm->rowptr=sm->colidx+nnz;
}

/* free a sparse CRS matrix */
void sba_crsm_free(struct sba_crsm *sm)
{
  sm->nr=sm->nc=sm->nnz=-1;
  free(sm->val);
  sm->val=sm->colidx=sm->rowptr=NULL;
}

static void sba_crsm_print(struct sba_crsm *sm, FILE *fp)
{
register int i;

  fprintf(fp, "matrix is %dx%d, %d non-zeros\nval: ", sm->nr, sm->nc, sm->nnz);
  for(i=0; i<sm->nnz; ++i)
    fprintf(fp, "%d ", sm->val[i]);
  fprintf(fp, "\ncolidx: ");
  for(i=0; i<sm->nnz; ++i)
    fprintf(fp, "%d ", sm->colidx[i]);
  fprintf(fp, "\nrowptr: ");
  for(i=0; i<=sm->nr; ++i)
    fprintf(fp, "%d ", sm->rowptr[i]);
  fprintf(fp, "\n");
}

/* build a sparse CRS matrix from a dense one. intended to serve as an example for sm creation */
static void sba_crsm_build(struct sba_crsm *sm, int *m, int nr, int nc)
{
int nnz;
register int i, j, k;

  /* count nonzeros */
  for(i=nnz=0; i<nr; ++i)
    for(j=0; j<nc; ++j)
      if(m[i*nc+j]!=0) ++nnz;

  sba_crsm_alloc(sm, nr, nc, nnz);

  /* fill up the sm structure */
  for(i=k=0; i<nr; ++i){
    sm->rowptr[i]=k;
    for(j=0; j<nc; ++j)
      if(m[i*nc+j]!=0){
        sm->val[k]=m[i*nc+j];
        sm->colidx[k++]=j;
      }
  }
  sm->rowptr[nr]=nnz;
}

/* returns the index of the (i, j) element. No bounds checking! */
int sba_crsm_elmidx(struct sba_crsm *sm, int i, int j)
{
register int low, high, mid, diff;

  low=sm->rowptr[i];
  high=sm->rowptr[i+1]-1;

  /* binary search for finding the element at column j */
  while(low<=high){
    /* following early termination test seems to actually slow down the search */
    //if(j<sm->colidx[low] || j>sm->colidx[high]) return -1; /* not found */
    
    /* mid=low+((high-low)>>1) ensures no index overflows */
    mid=(low+high)>>1; //(low+high)/2;
    diff=j-sm->colidx[mid];
    if(diff<0)
      high=mid-1;
    else if(diff>0)
      low=mid+1;
    else
      return mid;
  }

  return -1; /* not found */
}

/* similarly to sba_crsm_elmidx() above, returns the index of the (i, j) element using the
 * fact that the index of element (i, jp) was previously found to be jpidx. This can be
 * slightly faster than sba_crsm_elmidx(). No bounds checking!
 */
int sba_crsm_elmidxp(struct sba_crsm *sm, int i, int j, int jp, int jpidx)
{
register int low, high, mid, diff;

  diff=j-jp;
  if(diff>0){
    low=jpidx+1;
    high=sm->rowptr[i+1]-1;
  }
  else if(diff==0)
    return jpidx;
  else{ /* diff<0 */
    low=sm->rowptr[i];
    high=jpidx-1;
  }

  /* binary search for finding the element at column j */
  while(low<=high){
    /* following early termination test seems to actually slow down the search */
    //if(j<sm->colidx[low] || j>sm->colidx[high]) return -1; /* not found */
    
    /* mid=low+((high-low)>>1) ensures no index overflows */
    mid=(low+high)>>1; //(low+high)/2;
    diff=j-sm->colidx[mid];
    if(diff<0)
      high=mid-1;
    else if(diff>0)
      low=mid+1;
    else
      return mid;
  }

  return -1; /* not found */
}

/* returns the number of nonzero elements in row i and
 * fills up the vidxs and jidxs arrays with the val and column
 * indexes of the elements found, respectively.
 * vidxs and jidxs are assumed preallocated and of max. size sm->nc
 */
int sba_crsm_row_elmidxs(struct sba_crsm *sm, int i, int *vidxs, int *jidxs)
{
register int j, k;

  for(j=sm->rowptr[i], k=0; j<sm->rowptr[i+1]; ++j, ++k){
    vidxs[k]=j;
    jidxs[k]=sm->colidx[j];
  }

  return k;
}

/* returns the number of nonzero elements in col j and
 * fills up the vidxs and iidxs arrays with the val and row
 * indexes of the elements found, respectively.
 * vidxs and iidxs are assumed preallocated and of max. size sm->nr
 */
int sba_crsm_col_elmidxs(struct sba_crsm *sm, int j, int *vidxs, int *iidxs)
{
register int *nextrowptr=sm->rowptr+1;
register int i, l;
register int low, high, mid, diff;

  for(i=l=0; i<sm->nr; ++i){
    low=sm->rowptr[i];
    high=nextrowptr[i]-1;

    /* binary search attempting to find an element at column j */
    while(low<=high){
      //if(j<sm->colidx[low] || j>sm->colidx[high]) break; /* not found */

      mid=(low+high)>>1; //(low+high)/2;
      diff=j-sm->colidx[mid];
      if(diff<0)
        high=mid-1;
      else if(diff>0)
        low=mid+1;
      else{ /* found */
        vidxs[l]=mid;
        iidxs[l++]=i;
        break;
      }
    }
  }

  return l;
}

/* a more straighforward (but slower) implementation of the above function */
/***
int sba_crsm_col_elmidxs(struct sba_crsm *sm, int j, int *vidxs, int *iidxs)
{
register int i, k, l;

  for(i=l=0; i<sm->nr; ++i)
    for(k=sm->rowptr[i]; k<sm->rowptr[i+1]; ++k)
      if(sm->colidx[k]==j){
        vidxs[l]=k;
        iidxs[l++]=i;
      }

  return l;
}
***/

#if 0
/* returns 1 if there exists a row i having columns j and k,
 * i.e. a row i s.t. elements (i, j) and (i, k) are nonzero;
 * 0 otherwise
 */ 
int sba_crsm_common_row(struct sba_crsm *sm, int j, int k)
{
register int i, low, high, mid, diff;

  if(j==k) return 1;

  for(i=0; i<sm->nr; ++i){
    low=sm->rowptr[i];
    high=sm->rowptr[i+1]-1;
    if(j<sm->colidx[low] || j>sm->colidx[high] || /* j not found */
       k<sm->colidx[low] || k>sm->colidx[high])   /* k not found */
      continue;

    /* binary search for finding the element at column j */
    while(low<=high){
      mid=(low+high)>>1; //(low+high)/2;
      diff=j-sm->colidx[mid];
      if(diff<0)
        high=mid-1;
      else if(diff>0)
        low=mid+1;
      else
        goto jfound;
    }

    continue; /* j not found */

jfound:
    if(j>k){
      low=sm->rowptr[i];
      high=mid-1;
    }
    else{
      low=mid+1;
      high=sm->rowptr[i+1]-1;
    }

    if(k<sm->colidx[low] || k>sm->colidx[high]) continue; /* k not found */

    /* binary search for finding the element at column k */
    while(low<=high){
      mid=(low+high)>>1; //(low+high)/2;
      diff=k-sm->colidx[mid];
      if(diff<0)
        high=mid-1;
      else if(diff>0)
        low=mid+1;
      else /* found */
        return 1;
    }
  }

  return 0;
}
#endif


#if 0

/* sample code using the above routines */

main()
{
int mat[7][6]={
    {10, 0, 0, 0, -2, 0},
    {3,  9, 0, 0,  0, 3},
    {0,  7, 8, 7,  0, 0},
    {3,  0, 8, 7,  5, 0},
    {0,  8, 0, 9,  9, 13},
    {0,  4, 0, 0,  2, -1},
    {3,  7, 0, 9,  2, 0}
};

struct sba_crsm sm;
int i, j, k, l;
int vidxs[7], /* max(6, 7) */
    jidxs[6], iidxs[7];


  sba_crsm_build(&sm, mat[0], 7, 6);
  sba_crsm_print(&sm, stdout);

  for(i=0; i<7; ++i){
    for(j=0; j<6; ++j)
      printf("%3d ", ((k=sba_crsm_elmidx(&sm, i, j))!=-1)? sm.val[k] : 0);
    printf("\n");
  }

  for(i=0; i<7; ++i){
    k=sba_crsm_row_elmidxs(&sm, i, vidxs, jidxs);
    printf("row %d\n", i);
    for(l=0; l<k; ++l){
      j=jidxs[l];
      printf("%d %d  ", j, sm.val[vidxs[l]]); 
    }
    printf("\n");
  }

  for(j=0; j<6; ++j){
    k=sba_crsm_col_elmidxs(&sm, j, vidxs, iidxs);
    printf("col %d\n", j);
    for(l=0; l<k; ++l){
      i=iidxs[l];
      printf("%d %d  ", i, sm.val[vidxs[l]]); 
    }
    printf("\n");
  }

  sba_crsm_free(&sm);
}
#endif
