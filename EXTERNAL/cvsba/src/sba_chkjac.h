/////////////////////////////////////////////////////////////////////////////////
//// 
////  Prototypes and definitions for verification routines for the jacobians
////  employed in the expert & simple drivers for sparse bundle adjustment
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

#ifndef _SBA_CHKJAC_H_
#define _SBA_CHKJAC_H_

#ifdef __cplusplus
extern "C" {
#endif

#if 0
/* simple driver jacobians */
extern void sba_motstr_chkjac(
      void (*proj)(int jj, int ii, double *aj, double *bi, double *xij, void *adata),
      void (*projac)(int jj, int ii, double *aj, double *bi, double *Aij, double *Bij, void *adata),
      double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata);

extern void sba_mot_chkjac(
      void (*proj)(int jj, int ii, double *aj, double *xij, void *adata),
      void (*projac)(int jj, int ii, double *aj, double *Aij, void *adata),
      double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata);

extern void sba_str_chkjac(
      void (*proj)(int jj, int ii, double *bi, double *xij, void *adata),
      void (*projac)(int jj, int ii, double *bi, double *Bij, void *adata),
      double *aj, double *bi, int jj, int ii, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata);
#endif /* 0 */

/* expert driver jacobians */
extern void sba_motstr_chkjac_x(
      void (*func)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata),
      void (*jacf)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata),
      double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, int ncon, int mcon, int cnp, int pnp, int mnp, void *func_adata, void *jac_adata);

extern void sba_mot_chkjac_x(
      void (*func)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata),
      void (*jacf)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata),
      double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, int mcon, int cnp, int mnp, void *func_adata, void *jac_adata);

extern void sba_str_chkjac_x(
      void (*func)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata),
      void (*jacf)(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata),
      double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, int ncon, int pnp, int mnp, void *func_adata, void *jac_adata);


#ifdef __cplusplus
}
#endif

#endif /* _SBA_CHKJAC_H_ */
