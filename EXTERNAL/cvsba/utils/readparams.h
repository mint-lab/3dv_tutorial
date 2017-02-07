#ifndef _READPARAMS_H_
#define _READPARAMS_H_


#define FULLQUATSZ     4

#ifdef __cplusplus /* If this is a C++ compiler, use C linkage */
extern "C" {
#endif

extern void readInitialSBAEstimate(char *camsfname, char *ptsfname, int cnp, int pnp, int mnp, 
                                   void (*infilter)(double *pin, int nin, double *pout, int nout), int cnfp,
                                   int *ncams, int *n3Dpts, int *n2Dprojs,
                                   double **motstruct, double **initrot, double **imgpts, double **covimgpts, char **vmask);

extern void readCalibParams(char *fname, double ical[9]);
extern int readNumParams(char *fname);

extern void printSBAMotionData(FILE *fp, double *motstruct, int ncams, int cnp,
                               void (*outfilter)(double *pin, int nin, double *pout, int nout), int cnop);
extern void printSBAStructureData(FILE *fp, double *motstruct, int ncams, int n3Dpts, int cnp, int pnp);
extern void printSBAData(FILE *fp, double *motstruct, int cnp, int pnp, int mnp, 
                         void (*outfilter)(double *pin, int nin, double *pout, int nout), int cnop,
                         int ncams, int n3Dpts, double *imgpts, int n2Dprojs, char *vmask);

extern void saveSBAStructureDataAsPLY(char *fname, double *motstruct, int ncams, int n3Dpts, int cnp, int pnp, int withrgb);

#ifdef __cplusplus /* If this is a C++ compiler, end C linkage */
}
#endif

#endif /* _READPARAMS_H_ */
