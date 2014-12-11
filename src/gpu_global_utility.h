/**
 * Utility.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef UTILITY_H_
#define UTILITY_H_

typedef float float_point; //this type is for facilitating interchangeability of float and double

extern float_point gfPCost;	//cost for positive samples in training SVM model (i.e., error tolerance)
extern float_point gfNCost;	//cost for negative samples in training SVM model
extern float_point gfGamma;

extern int gNTest;
extern int gnNumofThread;

/* a set of parameters which looks like a grid*/
struct grid
{
	float_point *pfGamma;
	int nNumofGamma;
	float_point *pfCost;
	int nNumofC;
};

/* */
struct data_info
{
	int *pnLabel;					//labels for each training sample
	int nNumofExample;				//the number of training samples
	int nNumofDim;					//the number of dimensions
};


struct svm_param
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	float_point gamma;	/* for poly/rbf/sigmoid */
	float_point coef0;	/* for poly/sigmoid */

	/* these are for training only */
	float_point cache_size; /* in MB */
	float_point eps;	/* stopping criteria */
	float_point C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	float_point* weight;		/* for C_SVC */
	float_point nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	float_point p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

struct svm_model
{
	struct svm_param param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	//struct svm_node **SV;		/* SVs (SV[l]) */
	float_point **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	float_point *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	float_point *probA;		/* pariwise probability information */
	float_point *probB;

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */

	//new content
	float_point* supportVectors;	//support vector data
	int nDimension;		//data dimension
	int nNumofSV;		//total number of support vectors
	int *pnIndexofSV;	//index of support vectors
};

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#define Ceil(a, b) (a%b!=0)?((a/b)+1):(a/b)

//#define CUDA_KERNEL_DIM(a, b)  <<< a, b >>>

//get the size of free GPU memory in the form of float point
int GetFreeGPUMem();

#endif /* UTILITY_H_ */
