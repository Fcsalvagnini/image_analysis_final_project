#include "ift.h"

/* 
   Project 02, course MO445, Prof. Alexandre Falcao. 

   This code crops a region of interest (ROI) at the center of the
   source image and finds the best match between source and the
   corresponding ROI in the target image. Both regions of interest,
   source and target, are saved in the output folder for a new
   comparison using neural networks, which should indicate whether or
   not the images come from the same class. 

   Your task is to play with the parameters of this code (i.e.,
   CROP_SIZE and the deltas in the parameter file of the MSPS
   registration algorithm), with the architecture of the neural
   network in scripts/siemeseNN.py, and with the cost function of that
   network to maximize performance in the identification of
   individuals by fingerprint images.

*/ 

/* you may uncomment this line to verify extra information about the process */

//#define DEBUG 1

#define CROP_SIZE 300 /* size of the ROI in the source image: 300 x
			 300. You may change it to create W x H
			 ROIs. */

/* Data structure used by MSPS -- multi-scale parameter search -- a
   method to put a source image in the coordinate space of a target
   one (or vice-versa).

   LINK --

   https://www.tandfonline.com/doi/abs/10.1080/21681163.2015.1029643?journalCode=tciv20

 Guilherme C.S. Ruppert, Giovani Chiachia, Felipe P.G. Bergo, Fernanda
 O. Favretto, Clarissa L. Yasuda, Anderson Rocha & Alexandre X. Falcão
 (2017) Medical image registration based on watershed transform from
 greyscale marker and multi-scale parameter search, Computer Methods
 in Biomechanics and Biomedical Engineering: Imaging & Visualization,
 5:2, 138-156, DOI: 10.1080/21681163.2015.1029643

*/

typedef struct _fp_match_problem {
  iftImage  *target; /* target image to compute costs */
  iftImage  *source; /* source image */
  iftSet    *S;      /* set with points from the source image */
  iftPoint   center; /* center of the source points */
  int        Imax;   /* maximum intensity of the target image */
  char      *source_basename; /* basename of the source image */ 
  char      *target_basename; /* basename of the target image */ 
} FPMatchProblem;

/* Functions used by this code */

FPMatchProblem *CreateFPMatchProblem(iftImage *source, iftImage *target,
				     char *source_basename, char *target_basename);
void            DestroyFPMatchProblem(FPMatchProblem **fpmp);
float           FPMatchFitness(void *problem, float *theta);
iftMSPS        *InitMSPS(char *deltafile, FPMatchProblem *fpmp);
iftImage       *CropSourceImage(iftImage *source, int cropsize);
void            Align_and_SaveData(FPMatchProblem *fpmp, float *theta,
				   int line, char *out_dir);


/* Creates a fingerprint matching problem for the MSPS algorithm */

FPMatchProblem *CreateFPMatchProblem(iftImage *source, iftImage *target,
				     char *source_basename,
				     char *target_basename)
{
  FPMatchProblem *fpmp = (FPMatchProblem *)calloc(1,sizeof(FPMatchProblem));
  fpmp->target   = target;
  fpmp->source   = source;
  fpmp->Imax     = iftMax(iftMaximumValue(target),iftMaximumValue(source));
  fpmp->S        = NULL;
  fpmp->source_basename = source_basename;
  fpmp->target_basename = target_basename;
  
  for (int p=0; p < source->n; p++)
    iftInsertSet(&fpmp->S,p);

  fpmp->center.x = source->xsize/2;
  fpmp->center.y = source->ysize/2;
  
  return(fpmp);
}

/* Destroys the matching problem */

void DestroyFPMatchProblem(FPMatchProblem **fpmp)
{
  FPMatchProblem *aux = *fpmp;

  if (aux != NULL) {
    iftFree(aux);
    fpmp = NULL;
  }
}

/* Fitness function used by the MSPS algorithm to find the best match
   between the source ROI in the target image */

float FPMatchFitness(void *problem, float *theta)
{
  float cost           = 0;
  FPMatchProblem *fpmp = (FPMatchProblem *)problem;
  iftVector trans[2];
  iftMatrix *M[6];

  /* This limits image scaling between [0.90,1.10]. You may change
     this interval. */
  
  if (theta[3] > 1.1)
    theta[3] = 1.1;
  if (theta[3] < 0.9) 
    theta[3] = 0.90;

  /* sequence of image transformations to place the source ROI in
     different parts of the target image */
  
  trans[0].z = trans[1].z = 0;
  
  trans[0].x = -fpmp->center.x; trans[0].y  = -fpmp->center.y; 
  M[0]       = iftTranslationMatrix(trans[0]);
  M[1]       = iftScaleMatrix(theta[3],theta[3],theta[3]);
  M[2]       = iftMultMatrices(M[1],M[0]);
  M[3]       = iftRotationMatrix(IFT_AXIS_Z, theta[2]);
  M[4]       = iftMultMatrices(M[3],M[2]);
  trans[1].x = theta[0] + fpmp->target->xsize/2;
  trans[1].y = theta[1] + fpmp->target->ysize/2;
  M[5]       = iftTranslationMatrix(trans[1]);
  
  iftMatrix *T = iftMultMatrices(M[5],M[4]);

  for (int i=0; i < 6; i++)
    iftDestroyMatrix(&M[i]);

  /* Apply the transformation to put source and target ROIs in the
     same coordinate space. If the DEBUG flag defined, the code will
     output source and target ROIs in the same image using a color
     coding to indicate the quality of the matching. When the DEBUG
     flag is undefined, it will simply compute the cost of the
     matching. */ 
  
#ifdef DEBUG
  
  iftImage *temp = iftCopyImage(fpmp->target);
  iftSetCbCr(temp,(fpmp->Imax+1)/2);
  iftColor RGB, YCbCr, rgb, ycbcr; 

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr = iftRGBtoYCbCr(RGB,255);
  iftAdjRel *A = iftCircular(1.0);
  
  iftSet *S = fpmp->S;
  int npts  = 0; 
  while (S != NULL) {
    int p      = S->elem;
    iftVoxel u = iftGetVoxelCoord(fpmp->source,p);
    iftPoint P1, P2;
    P1.x = u.x; P1.y = u.y; P1.z = 0;
    P2   = iftTransformPoint(T,P1);
    iftVoxel v;
    v.x  = iftRound(P2.x); v.y  = iftRound(P2.y); v.z = 0;
    if (iftValidVoxel(fpmp->target,v)){
      int q = iftGetVoxelIndex(fpmp->target,v);
      cost += fabs(fpmp->target->val[q]-fpmp->source->val[p]);
      rgb.val[0]   = fpmp->source->val[p];
      rgb.val[1]   = (fpmp->source->val[p]+fpmp->target->val[q])/2;
      rgb.val[2]   = fpmp->target->val[q];
      ycbcr        = iftRGBtoYCbCr(rgb,255);
      temp->val[q] = ycbcr.val[0];
      temp->Cb[q]  = ycbcr.val[1];
      temp->Cr[q]  = ycbcr.val[2];	
      if ((u.x == 0)||(u.y==0)||(u.x==fpmp->source->xsize-1)||
	  (u.y==fpmp->source->ysize-1)){
	iftDrawPoint(temp,v,YCbCr,A,255);
      }	
    } else {
      cost += fpmp->Imax;
    }
    S = S->next;
    npts++;
  }

  printf("%f %f %f %f %f\n",cost/npts,theta[0],theta[1],theta[2],theta[3]); 
  
  char filename[200];
  sprintf(filename,"%s_%s_%f_%f_%f_%f.png",fpmp->source_basename,fpmp->target_basename,theta[0],theta[1],theta[2],theta[3]);
  iftWriteImageByExt(temp,filename);
  iftDestroyImage(&temp);  
  iftDestroyAdjRel(&A);
  
#else
  
  iftSet *S = fpmp->S;
  int npts  = 0; 
  while (S != NULL) {
    int p      = S->elem;
    iftVoxel u = iftGetVoxelCoord(fpmp->source,p);
    iftPoint P1, P2;
    P1.x = u.x; P1.y = u.y; P1.z = 0;
    P2   = iftTransformPoint(T,P1);
    iftVoxel v;
    v.x  = iftRound(P2.x); v.y  = iftRound(P2.y); v.z = 0;
    if (iftValidVoxel(fpmp->target,v)){
      int q = iftGetVoxelIndex(fpmp->target,v);
      cost += fabs(fpmp->target->val[q]-fpmp->source->val[p]);
    } else {
      cost += fpmp->Imax;
    }
    S = S->next;
    npts++;
  }


#endif
  
  iftDestroyMatrix(&T);
  return(cost/npts);
}



iftImage *CropSourceImage(iftImage *source, int cropsize)
{
  iftImage  *img  = iftCopyImage(source);
  iftAdjRel *A    = iftCircular(3.5);
  iftImage  *bin  = iftBelowAdaptiveThreshold(img, NULL, A, 0.99, 1, 255);
  iftDestroyAdjRel(&A);
  
  iftVoxel  pos;
  iftBoundingBox bb = iftMinBoundingBox(bin, &pos);
  iftDestroyImage(&bin);
  
  pos.x      = (bb.begin.x+bb.end.x)/2; 
  pos.y      = (bb.begin.y+bb.end.y)/2;
  int sz     = iftMax(iftMax(iftMax(cropsize/2-pos.x,cropsize/2-pos.y),
			     ((pos.x + cropsize/2)-(img->xsize-1))),
		      ((pos.y + cropsize/2)-(img->ysize-1)));

  if (sz > 0){
    iftImage *aux = iftAddFrame(img,sz,255);
    iftDestroyImage(&img);
    img    = aux;
    pos.x += sz;
    pos.y += sz;
  }
	
  bb.begin.x = pos.x - cropsize/2;
  bb.begin.y = pos.y - cropsize/2;
  bb.end.x   = pos.x + cropsize/2 - 1;
  bb.end.y   = pos.y + cropsize/2 - 1;
  iftImage *crop = iftExtractROI(img,bb);
  iftDestroyImage(&img);

  return(crop);
}

void Align_and_SaveData(FPMatchProblem *fpmp, float *theta, int line,
			char *out_dir)
{
  iftVector trans[2];
  iftMatrix *M[6];
  char filename[200];
  
#ifdef DEBUG  
  printf("Best %f %f %f %f\n",theta[0],theta[1],theta[2],theta[3]); 
#endif
  
  trans[0].z = trans[1].z = 0;

  trans[0].x = -fpmp->center.x; trans[0].y  = -fpmp->center.y; 
  M[0]       = iftTranslationMatrix(trans[0]);
  M[1]       = iftScaleMatrix(theta[3],theta[3],theta[3]);
  M[2]       = iftMultMatrices(M[1],M[0]);
  M[3]       = iftRotationMatrix(IFT_AXIS_Z, theta[2]);
  M[4]       = iftMultMatrices(M[3],M[2]);
  trans[1].x = theta[0] + fpmp->target->xsize/2;
  trans[1].y = theta[1] + fpmp->target->ysize/2;
  M[5]       = iftTranslationMatrix(trans[1]);
  
  iftMatrix *T = iftMultMatrices(M[5],M[4]);

  for (int i=0; i < 6; i++)
    iftDestroyMatrix(&M[i]);

  iftImage *aligned_target =
    iftCreateImage(fpmp->source->xsize,fpmp->source->ysize,fpmp->source->zsize);

#ifdef DEBUG
  
  iftSetCbCr(aligned_target, (fpmp->Imax+1)/2);

  iftVoxel u, v;
  iftPoint P1, P2;
  iftColor RGB, YCbCr; 
  
  v.z = u.z = P1.z = P2.z = 0;
  for (v.y = 0; v.y < aligned_target->ysize; v.y++){
    for (v.x = 0; v.x < aligned_target->xsize; v.x++) {
      int q    = iftGetVoxelIndex(aligned_target, v);
      P1.x = v.x;
      P1.y = v.y;
      P2   = iftTransformPoint(T, P1);
      u.x  = iftRound(P2.x);
      u.y  = iftRound(P2.y);
      if (iftValidVoxel(fpmp->target, u)) {
	int p = iftGetVoxelIndex(fpmp->target, u);
	RGB.val[0] = fpmp->target->val[p];
	RGB.val[1] = (fpmp->target->val[p]+fpmp->source->val[q])/2;
	RGB.val[2] = fpmp->source->val[q];
	YCbCr = iftRGBtoYCbCr(RGB,255); 
	aligned_target->val[q] = YCbCr.val[0];
	aligned_target->Cb[q]  = YCbCr.val[1];
	aligned_target->Cr[q]  = YCbCr.val[2];
      }
    }
  }
  sprintf(filename,"%s/%s_%s.png",out_dir,fpmp->source_basename,fpmp->target_basename);  
  iftWriteImageByExt(aligned_target,filename);
  iftDestroyImage(&aligned_target);

#else

  iftVoxel u, v;
  iftPoint P1, P2;
  
  v.z = u.z = P1.z = P2.z = 0;
  for (v.y = 0; v.y < aligned_target->ysize; v.y++){
    for (v.x = 0; v.x < aligned_target->xsize; v.x++) {
      int q    = iftGetVoxelIndex(aligned_target, v);
      P1.x = v.x;
      P1.y = v.y;
      P2   = iftTransformPoint(T, P1);
      u.x  = iftRound(P2.x);
      u.y  = iftRound(P2.y);
      if (iftValidVoxel(fpmp->target, u)) {
	int p = iftGetVoxelIndex(fpmp->target, u);
	aligned_target->val[q] = fpmp->target->val[p];
      }
    }
  }
  
  sprintf(filename,"%s/%s_%d.png",out_dir,fpmp->source_basename,line);
  iftWriteImageByExt(fpmp->source,filename);
  sprintf(filename,"%s/%s_%d.png",out_dir,fpmp->target_basename,line);
  iftWriteImageByExt(aligned_target,filename);
  iftDestroyImage(&aligned_target);

#endif  

  iftDestroyMatrix(&T);
  
}

iftMSPS *InitMSPS(char *deltafile, FPMatchProblem *fpmp)
{
  iftMSPS *msps;
  FILE *fp = fopen(deltafile,"r");
  int n, m;

  fscanf(fp,"%d %d",&n,&m);

  msps = iftCreateMSPS(n,m,FPMatchFitness,fpmp);
  
  for (int parameter=0; parameter < n; parameter++) {
    for (int scale=0; scale < m; scale++) {
      fscanf(fp,"%f",
	     &msps->delta->val[iftGetMatrixIndex(msps->delta,parameter,scale)]);
    }
  }


#ifdef DEBUG
  for (int parameter=0; parameter < n; parameter++) {
    printf("parameter %d\n",parameter+1);
    for (int scale=0; scale < m; scale++) {
      printf("%f ",msps->delta->val[iftGetMatrixIndex(msps->delta,parameter,scale)]);
    }
    printf("\n");
  }
#endif
  
  fclose(fp);

  return(msps);
}
  
int main(int argc, char **argv) {

  timer *tstart=NULL;
  char   filename[200];
  
  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 5) {
    printf("project02 <P1> <P2> <P3> <P4>\n");
    printf("P1: folder with all fingerprint images\n");
    printf("P2: comparison file .txt\n");
    printf("P3: delta file (deltas.txt)\n");
    printf("P4: folder with registered images\n");
    exit(0);
  }

  tstart = iftTic();

  FILE *comp    = fopen(argv[2],"r");
  int ncomps;
  fscanf(comp,"%d",&ncomps);
  char *out_dir = argv[4];
  iftMakeDir(out_dir);
  FILE *score   = fopen("scores.txt","w");
  FILE *newcomp = fopen("compare_new.txt","w");
  
  for (int i = 0; i < ncomps; i++) {

    /* read images for comparison */

    char file1[20], file2[20];
    fscanf(comp,"%s %s",file1,file2);
    sprintf(filename,"%s/%s",argv[1],file1);
    iftImage *source        = iftReadImageByExt(filename);
    sprintf(filename,"%s/%s",argv[1],file2);
    iftImage *target        = iftReadImageByExt(filename);    
    char *source_basename   = iftFilename(file1,".png");
    char *target_basename   = iftFilename(file2,".png");

    fprintf(newcomp,"%s_%d.png %s_%d.png\n",source_basename,i+1,
	    target_basename,i+1);
    
    /* crop source image */

    iftImage *aux = CropSourceImage(source, CROP_SIZE);
    iftDestroyImage(&source);
    source = aux;
    
    /* align source and target, returning the matching score and saving
       the aligned images in the output folder */
    
    FPMatchProblem *fpmp  = CreateFPMatchProblem(source, target, source_basename, target_basename);   
    iftMSPS *msps         = InitMSPS(argv[3], fpmp);
    
    fprintf(score,"%.4f %s %s\n",iftMSPSMin(msps),file1,file2);
    
     /* visualize the alignment of target in the coordinate space of source
       with the skeleton of source on the top of it */ 

    Align_and_SaveData(fpmp, msps->theta, i+1, out_dir);

    /* free memory */
    
    iftDestroyMSPS(&msps);
    DestroyFPMatchProblem(&fpmp);
    iftFree(source_basename);
    iftFree(target_basename);
    iftDestroyImage(&source);
    iftDestroyImage(&target);
  }

  fclose(comp);
  fclose(score);
  fclose(newcomp);
  
  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return 0;
}



