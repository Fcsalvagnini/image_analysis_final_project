#include "ift.h"

/*
   Project 01, course MO445, Prof. Alexandre Falcao.

   This code crops a region of interest containing a fingerprint with
   as minimum as possible surrounding noise.

   The code contains three functions iftAsfCOBin(), iftCloseBasins(),
   iftErodeBin() that are implemented by the Image Foresting
   Transform. Your task is to substitute them by the corresponding
   functions implemented by you. In order to do that, you should fill
   the code of the functions below. You should use iftAddFrame for
   padding zeroes and iftRemFrame to return the original image size
   whenever needed. Object pixels in these functions are pixels with
   value different from zero (e.g., usually 1 or 255) and background
   pixels are those with value equal to zero.  Object pixels are
   internal border pixels when they have a four-neighbor outside the
   object and background pixels are external border pixels when they
   have a four-neighbor inside an object.
 */


/* it returns pixels at the border of the image */

// iftSet *MyImageBorder(iftImage *bin)
// {

// }

/* it returns a set with internal border pixels */

iftSet *MyObjectBorder(iftImage *bin)
{
   iftAdjRel *A = iftCircular(sqrtf(2));
   iftSet *S=NULL;
   // Iterates over the image pixels
   for (size_t p=0; p <= bin->n; p++) {
      if (bin->val[p] > 0) {
         iftVoxel u = iftGetVoxelCoord(bin, p);
         // Iterates over the adjacent matrix
         for (size_t i=1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            // Verfifies if its a valid Voxel (To avoid memory dump)
            if (iftValidVoxel(bin, v)) {
               int q = iftGetVoxelIndex(bin, v);
               if (bin->val[q] == 0) {
                  iftInsertSet(&S, p);
                  break;
               }
            }
         }
      }
   }

   return S;
}

/* it returns a set with external border pixels */

iftSet *MyBackgroundBorder(iftImage *bin)
{
   iftAdjRel *A = iftCircular(sqrtf(2));
   iftSet *S=NULL;
   // Iterates over the image pixels
   for (size_t p=0; p <= bin->n; p++) {
      if (bin->val[p] == 0) {
         iftVoxel u = iftGetVoxelCoord(bin, p);
         // Iterates over the adjacent matrix
         for (size_t i=1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            // Verfifies if its a valid Voxel (To avoid memory dump)
            if (iftValidVoxel(bin, v)) {
               int q = iftGetVoxelIndex(bin, v);
               if (bin->val[q] >= 0) {
                  iftInsertSet(&S, p);
                  break;
               }
            }
         }
      }
   }

   return S;
}

/* it dilates objects */

iftImage *MyDilateBin(iftImage *bin, iftSet **S, float radius)
{
   /* Creates cost matrix with all values as 0, and then
   set then all to INFINITY_INT*/
   iftImage *cost = iftCreateImageFromImage(bin);
   iftSetImage(cost, IFT_INFINITY_INT);
   iftImage *dilated_bin = iftCopyImage(bin);
   printf("[0] Set size => %d\n", iftSetSize(*S));
   *S = MyObjectBorder(bin);
   printf("[1] Set size => %d\n", iftSetSize(*S));

   return dilated_bin;
}

/* it erodes objects */

// iftImage *MyErodeBin(iftImage *bin, iftSet **S, float radius)
// {



// }

/* it executes dilation followed by erosion */

// iftImage *MyCloseBin(iftImage *bin, float radius)
// {



// }

/* it executes erosion followed by dilation */

// iftImage *MyOpenBin(iftImage *bin, float radius)
// {



// }

/* it executes closing followed by opening */

// iftImage *MyAsfCOBin(iftImage *bin, float radius)
// {



// }

/* it closes holes in objects */

// iftImage *MyCloseBasins(iftImage *bin)
// {



// }

int main(int argc, char *argv[])
{
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


   if (argc != 3) {
      printf("project01 <P1> <P2>\n");
      printf("P1: folder with original images\n");
      printf("P2: folder with cropped images\n");
      exit(0);
   }

   tstart = iftTic();

   iftFileSet *fs   = iftLoadFileSetFromDirBySuffix(argv[1],".png", 1);
   int nimages      = fs->n;
   char *out_dir    = argv[2];
   iftMakeDir(out_dir);
   iftAdjRel *A     = iftCircular(3.5), *B = iftCircular(1.5);
   for (int i=0; i < nimages; i++) {
      iftSet *S = NULL;
      char *basename = iftFilename(fs->files[i]->path,".png");
      iftImage *orig = iftReadImageByExt(fs->files[i]->path);
      /* normalize  image */
      iftImage *norm = iftNormalize(orig,0,255);
      /* binarize image */
      iftImage *aux1 = iftBelowAdaptiveThreshold(norm, NULL, A, 0.98, 2, 255);
      /* remove noise components from the background */
      iftImage *aux2 = iftSelectCompAboveArea(aux1,B,100);
      iftDestroyImage(&aux1);
      printf("[2] Set size => %d\n", iftSetSize(S));
      aux1 = MyDilateBin(aux2, &S, 15.0);
      printf("[3] Set size => %d\n", iftSetSize(S));

      /* apply morphological filtering to make the fingerprint the
      largest component: this operation must add frame and remove it
      afterwards.
      */
      //  aux1           = MyAsfCOBin(aux2,15.0);//iftAsfCOBin(aux2,15.0);
      //  iftDestroyImage(&aux2);
      //  /* close holes inside the components to allow subsequent erosion
      //     from the external borders only */
      //  aux2           = MyCloseBasins(aux1); //iftCloseBasins(aux1,NULL,NULL);
      //  iftDestroyImage(&aux1);
      //  /* erode components and select the largest one to estimate its
      //     center as close as possible to the center of the fingerprint */
      //  iftSet *S = NULL;
      //  aux1           = MyErodeBin(aux2,&S,30.0);//iftErodeBin(aux2,&S,30.0);
      //  iftDestroySet(&S);
      //  iftDestroyImage(&aux2);
      //  aux2           = iftSelectLargestComp(aux1,B);

      //  /* crop the normalized image by the minimum bounding box of the
      //     resulting mask (largest component) */

      //  iftDestroyImage(&aux1);
      //  iftVoxel pos;
      //  iftBoundingBox bb = iftMinBoundingBox(aux2, &pos);
      //  aux1              = iftExtractROI(norm,bb);

      sprintf(filename,"%s/%s.png",out_dir,basename);
      //  iftWriteImageByExt(aux1,filename);
      iftDestroyImage(&aux1);
      iftDestroyImage(&aux2);
      iftDestroyImage(&orig);
      iftDestroyImage(&norm);
      iftDestroySet(&S);
      iftFree(basename);
      printf("Processed %d/%d images\n",i+1,nimages);
   }

  iftDestroyFileSet(&fs);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);

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








