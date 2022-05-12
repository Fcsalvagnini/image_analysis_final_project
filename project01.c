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
   // Adjacency relation of neighbourhood 4
   iftAdjRel *A = iftCircular(1);
   iftSet *S=NULL;
   // Iterates over the image pixels
   for (size_t p=0; p <= bin->n; p++) {
      // Only for border values (!= 0)
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

   iftDestroyAdjRel(&A);

   return S;
}

/* it returns a set with external border pixels */
iftSet *MyBackgroundBorder(iftImage *bin)
{
   // Adjacency relation of neighbourhood 4
   iftAdjRel *A = iftCircular(1);
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

   iftDestroyAdjRel(&A);

   return S;
}

/* it dilates objects */
iftImage *MyDilateBin(iftImage *bin, iftSet **S, float radius)
{
   // Adjacency relation of neighbourhood 8
   iftAdjRel *A = iftCircular(sqrtf(2));
   int tmp = 0;
   iftGQueue *Q=NULL;

   /* Adds padding to the input image, to deal with dilatations close
   to the image borders*/
   iftImage *padded_bin = iftAddFrame(bin, radius, 0);

   /* Creates cost and root matrix with all values as 0, and then
   set then all to INFINITY_INT*/
   iftImage *cost = iftCreateImageFromImage(padded_bin);
   iftImage *root = iftCreateImageFromImage(padded_bin);
   iftSetImage(cost, IFT_INFINITY_INT);
   iftSetImage(root, IFT_INFINITY_INT);

   // [TODO] Optimize the number of buckets to avoid extra allocations
   Q = iftCreateGQueue(256, padded_bin->n, cost->val);

   // If SetSize is 0, then get object border
   if (iftSetSize(*S) == 0) {
      *S = MyObjectBorder(padded_bin);
   }

   // Empties the S set, correctly setting the Cost and R matrix
   while (*S != NULL) {
      int p = iftRemoveSet(S);
      cost->val[p] = 0;
      root->val[p] = p;
      iftInsertGQueue(&Q, p);
   }

   while (!iftEmptyGQueue(Q))
   {
      int p = iftRemoveGQueue(Q);

      if (cost->val[p] <= pow(radius, 2)) {
         padded_bin->val[p] = 255;
         iftVoxel edge_voxel = iftGetVoxelCoord(padded_bin, root->val[p]);
         // Iterates over the adjacent matrix
         iftVoxel u = iftGetVoxelCoord(padded_bin, p);
         for (size_t i=1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(padded_bin, v)) {
               int q = iftGetVoxelIndex(padded_bin, v);
               if ( (cost->val[q] > cost->val[p]) && padded_bin->val[q] == 0) {
                  // Computes the new cost value regarding the edge voxel
                  tmp = pow(edge_voxel.x - v.x, 2) + pow(edge_voxel.y - v.y, 2);
                  if (tmp < cost->val[q]) {
                     // Verifies if its in the Queue and removes it
                     // if (cost->val[q] != IFT_INFINITY_INT) {
                     if (Q->L.elem[q].color == IFT_GRAY) {
                        iftRemoveGQueueElem(Q, q);
                     }
                     cost->val[q] = tmp;
                     root->val[q] = root->val[p];
                     iftInsertGQueue(&Q, q);
                  }
               }
            }
         }
      }
      else {
         iftInsertSet(S, p);
      }
   }

   // Removes padding from image
   iftImage *dilated_bin = iftRemFrame(padded_bin, radius);

   iftDestroyImage(&cost);
   iftDestroyImage(&root);
   iftDestroyImage(&padded_bin);
   iftDestroyAdjRel(&A);
   iftDestroyGQueue(&Q);

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
   printf("[INFO] Starting code execution\n");
   timer *t_start=iftTic();
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

   printf("[INFO] Reading input Images:\n");
   iftFileSet *fs = iftLoadFileSetFromDirBySuffix(argv[1],".png", 1);
   int nimages = fs->n;
   char *out_dir = argv[2];
   iftMakeDir(out_dir);
   iftAdjRel *A = iftCircular(3.5), *B = iftCircular(1.5);
   for (int i=0; i < 5; i++) {
      iftSet *S = NULL;
      char *img_basename = iftFilename(fs->files[i]->path,".png");
      iftImage *orig = iftReadImageByExt(fs->files[i]->path);
      /* normalize  image */
      iftImage *norm = iftNormalize(orig,0,255);
      /* binarize image */
      iftImage *aux1 = iftBelowAdaptiveThreshold(norm, NULL, A, 0.98, 2, 255);
      /* remove noise components from the background */
      iftImage *aux2 = iftSelectCompAboveArea(aux1,B,100);
      iftDestroyImage(&aux1);

      // Space to test implementations
      // Add border and removes it after
      aux1 = MyDilateBin(aux2, &S, 15.0);
      // aux1 = iftDilateBin(aux2, &S, 15.0);
      iftDestroyImage(&aux2);
      //

      /* apply morphological filtering to make the fingerprint the
      largest component: this operation must add frame and remove it
      afterwards.
      */
      // aux1           = MyAsfCOBin(aux2,15.0);//iftAsfCOBin(aux2,15.0);
      // iftDestroyImage(&aux2);
      /* close holes inside the components to allow subsequent erosion
      from the external borders only */
      // aux2           = MyCloseBasins(aux1); //iftCloseBasins(aux1,NULL,NULL);
      // iftDestroyImage(&aux1);
      /* erode components and select the largest one to estimate its
      center as close as possible to the center of the fingerprint */
      // iftSet *S = NULL;
      //  aux1           = MyErodeBin(aux2,&S,30.0);//iftErodeBin(aux2,&S,30.0);
      // iftDestroySet(&S);
      // iftDestroyImage(&aux2);
      //  aux2           = iftSelectLargestComp(aux1,B);

      /* crop the normalized image by the minimum bounding box of the
      //     resulting mask (largest component) */

      // iftDestroyImage(&aux1);
      // iftVoxel pos;
      // iftBoundingBox bb = iftMinBoundingBox(aux2, &pos);
      // aux1              = iftExtractROI(norm,bb);

      sprintf(filename,"%s/%s.png",out_dir,img_basename);
      iftWriteImageByExt(aux1,filename);
      iftDestroyImage(&orig);
      iftDestroyImage(&norm);
      iftDestroyImage(&aux1);
      iftDestroyImage(&aux2);
      iftDestroySet(&S);
      iftFree(img_basename);
      printf("Processed %d/%d images\n",i+1,nimages);
   }

  iftDestroyFileSet(&fs);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);

   timer *t_end=iftToc();
   char *formatted_time = iftFormattedTime(iftCompTime(t_start, t_end));
   puts("\nDone...");
   puts(formatted_time);
   iftFree(formatted_time);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return 0;
}