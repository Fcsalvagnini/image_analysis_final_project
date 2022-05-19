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
iftSet *MyImageBorder(iftImage *bin)
{
   // Adjacency relation of neighbourhood 4
   iftAdjRel *A = iftCircular(1);
   iftSet *S=NULL;
   // Iterates over the image pixels
   for (size_t p=0; p < bin->n; p++) {
      iftVoxel u = iftGetVoxelCoord(bin, p);
      // Iterates over the adjacent matrix
      for (size_t i=1; i < A->n; i++) {
         iftVoxel v = iftGetAdjacentVoxel(A, u, i);
         /* Verify if its a valid adjacent Voxel. If not, the pixel p is at
         image border */
         if (!iftValidVoxel(bin, v)) {
            iftInsertSet(&S, p);
            break;
         }
      }
   }

   iftDestroyAdjRel(&A);

   return S;
}

/* it returns a set with internal border pixels */
iftSet *MyObjectBorder(iftImage *bin)
{
   // Adjacency relation of neighbourhood 4
   iftAdjRel *A = iftCircular(1);
   iftSet *S=NULL;
   // Iterates over the image pixels
   for (size_t p=0; p < bin->n; p++) {
      // Only for border values (!= 0)
      if (bin->val[p] > 0) {
         iftVoxel u = iftGetVoxelCoord(bin, p);
         // Iterates over the adjacent matrix
         for (size_t i=1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            // Verifies if its a valid Voxel (To avoid memory dump)
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
   for (size_t p=0; p < bin->n; p++) {
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
   iftImage *dilated_bin = iftCopyImage(bin);

   // Adjacency relation of neighbourhood 8
   iftAdjRel *A = iftCircular(sqrtf(2));
   int tmp = 0;
   iftGQueue *Q=NULL;

   // /* Creates cost and root matrix with all values as 0, and then
   // set then all to INFINITY_INT*/
   iftImage *cost = iftCreateImageFromImage(bin);
   iftImage *root = iftCreateImageFromImage(bin);
   iftSetImage(cost, IFT_INFINITY_INT);
   iftSetImage(root, IFT_INFINITY_INT);

   // [TODO] Optimize the number of buckets to avoid extra allocations
   Q = iftCreateGQueue(256, bin->n, cost->val);

   // If SetSize is 0, then get object border
   if (*S == NULL) {
      *S = MyObjectBorder(bin);
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
         dilated_bin->val[p] = bin->val[root->val[p]];
         iftVoxel edge_voxel = iftGetVoxelCoord(bin, root->val[p]);
         // Iterates over the adjacent matrix
         iftVoxel u = iftGetVoxelCoord(bin, p);
         for (size_t i=1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(bin, v)) {
               int q = iftGetVoxelIndex(bin, v);
               if ( (cost->val[q] > cost->val[p]) && bin->val[q] == 0) {
                  // Computes the new cost value regarding the edge voxel
                  tmp = pow(edge_voxel.x - v.x, 2) + pow(edge_voxel.y - v.y, 2);
                  if (tmp < cost->val[q]) {
                     // Verifies if its in the Queue and removes it
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

   iftDestroyImage(&cost);
   iftDestroyImage(&root);
   iftDestroyAdjRel(&A);
   iftDestroyGQueue(&Q);

   return dilated_bin;
}

/* it erodes objects */

iftImage *MyErodeBin(iftImage *bin, iftSet **S, float radius)
{
   iftImage *eroded_bin = iftCopyImage(bin);

   // Adjacency relation of neighbourhood 8
   iftAdjRel *A = iftCircular(sqrtf(2));
   int tmp = 0;
   iftGQueue *Q=NULL;

   // /* Creates cost and root matrix with all values as 0, and then
   // set then all to INFINITY_INT*/
   iftImage *cost = iftCreateImageFromImage(bin);
   iftImage *root = iftCreateImageFromImage(bin);
   iftSetImage(cost, IFT_INFINITY_INT);
   iftSetImage(root, IFT_INFINITY_INT);

   // [TODO] Optimize the number of buckets to avoid extra allocations
   Q = iftCreateGQueue(256, bin->n, cost->val);

   // If SetSize is 0, then get object border
   if (*S == NULL) {
      *S = MyBackgroundBorder(bin);
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
         eroded_bin->val[p] = bin->val[root->val[p]];
         iftVoxel edge_voxel = iftGetVoxelCoord(bin, root->val[p]);
         // Iterates over the adjacent matrix
         iftVoxel u = iftGetVoxelCoord(bin, p);
         for (size_t i=1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftValidVoxel(bin, v)) {
               int q = iftGetVoxelIndex(bin, v);
               if ( (cost->val[q] > cost->val[p]) && bin->val[q] != 0) {
                  // Computes the new cost value regarding the edge voxel
                  tmp = pow(edge_voxel.x - v.x, 2) + pow(edge_voxel.y - v.y, 2);
                  if (tmp < cost->val[q]) {
                     // Verifies if its in the Queue and removes it
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

   iftDestroyImage(&cost);
   iftDestroyImage(&root);
   iftDestroyAdjRel(&A);
   iftDestroyGQueue(&Q);

   return eroded_bin;
}

/* it executes dilation followed by erosion */
iftImage *MyCloseBin(iftImage *bin, float radius)
{
   iftSet *S = NULL;
   iftImage *padded_bin = iftAddFrame(bin, radius + 2, 0);
   iftImage *dilated_padded_bin = MyDilateBin(padded_bin, &S, radius);
   iftImage *eroded_padded_bin = MyErodeBin(dilated_padded_bin, &S, radius);

   iftImage *closed_bin = iftRemFrame(eroded_padded_bin, radius + 2);

   iftDestroyImage(&padded_bin);
   iftDestroyImage(&dilated_padded_bin);
   iftDestroyImage(&eroded_padded_bin);
   iftDestroySet(&S);

   return closed_bin;
}

/* it executes erosion followed by dilation */
iftImage *MyOpenBin(iftImage *bin, float radius)
{
   iftSet *S = NULL;
   iftImage *padded_bin = iftAddFrame(bin, radius + 2, 0);
   iftImage *eroded_padded_bin = MyErodeBin(padded_bin, &S, radius);
   iftImage *dilated_padded_bin = MyDilateBin(eroded_padded_bin, &S, radius);

   iftImage *opened_bin = iftRemFrame(dilated_padded_bin, radius + 2);

   iftDestroyImage(&padded_bin);
   iftDestroyImage(&eroded_padded_bin);
   iftDestroyImage(&dilated_padded_bin);
   iftDestroySet(&S);

   return opened_bin;
}

/* it executes closing followed by opening */
iftImage *MyAsfCOBin(iftImage *bin, float radius)
{
   /* As it comprises a Closing Followed by an Opening, we have the following
   ops:
   - Closing: Dilation followed by erosion
   - Opening: Erosion followed by Dilation
   We could then summarize it as:
      - Dilate(radius) + Erosion(2*radius) + Dilate(radius)
   */
   iftSet *S = NULL;
   iftImage *padded_bin = iftAddFrame(bin, radius + 2, 0);

   iftImage *dilated_padded_bin = MyDilateBin(padded_bin, &S, radius);
   iftImage *eroded_padded_bin = MyErodeBin(dilated_padded_bin, &S, 2*radius);
   iftDestroyImage(&dilated_padded_bin);
   dilated_padded_bin = MyDilateBin(eroded_padded_bin, &S, radius);

   iftImage *closed_opened_bin = iftRemFrame(dilated_padded_bin, radius + 2);

   iftDestroyImage(&padded_bin);
   iftDestroyImage(&dilated_padded_bin);
   iftDestroyImage(&eroded_padded_bin);
   iftDestroySet(&S);

   return closed_opened_bin;
}

/* it closes holes in objects */
iftImage *MyCloseBasins(iftImage *bin)
{
   iftSet *S = MyImageBorder(bin);
   iftAdjRel *A = iftCircular(1);
   iftGQueue *Q = NULL;
   int tmp = 0;
   iftImage *closed_basins = iftCreateImageFromImage(bin);
   iftSetImage(closed_basins, IFT_INFINITY_INT);
   Q = iftCreateGQueue(256, bin->n, closed_basins->val);

   while (S != NULL) {
      int p = iftRemoveSet(&S);
      closed_basins->val[p] = bin->val[p];
      iftInsertGQueue(&Q, p);
   }

   while (!iftEmptyGQueue(Q))
   {
      int p = iftRemoveGQueue(Q);
      iftVoxel u = iftGetVoxelCoord(bin, p);
      for (size_t i=1; i < A->n; i++) {
         iftVoxel v = iftGetAdjacentVoxel(A, u, i);
         if (iftValidVoxel(bin, v)) {
            int q = iftGetVoxelIndex(bin, v);
            if (closed_basins->val[q] > closed_basins->val[p]) {
               tmp = iftMax(closed_basins->val[p], bin->val[q]);
               if (tmp < closed_basins->val[q]) {
                  closed_basins->val[q] = tmp;
                  iftInsertGQueue(&Q, q);
               }
            }
         }
      }
   }

   iftDestroyAdjRel(&A);
   iftDestroyGQueue(&Q);
   iftDestroySet(&S);

   return closed_basins;
}

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

   // File to write bounding box
   FILE *bounding_box_file;
   char bounding_box_file_path[100];
   sprintf(bounding_box_file_path,"%s/ift_cropped_bb.csv",out_dir);
   bounding_box_file = fopen(bounding_box_file_path, "w");
   /* fopen() return NULL if unable to open file in given mode. */
   if (bounding_box_file == NULL)
   {
      printf("[INFO] Unable to open '%s' file to write bounding boxes.\n", bounding_box_file_path);
      exit(0);
   }
   // Writes header to csv file
   fprintf(bounding_box_file, "file,xmin,ymin,xmax,ymax\n");

   for (int i=0; i < nimages; i++) {
      char *img_basename = iftFilename(fs->files[i]->path,".png");
      iftImage *orig = iftReadImageByExt(fs->files[i]->path);
      /* normalize  image */
      iftImage *norm = iftNormalize(orig,0,255);
      /* binarize image */
      iftImage *aux1 = iftBelowAdaptiveThreshold(norm, NULL, A, 0.98, 2, 255);
      /* remove noise components from the background */
      iftImage *aux2 = iftSelectCompAboveArea(aux1,B,100);
      iftDestroyImage(&aux1);

      /* Lines to compare out implementation against the professor ones:
      To compare the implementations, please uncoment the methods that you want
      to compare, save the images to different folders, and use the
      compare_pairs_of_images.py script.
      // aux1 = MyDilateBin(aux2, &S, 15.0);
      // aux1 = MyErodeBin(aux2, &S, 2.0);
      // aux1 = iftErodeBin(aux2, &S, 2.0);
      // aux1 = iftDilateBin(aux2, &S, 15.0);
      // aux1 = MyCloseBin(aux2, 15.0);
      // aux1 = MyOpenBin(aux2, 3.0);
      // aux1 = MyAsfCOBin(aux2, 5);
      // aux1 = iftAsfCOBin(aux2, 5);
      // aux1 = MyCloseBasins(aux2);
      // S = MyImageBorder(aux2);
      // aux1 = iftCloseBasins(aux2, S, NULL);
      // aux1 = iftOpenBin(aux2, 3.0);
      // aux1 = iftCloseBin(aux2, 15.0);
      iftDestroyImage(&aux2);
      */

      /* apply morphological filtering to make the fingerprint the
      largest component: this operation must add frame and remove it
      afterwards.
      */
      aux1 = MyAsfCOBin(aux2,15.0);//iftAsfCOBin(aux2,15.0);
      iftDestroyImage(&aux2);
      /* close holes inside the components to allow subsequent erosion
      from the external borders only */
      aux2 = MyCloseBasins(aux1); //iftCloseBasins(aux1,NULL,NULL);
      iftDestroyImage(&aux1);
      /* erode components and select the largest one to estimate its
      center as close as possible to the center of the fingerprint */
      iftSet *S = NULL;
      aux1 = MyErodeBin(aux2,&S,30.0);//iftErodeBin(aux2,&S,30.0);
      iftDestroySet(&S);
      iftDestroyImage(&aux2);
      aux2 = iftSelectLargestComp(aux1,B);

      /* crop the normalized image by the minimum bounding box of the
      resulting mask (largest component) */
      iftDestroyImage(&aux1);
      iftVoxel pos;
      iftBoundingBox bb = iftMinBoundingBox(aux2, &pos);
      aux1 = iftExtractROI(norm,bb);
      fprintf(bounding_box_file,
               "%s.png,%d,%d,%d,%d\n",
               img_basename, bb.begin.x, bb.begin.y, bb.end.x, bb.end.y);

      sprintf(filename,"%s/%s.png",out_dir,img_basename);
      iftWriteImageByExt(aux1,filename);
      iftDestroyImage(&orig);
      iftDestroyImage(&norm);
      iftDestroyImage(&aux1);
      iftDestroyImage(&aux2);
      iftFree(img_basename);
      printf("Processed %d/%d images\n",i+1,nimages);
   }

  iftDestroyFileSet(&fs);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  fclose(bounding_box_file);

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