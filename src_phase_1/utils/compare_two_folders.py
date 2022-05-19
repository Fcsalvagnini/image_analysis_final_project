import cv2
import argparse
import os
import math

def compare_images(folder_1, folder_2):
    files = os.listdir(folder_1)
    # Filter out .csv file
    files = [file for file in files if ".png" in file]

    n_files = len(files)
    diff_images = []

    for file in files:
        img_1 = cv2.imread(os.path.join(folder_1, file))
        img_2 = cv2.imread(os.path.join(folder_2, file))

        equal_pixels = (img_1 == img_2).sum()
        n_pixels = math.prod(img_1.shape)

        if (equal_pixels != n_pixels):
            diff_images.append(file)

    print("{}% of the images have the same pixel values".format(
        ((n_files) - len(diff_images))/n_files * 100
    ))

    if len(diff_images) > 0:
        print("The following images have different pixel values:")
        for file in diff_images:
            print(f"- {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Images Comparer",
        "Use it to compare images from two different folders",
        "Given two folders holding images with the same names compares " \
             "if all images have the same pixel values"
    )
    parser.add_argument("folder_1", type=str,
                        help="Patj to to folder 1")
    parser.add_argument("folder_2", type=str,
                        help="Path to folder 2")
    args = parser.parse_args()

    compare_images(args.folder_1, args.folder_2)