import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset IDS Generator")
    parser.add_argument(
        '--input_folder', type=str, help='Path to dataset images',
        default='../data/images_02/'
    )
    parser.add_argument(
        '--output_folder', type=str, help='Path to save the subset files',
        default="../compare_files/"
    )
    args = parser.parse_args()

    images = os.listdir(args.input_folder)
    subject_ids = np.unique([image.split("_")[0] for image in images])
    
    subsets = {
        "train": subject_ids[: int(0.7 * len(subject_ids))],
        "validation": subject_ids[
                        int(0.7 * len(subject_ids)): int(0.85 * len(subject_ids))],
        "test": subject_ids[int(0.85 * len(subject_ids)):]
    }

    for subset in subsets.keys():
        subset_images = []
        subset_ids = subsets[subset]
        for subject_id in subset_ids:
            filtered_images = [image for image in images if subject_id in image.split("_")[0]]
            for filtered_image in filtered_images:
                subset_images.append(filtered_image)
        path_to_save = os.path.join(args.output_folder, f"{subset}.txt")
        with open(path_to_save, "w") as file:
            file.writelines(image_name + "\n" for image_name in subset_images)
    