import argparse
import shutil
import os

"""
Sample of how to execute the code:

python gen_cropped_dataset.py --input_folder ../data/images_02_cropped/ 
--path_to_compare_file ../compare_files/compare_splited_v1_train.txt 
--path_to_save_new_compare_file ../compare_files/compare_splited_v3_train_new.txt 
--output_folder ../data/cropped_images_v3_train/
"""

def get_all_image_pairs(compare_file_path):
    image_pairs = []
    with open(compare_file_path, "r") as compare_file:
        lines = compare_file.read().splitlines()
        for line in lines:
            image_pairs.append(line.split(" "))

    return image_pairs[1:]

def create_dataset_folder(pairs, input_folder, output_folder, path_to_new_compare_file):
    comp_idx = 1
    pairs_to_write = []
    os.makedirs(output_folder, exist_ok=True)
    for pair in pairs:
        src_img_1 = os.path.join(input_folder, pair[0])
        src_img_2 = os.path.join(input_folder, pair[1])
        img_1_name, ext = os.path.splitext(os.path.basename(src_img_1))
        img_2_name, _ = os.path.splitext(os.path.basename(src_img_2))
        pairs_to_write.append(
            [f"{img_1_name}_{comp_idx}{ext}",
            f"{img_2_name}_{comp_idx}{ext}"]
        )
        dst_img_1 = os.path.join(output_folder, pairs_to_write[-1][0])
        dst_img_2 = os.path.join(output_folder, pairs_to_write[-1][1])

        shutil.copy(src_img_1, dst_img_1)
        shutil.copy(src_img_2, dst_img_2)

        comp_idx += 1

    with open(path_to_new_compare_file, "w") as new_comp_file:
        new_comp_file.writelines(
            f"{img_1} {img_2}\n" for img_1, img_2 in pairs_to_write
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generated Cropped Dataset")
    parser.add_argument(
        '--input_folder', type=str, help='Path to images_02 Cropped',
        default='../data/images_02_cropped/'
    )
    parser.add_argument(
        '--path_to_compare_file', type=str, help='Path to Compare File',
        default='../compare_files/compare_splited_v1_train.txt'
    )
    parser.add_argument(
        '--path_to_save_new_compare_file', type=str, help='Path to Compare File',
        default='../compare_files/compare_splited_v3_train_new.txt'
    )
    parser.add_argument(
        '--output_folder', type=str, help='Path to save the dataset',
        default='../data/cropped_images_v3_train/'
    )
    args = parser.parse_args()


    image_pairs = get_all_image_pairs(args.path_to_compare_file)

    create_dataset_folder(
        pairs=image_pairs, 
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        path_to_new_compare_file=args.path_to_save_new_compare_file,
    )



