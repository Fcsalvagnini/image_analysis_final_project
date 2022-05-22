import argparse

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


"""
script usage example:
    python3 calculate_IoU.py --gt_csv=<gt-csv-path> --ift_csv=<ift-csv-path> [--save_image=<bool>] [--path=<image-path>]
        args:
            --gt_csv: groundtruth bounding box data stored as "csv" file formay [required]
            --gt_csv: ift bounding box data stored as "csv" file format [required]
            --save_image: boolean option of wheter save or not the output images (listed on gt_csv) with groundtruth, ift bbox, and correspondent IoU metric [optional]
            --path: 
             
    usage example:
    python3 calculate_IoU.py --gt_csv=../misc/fingerprint_seed13_50.csv --ift_csv=../output_folder/output_final/ift_cropped_bb.csv --save_image=True --path=../images_01

"""

def read_csv(csv_file):
    return pd.read_csv(csv_file, index_col=False)

def df2dict(df):
    return df.set_index('file').T.to_dict('dict')

def retrieve_bbox(img_dict: dict):
    return list(img_dict.values())

def read_img(imPath):
    return Image.open(imPath).convert('RGB')

def checkDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def plot_boxes(imPath, box_gt, box_ift, _iou, saving_path='bbox_comparison'):
    
    image = read_img(imPath)
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = box_gt
    xmin_ift, ymin_ift, xmax_ift, ymax_ift = box_ift
    plt.figure(figsize=(6,6))
    
    plt.imshow(image)
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((xmin_gt, ymin_gt), xmax_gt-xmin_gt, ymax_gt-ymin_gt,
                        fill=None, alpha=1, edgecolor='r', label='GT'))
    plt.text(xmin_gt, ymin_gt+25, 'GT', c='r')
    currentAxis.add_patch(Rectangle((xmin_ift, ymin_ift), xmax_ift-xmin_ift, ymax_ift-ymin_ift,
                        fill=None, alpha=1, edgecolor='dodgerblue', label='IFT'))
    plt.text(xmin_ift, ymin_ift+25, 'IFT', c='dodgerblue')
    plt.title(f'IoU: {_iou:.3f}', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(saving_path, imPath.split('/')[-1]))
    


def iou(gt:dict, ift:dict, save_image, path, saving_path, csv_iou='iou_fp.csv'):
    print('[INFO] Calculating IoU')
    data = list()
    total_ims = len(gt.keys())
    for i, key in enumerate(gt.keys()):
        gt_box = retrieve_bbox(gt[key])
        ift_box = retrieve_bbox(ift[key])
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_box
        xmin_ift, ymin_ift, xmax_ift, ymax_ift = ift_box
        
        x1 = max(xmin_gt, xmin_ift)    
        y1 = max(ymin_gt, ymin_ift)    
        x2 = min(xmax_gt, xmax_ift)    
        y2 = min(ymax_gt, ymax_ift)    

        iA = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        gt_bboxAA = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt) 
        ift_bboxAA = (xmax_ift - xmin_ift) * (ymax_ift - ymin_ift) 

        _iou = iA / float(gt_bboxAA + ift_bboxAA - iA)
        _data = {'img': key, 'iou': _iou}
        data.append(_data)
        if save_image:
            print(f'[INFO] {i+1}/{total_ims} Generating bbox plots', end='\r')
            imPath = os.path.join(path, key)
            plot_boxes(imPath, gt_box, ift_box, _iou, saving_path)

    print(f'[INFO] Generating IoU csv -> {csv_iou}')
    df_iou = pd.DataFrame(data)
    df_iou.to_csv(csv_iou, index_label=False)

def main(gt_csv, ift_csv, save_image, path):
    saving_path=None
    df_gt, df_ift = read_csv(gt_csv), read_csv(ift_csv)
    dict_gt, dict_ift = df2dict(df_gt), df2dict(df_ift)
    if save_image:
        saving_path='bbox_comparison'
        checkDir(saving_path)
    
    iou(dict_gt, dict_ift, save_image, path, saving_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv', type=str, default='fingerprint_seed13_50.csv',
                        help='Ground Truth bbox annotation csv path', required=True)
    parser.add_argument('--ift_csv', type=str, help='IFT bbox annotation csv path', required=True)
    parser.add_argument('--save_image', type=bool, help='Save bbox comparison (GT + IFT', default=False)
    parser.add_argument('--path', type=str, help='Fingerprint images path', default=None)

    args = parser.parse_args()
    gt_csv = args.gt_csv
    ift_csv = args.ift_csv
    save_image = args.save_image
    path = args.path
    if save_image and not (path):
        raise Exception(f"Setting '--save_image' as {save_image} demands the input of '--path' arg. Please, call it again passing the image path.")
    main(gt_csv, ift_csv, save_image, path)