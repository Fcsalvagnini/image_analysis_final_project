import argparse

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



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
    plt.figure()
    plt.imshow(image)
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((xmin_gt, ymin_gt), xmax_gt-xmin_gt, ymax_gt-ymin_gt,
                        fill=None, alpha=1, edgecolor='r', label='GT'))
    plt.text(xmin_gt, ymin_gt+25, 'GD', c='r')
    currentAxis.add_patch(Rectangle((xmin_ift, ymin_ift), xmax_ift-xmin_ift, ymax_ift-ymin_ift,
                        fill=None, alpha=1, edgecolor='dodgerblue', label='IFT'))
    plt.text(xmin_ift, ymin_ift+25, 'IFT', c='dodgerblue')
    plt.title(f'IoU: {_iou:.3f}')
    plt.axis('off')
    
    plt.savefig(os.path.join(saving_path, imPath.split('/')[-1]))


def iou(gt:dict, ift:dict, save_image, path, csv_iou='iou_fp.csv'):
    print('[INFO] Calculanting IoU')
    data = list()
    for key in gt.keys():
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = retrieve_bbox(gt[key])
        xmin_ift, ymin_ift, xmax_ift, ymax_ift = retrieve_bbox(ift[key])
        
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
            imPath = os.path.join(path, key)
            plot_boxes(imPath,
                       [xmin_gt, ymin_gt, xmax_gt, ymax_gt],
                       [xmin_ift, ymin_ift, xmax_ift, ymax_ift],
                        _iou)

    print(f'[INFO] Generating IoU csv -> {csv_iou}')
    df_iou = pd.DataFrame(data)
    df_iou.to_csv(csv_iou, index_label=False)

def main(gt_csv, ift_csv, save_image, path):
    df_gt, df_ift = read_csv(gt_csv), read_csv(ift_csv)
    dict_gt, dict_ift = df2dict(df_gt), df2dict(df_ift)
    
    if save_image:
        checkDir(path)
    
    iou(dict_gt, dict_ift, save_image, path)


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

    main(gt_csv, ift_csv, save_image, path)