import argparse
import os
import pandas as pd

def read_csv(csv_file):
    return pd.read_csv(csv_file, index_col=False)

def df2dict(df):
    return df.set_index('file').T.to_dict('dict')

def retrieve_bbox(img_dict: dict):
    return list(img_dict.values())

def iou(gt:dict, ift:dict, csv_iou='iou_fp.csv'):
    print('[INFO] Calculanting IoU')
    data = list()
    for key in gt.keys():
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = retrieve_bbox(gt[key])
        xmin_ift, ymin_ift, xmax_ift, ymax_ift = retrieve_bbox(ift[key])
        
        x1 = max(xmin_gt, xmin_ift)    
        y1 = max(ymin_gt, ymin_ift)    
        x2 = max(xmax_gt, xmax_ift)    
        y2 = max(ymax_gt, ymax_ift)    

        iA = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        gt_bboxAA = (xmax_gt - xmin_gt + 1) * (ymax_gt - ymin_gt + 1) 
        igt_bboxAA = (xmax_gt - xmin_gt + 1) * (ymax_gt - ymin_gt + 1) 

        _iou = iA / float(gt_bboxAA + igt_bboxAA - iA)
        _data = {'img': key, 'iou': _iou}
        data.append(_data)

    print(f'[INFO] Generating IoU csv -> {csv_iou}')
    df_iou = pd.DataFrame(data)
    df_iou.to_csv(csv_iou, index_label=False)

def main(gt_csv, ift_csv):
    df_gt, df_ift = read_csv(gt_csv), read_csv(ift_csv)
    dict_gt, dict_ift = df2dict(df_gt), df2dict(df_ift)
    
    iou(dict_gt, dict_ift)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_csv', type=str, default='fingerprint_seed13_50.csv',
                        help='Ground Truth bbox annotation csv path')
    parser.add_argument('ift_csv', type=str, help='IFT bbox annotation csv path')

    args = parser.parse_args()
    gt_csv = args.gt_csv
    ift_csv = args.ift_csv

    main(gt_csv, ift_csv)