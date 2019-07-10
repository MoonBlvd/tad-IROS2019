import os
import sys
sys.path.append('../')
import pickle as pkl
import numpy as np
import glob
import yaml
from sklearn import metrics
from lib.anomaly_measures import *
from lib.utils.eval_utils import Evaluator
import matplotlib.pyplot as plt

import copy
from config.config import parse_args, visualize_config
import pdb
def main(args):
    # load data
    all_files = sorted(glob.glob(os.path.join(args.save_dir, '*.pkl')))
    print("Number of videos: ", len(all_files))
    
    # initialize evaluator and labels
    evaluator = Evaluator(args, label_file=args.label_file) 

    #Load the fol-ego prediction results and compute AD measures
    all_mean_iou_anomaly_scores = {}
    all_min_iou_anomaly_scores = {}
    all_ego_anomaly_scores = {}
    all_mask_anomaly_scores = {}
    all_pred_std_mean_anomaly_scores = {}
    all_pred_std_max_anomaly_scores = {}


    for file_idx, fol_ego_file in enumerate(all_files):
        video_name = fol_ego_file.split('/')[-1].split('.')[0]
        video_len = evaluator.video_lengths[video_name]
        '''save anomaly scores in dictionary'''
        all_mean_iou_anomaly_scores[video_name] = np.zeros(video_len)
        all_min_iou_anomaly_scores[video_name] = np.zeros(video_len)
        all_ego_anomaly_scores[video_name] = np.zeros(video_len)
        all_mask_anomaly_scores[video_name] = np.zeros(video_len)
        all_pred_std_mean_anomaly_scores[video_name] = np.zeros(video_len)
        all_pred_std_max_anomaly_scores[video_name] = np.zeros(video_len)
        
        fol_ego_data = pkl.load(open(fol_ego_file,'rb'))
        for frame in fol_ego_data:
            '''compute iou metrics'''
            L_bbox = iou_metrics(frame['bbox_pred'], 
                                frame['bbox_gt'],
                                multi_box='average', 
                                iou_type='average')
            all_mean_iou_anomaly_scores[video_name][frame['frame_id']] = L_bbox
            
            L_Mask, pred_mask, observed_mask = mask_iou_metrics(frame['bbox_pred'], 
                                                                frame['bbox_gt'], 
                                                                args.W, args.H, 
                                                                multi_box='latest')
            all_mask_anomaly_scores[video_name][frame['frame_id']] = L_Mask
            
            L_bbox = iou_metrics(frame['bbox_pred'], 
                                frame['bbox_gt'],
                                multi_box='average', 
                                iou_type='min')
            all_min_iou_anomaly_scores[video_name][frame['frame_id']] = L_bbox    
            
            
            L_pred_bbox_mean, L_pred_bbox_max, anomalous_object, _ = prediction_std(frame['bbox_pred'])
            all_pred_std_mean_anomaly_scores[video_name][frame['frame_id']] = L_pred_bbox_mean
            all_pred_std_max_anomaly_scores[video_name][frame['frame_id']] = L_pred_bbox_max
            
        if file_idx % 10 == 0:
            print(file_idx)
    
    auc, fpr, tpr = Evaluator.compute_AUC(all_mean_iou_anomaly_scores, evaluator.labels)
    print("FVL MEAN IOU AUC: ", auc)
    auc, fpr, tpr = Evaluator.compute_AUC(all_mask_anomaly_scores, evaluator.labels)
    print("FVL Mask AUC: ", auc)
    auc, fpr, tpr = Evaluator.compute_AUC(all_min_iou_anomaly_scores, evaluator.labels)
    print("FVL MIN IOU AUC: ", auc)
    auc, fpr, tpr = Evaluator.compute_AUC(all_pred_std_mean_anomaly_scores, evaluator.labels)
    print("FVL PRED STD MEAN AUC: ", auc)
    auc, fpr, tpr = Evaluator.compute_AUC(all_pred_std_max_anomaly_scores, evaluator.labels)
    print("FVL PRED STD MAX AUC: ", auc)

if __name__=='__main__':
    args = parse_args()
    main(args)