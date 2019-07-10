'''
Feb 23 2019
run our fol+ego model on two datasets and save all the information needed for anomaly detection
The save fiels are like follows:

* Each video has a corresponding dictionary saved as .pkl
* Each dictionary is:
    * "frame_id": a list [0,1,2,...]
    * "ego_motion_obs": observed ego motion at each time,  an array with size (time, 3)
    * "ego_motion_pred": previously predicted ego motion at each 
        time a list of arrays with different shape, [(1,3), (2,3),...,(5,3), (5,3)...]
    * "bbox_gt": a list of dictionary, [dict(), dict(),...]
        * each dict is: bbox_gt[i][track_id] is a numpy array with shape (1,4)
    * "bbox_pred": a list of dictionary, [dict(), dict(),...]
        * each dict is: bbox_pred[pred][track_id] is a numpy array with shape (n, 4) are the prediction of this object from $n$ previous steps
'''
import sys
sys.path.append('../')
import os 
import numpy as np
import time
import yaml
import glob
import pickle as pkl

import cv2
import copy

import torch
import argparse
from torch.utils import data
from config.config import * 

from lib.utils.fol_dataloader import HEVIDataset
from lib.models.rnn_ed import FolRNNED, EgoRNNED
from lib.models.trackers import Tracker, AllTrackers, EgoTracker 
from lib.models.trackers import bbox_loss

from lib.utils.flow_utils import show_flow
from lib.utils.data_prep_utils import *
from lib.utils.eval_utils import Evaluator
from lib.utils.visualize_utils import vis_multi_prediction

from config.config import parse_args, visualize_config

print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, visualize=False):

    # initialize FOL model and ego_pred model
    fol_predictor =  FolRNNED(args).to(device)
    fol_predictor.load_state_dict(torch.load(args.best_fol_model)) 
    ego_motion_predictor =  EgoRNNED(args).to(device)
    ego_motion_predictor.load_state_dict(torch.load(args.best_ego_pred_model))

    # load the pre-computed tracklets of the test data
    all_track_files = sorted(glob.glob(os.path.join(args.track_dir, '*.npy')))

    # Run inference on each test video, save 
    for video_idx, track_file in enumerate(all_track_files):
        output_dict_list = []
        
        video_name = track_file.split('/')[-1].split('.')[0]
        print(video_name)
        if os.path.isfile(os.path.join(args.save_dir, video_name+'.pkl')):
            print(video_name, " has been processed!")
            continue
        
        
        flow_folder = os.path.join(args.flow_dir, video_name)
        image_folder = os.path.join(args.img_dir, video_name, 'images')
        
        video_len = len(glob.glob(os.path.join(image_folder, '*')))
        
        # load tracking data and ego motion data
        track_data = np.load(track_file)
        ego_motion_data = np.load(os.path.join(args.ego_motion_dir, video_name+'.npy'))
        
        '''initialize tracker object for anomaly detection'''
        all_trackers = AllTrackers()
        ego_motion_tracker = EgoTracker(args, 
                                        ego_motion_predictor.predict, 
                                        ego_motion_data[0:1,:])
        
        output_dict = {}
        output_dict['frame_id'] = 0
        output_dict['ego_motion_obs'] = ego_motion_data[0:1,:]
        output_dict['ego_motion_perd'] = []
        
        '''initialize data row id and frame id'''
        i = 0
        if len(track_data) == 0:
            current_frame_id = video_len
        else:
            current_frame_id = int(track_data[0][0])
        
        if current_frame_id > 1:
            '''if the first frame doesnt have bbox detected, save as empty'''
            output_dict['bbox_gt'] = []
            output_dict['bbox_pred'] = {}
            output_dict_list.append(output_dict)
        
        for frame_id in range(1, current_frame_id-1):
            '''
            Bbox and flow may not start from the first frame, so we loop ego motion first
            until bboxes are firstly detected.
            '''
            ego_motion_tracker.update(ego_motion_data[frame_id:frame_id+1,:])
            
            # save in the output
            output_dict = {}
            output_dict['frame_id'] = frame_id
            output_dict['ego_motion_obs'] = ego_motion_data[frame_id:frame_id+1,:]
            output_dict['ego_motion_perd'] = ego_motion_tracker.pred_ego_t
            output_dict['bbox_gt'] = []
            output_dict['bbox_pred'] = {}
            output_dict_list.append(output_dict)
        
        while current_frame_id <= video_len:#track_data[-1][0]:
            # read the flow of that frame
            flow_file = os.path.join(flow_folder, 
                                    str(format(current_frame_id,'06'))+'.flo')
            flow = read_flo(flow_file)
            flow = np.expand_dims(flow, axis=0)
            
            '''Ego motion update'''
            pred_ego_motion_chanegs = ego_motion_tracker.update(ego_motion_data[current_frame_id-1:current_frame_id,:])
            
            '''Object motion update'''
            # loop over all detections in one frame
            all_observed_boxes = []
            if i < len(track_data):
                # update the tracking based on observation
                while track_data[i][0] == current_frame_id:
                    track_id = int(track_data[i][1])
                    bbox = np.expand_dims(track_data[i][2:6], axis=0) # bbox is in tlwh, need to convert to cxcywh!!
                    bbox[:,0] += bbox[:,2]/2
                    bbox[:,1] += bbox[:,3]/2
                    all_observed_boxes.append(track_data[i][1:6])
                    if track_id not in all_trackers.tracker_ids:
                        # add a new tracker to thetracker list
        #                 print("Adding a new tracker: ", track_id)
                        new_tracker = Tracker(args, 
                                            fol_predictor.predict, 
                                            track_id, 
                                            bbox, 
                                            flow,
                                            pred_ego_motion_chanegs,
                                            None)
                        all_trackers.append(new_tracker)
                    else:
                        # update trackers if they exist already
                        all_trackers.trackers[track_id].update_observed(bbox, 
                                                                        flow, 
                                                                        pred_ego_motion_chanegs, 
                                                                        feature=None)
                    i += 1
                    if i >= len(track_data):
                        break

            all_observed_boxes = np.array(all_observed_boxes)
            try:
                all_observed_track_id = all_observed_boxes[:,0]
            except:
                all_observed_track_id = []
            
            ## TODO: what if a car disappear for several frame and then show up?
            '''update missed trackers using the predicted box'''
            tracker_ids = all_trackers.tracker_ids
            for tracker_id in tracker_ids:
                if tracker_id not in all_observed_track_id:
                    all_trackers.trackers[tracker_id].update_missed(flow,
                                                                    pred_ego_motion_chanegs)
                    # remove teh tracker if it has been missed several times
                    if all_trackers.trackers[tracker_id].missed_time >= args.max_age:
    #                     print("Removing tracker %d"%tracker_id)
                        all_trackers.remove(tracker_id)
        
            # save in the output
            output_dict = {}
            output_dict['frame_id'] = current_frame_id - 1
            output_dict['ego_motion_obs'] = ego_motion_data[current_frame_id-1:current_frame_id,:]
            output_dict['ego_motion_perd'] = ego_motion_tracker.pred_ego_t
            
            output_dict['bbox_gt'] = {}
            for tracker_id in all_observed_track_id:
                output_dict['bbox_gt'][tracker_id] = all_trackers.trackers[tracker_id].bbox
            
            output_dict['bbox_pred'] = {}
            for tracker_id in all_trackers.tracker_ids:
                output_dict['bbox_pred'][tracker_id] = all_trackers.trackers[tracker_id].pred_boxes_t
                
            output_dict_list.append(output_dict)
                
            ## TODO:
            '''visualize prediction'''
            if visualize:
                img_name = str(format(current_frame_id,'06'))+'.jpg'
                img_file = os.path.join(image_folder, img_name)
                img = cv2.imread(img_file) # (H, W, 3)
                
                save_dir = os.path.join(OUT_DIR, video_name)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                save_dir = os.path.join(save_dir, img_name)
                
                vis_multi_prediction(img, 
                                    all_trackers.trackers, 
                                    all_observed_boxes, save_dir=save_dir)
                

            '''update current frame id'''
            current_frame_id += 1 #= int(track_data[i][0])
        
        # save the outputs of FOL+Ego_pred for anomaly detection evaluation
        with open(os.path.join(args.save_dir, video_name+'.pkl'),'wb') as f:
            pkl.dump(output_dict_list, f)

if __name__=='__main__':
    args = parse_args()
    main(args, visualize=False)