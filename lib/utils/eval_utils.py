import os
import numpy as np
import glob
import pickle as pkl
from sklearn import metrics

class Evaluator():
    def __init__(self, args, label_file=None):
        self.args = args
        if self.args.test_dataset == "taiwan_sa":
            self.labels = self.load_taiwan_sa()
        elif self.args.test_dataset == "A3D":
            self.labels = self.load_A3D(label_file)
        else:
            raise NameError(self.args.test_dataset + " is unknown!")

    
    def load_taiwan_sa(self):
        '''
        In taiwan dataset, all anomalies are the last 25 frames, so we don't load file. 
        Instead we generate labels directory
        '''
        all_videos = sorted(glob.glob(os.path.join(self.args.test_root,'*')))
        print("Number of testing videos: ", len(all_videos))
        labels = {}
        self.video_lengths = {}
        for video in all_videos:
            video_name = video.split('/')[-1]
            tmp_labels = np.zeros(100)
            tmp_labels[-25:] = 1
            labels[video_name] = tmp_labels
            self.video_lengths[video_name] = 100
        return labels
    
    def load_A3D(self, label_file):
        '''
        A3D labels are saved as pkl
        '''
        self.full_labels = pkl.load(open(label_file, 'rb'))
        print("Number of testing videos: ", len(self.full_labels.keys()))
        labels = {}
        self.video_lengths = {}
        for video_name, value in self.full_labels.items():
#             if not value['ego_envolve'] and not value['ego_only']:

            labels[video_name] = value['target']
            self.video_lengths[video_name] = int(value['clip_end']) - int(value['clip_start']) + 1
        return labels
    
    @staticmethod
    def compute_AUC(all_anomaly_scores, all_labels, normalize=True, ignore=[]):
        # precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        # auc = metrics.auc(recall, precision)
        
        scores, labels, zero_score_videos = Evaluator.get_score_label(all_anomaly_scores, 
                                                                   all_labels,
                                                                   normalize=normalize,
                                                                   ignore=ignore)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

        auc = metrics.auc(fpr, tpr)
        return auc, fpr, tpr
    
#     def merge_fvl_ego(all_fvl_anomaly_scores, 
#                       all_ego_anomaly_scores, 
#                       all_labels):
#         fvl_scores, labels, zero_score_videos = Evaluator.get_score_label(all_fvl_anomaly_scores, 
#                                                    all_labels)
#         ego_scores, labels, zero_score_videos = Evaluator.get_score_label(all_ego_anomaly_scores, 
#                                                    all_labels)
    
    @staticmethod
    def get_score_label(all_anomaly_scores, all_labels, normalize=True, ignore=[]):
        '''
        Params:
            all_anomaly_scores: a dict of anomaly scores of each video
            all_labels: a dict of anomaly labels of each video
        '''
        
        
        anomaly_scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        zero_score_videos = []
        for key, scores in all_anomaly_scores.items():
            if key in ignore:
                continue
            if np.max(scores) - np.min(scores) > 0:
                if normalize:
                    scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))
                anomaly_scores = np.concatenate((anomaly_scores, scores), axis=0)
                labels = np.concatenate((labels, all_labels[key]), axis=0)
            else:
                zero_score_videos.append(key)
        return anomaly_scores, labels, zero_score_videos

    

def bbox2pixel_traj(bbox,predict_diff=True,prev_box=None,W=1280,H=640):
        '''
        for plotting purpose
        [cx,cy,w,h]
        '''
        if predict_diff:
            traj = np.vstack([prev_box[:,:4],bbox[:,:4]+prev_box[:,:4]])
        else:
            if prev_box is None:
                traj = bbox[:,:4]
            else:
                traj = np.vstack([prev_box[:,:4],bbox[:,:4]])
        traj[:,0] *= W
        traj[:,1] *= H
        traj[:,2] *= W
        traj[:,3] *= H

        return traj
def pred_to_box(pred, prev_box=None, W=1280, H=640):
    
    pred[:,[0,2]]  = pred[:,[0,2]] * W
    pred[:,[1,3]]  = pred[:,[1,3]] * H
    
    init_box = prev_box
    init_box[:,[0,2]]  = init_box[:,[0,2]] * W
    init_box[:,[1,3]]  = init_box[:,[1,3]] * H
    
    if prev_box is not None:
        traj = np.vstack([init_box[:,:2],pred[:,:2]+init_box[:,:2]])
    else:
        traj = bbox[:,:2]


def compute_IOU(bbox_true, bbox_pred, format='xywh'):
    '''
    compute IOU
    [cx, cy, w, h] or [x1, y1, x2, y2]
    '''
    if format == 'xywh':
        xmin = np.max([bbox_true[0] - bbox_true[2]/2, bbox_pred[0] - bbox_pred[2]/2]) 
        xmax = np.min([bbox_true[0] + bbox_true[2]/2, bbox_pred[0] + bbox_pred[2]/2])
        ymin = np.max([bbox_true[1] - bbox_true[3]/2, bbox_pred[1] - bbox_pred[3]/2])
        ymax = np.min([bbox_true[1] + bbox_true[3]/2, bbox_pred[1] + bbox_pred[3]/2])
        w_true = bbox_true[2]
        h_true = bbox_true[3]
        w_pred = bbox_pred[2]
        h_pred = bbox_pred[3]
    elif format == 'x1y1x2y2':
        xmin = np.max([bbox_true[0], bbox_pred[0]])
        xmax = np.min([bbox_true[2], bbox_pred[2]])
        ymin = np.max([bbox_true[1], bbox_pred[1]])
        ymax = np.min([bbox_true[3], bbox_pred[3]])
        w_true = bbox_true[2] - bbox_true[0]
        h_true = bbox_true[3] - bbox_true[1]
        w_pred = bbox_pred[2] - bbox_pred[0]
        h_pred = bbox_pred[3] - bbox_pred[1]
    else:
        raise NameError("Unknown format {}".format(format))
    w_inter = np.max([0, xmax - xmin])
    h_inter = np.max([0, ymax - ymin])
    intersection = w_inter * h_inter
    union = (w_true * h_true + w_pred * h_pred) - intersection

    return intersection/union