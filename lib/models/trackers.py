import numpy as np
import torch 
from collections import deque
from lib.utils.data_prep_utils import roi_pooling_opencv, bbox_normalize
from lib.utils.eval_utils import compute_IOU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class Tracker():
    def __init__(self, args, pred_model, track_id, bbox, image_flow, ego_pred, feature):
        '''
        Params:
            args: cmd line arguments
            pred_model: link to the prediction function
            track_id: int
            box:(1,4)
            image_flow: (1,320,192,2)
            feature: None
        '''
        self.args = args
        self.feature = feature
        self.bbox_h = torch.zeros(1, self.args.box_enc_size).to(device)
        self.flow_h = torch.zeros(1, self.args.flow_enc_size).to(device)
        
        # get a random color for a tracked object
        self.color = np.random.rand(3) * 255

        self.track_id = track_id
        self.age = 0
        self.missed_time = 0
        self.predictor = pred_model

        self.all_predictions = []#deque() # a list of past predictions with length pred_timesteps + 1
                                       # e.g. [(pred_timesteps,4),(pred_timesteps,4),...]

        # update the tracker for the first time
        self.flow = np.expand_dims(np.zeros(self.args.flow_roi_size), axis=0)
        self.update_observed(bbox, image_flow, ego_pred, feature)

        # all previously predicted boxes
        self.pred_boxes_t = []

    def update_observed(self, bbox, image_flow, ego_pred, feature=None):
        '''
        Given new observations, update the tracker of that object
        Params: 
            bbox: (1,4)
            flow: (320,192,2)
        Returns:
        '''
        self.bbox = bbox_normalize(bbox,W=self.args.W, H=self.args.H)
        try:
            # update the flow if the pseudo bbox size is acceptable
            self.flow = roi_pooling_opencv(self.bbox, 
                                    image_flow, 
                                    size=[5,5])
        except:
            # otherwise use the previous flow feature
            pass

        self.predict(ego_pred)
        # update the past prediction queue 
        self.all_predictions.append(self.pred_bboxes)
        if len(self.all_predictions) > self.args.pred_timesteps + 1:
            self.all_predictions.pop(0)#popleft()
        
        # find the pred boxes at t
        self.find_pred_boxes_at_t()

        # update the age, and reset the missed time to zero
        self.age += (self.missed_time+1)
        self.missed_time = 0
        
    def update_missed(self, image_flow, ego_pred):
        '''
        If an existing bbox is missed, use the previously predicted as the 'pseudo' detection 
        '''
        self.bbox = self.pred_bboxes[0:1,:]
        try:
            # update the flow if the pseudo bbox size is acceptable
            self.flow = roi_pooling_opencv(self.bbox, 
                                    image_flow, 
                                    size=[5,5])
        except:
            # otherwise use the previous flow feature
            pass
        self.predict(ego_pred)
        # update the past prediction queue 
        self.all_predictions.append(self.pred_bboxes)
        if len(self.all_predictions) > self.args.pred_timesteps + 1:
            self.all_predictions.pop(0)#popleft()
        
        # find the pred boxes at t
        self.find_pred_boxes_at_t()

        # add a missed time
        self.missed_time += 1
    
    def predict(self, ego_pred):
        '''
        Run predictor to update hidden states and pred_bboxes
        Params:
            ego_pred: (1, pred_timesteps, 3)
        '''
        try:
            self.bbox = torch.FloatTensor(self.bbox).to(device)
        except:
            # if the predict is for updating missed object, don't need to convert to cuda
            pass
        try:
            self.flow = torch.FloatTensor(self.flow).to(device)
        except:
            pass
        box_changes, self.bbox_h, self.flow_h = self.predictor(self.bbox, 
                                                                self.flow, 
                                                                self.bbox_h, 
                                                                self.flow_h,
                                                                ego_pred)
        self.pred_bboxes = self.bbox + box_changes
        # get rid of the first dim which is batch size
        self.pred_bboxes = self.pred_bboxes[0,:,:] 
    
    def find_pred_boxes_at_t(self):
        '''
        Find the previous predicted boxes of current time
        pred_boxes_t = [X_{t}{t-1}, X_{t}{t-2}, X_{t}{t-3},...]
        '''
        prev_predictions = self.all_predictions[:-1]
        prev_steps = len(prev_predictions)
        # skip if the car is firstly observed
        self.pred_boxes_t = []
        if prev_predictions:
            for i in range(prev_steps):
                if len(self.pred_boxes_t) == 0:
                    self.pred_boxes_t = prev_predictions[prev_steps-i-1][i:i+1,:]
                else:
                    self.pred_boxes_t = torch.cat((self.pred_boxes_t,
                                            prev_predictions[prev_steps-i-1][i:i+1,:]),
                                            dim=0)
class AllTrackers():
    def __init__(self):
        self.trackers = {}
        self.tracker_ids = []

    def append(self, tracker):
        '''
        add a new tracker to the tracker list
        '''
        self.trackers[tracker.track_id] = tracker
        self.tracker_ids.append(tracker.track_id)
    
    def remove(self, track_id):
        '''
        remove a disappeared tracker from the tracker list
        '''
        try:
            self.trackers.pop(track_id)    
            self.tracker_ids.remove(track_id)
        except KeyError:
            print("Key %d not found"%(track_id))
    
def bbox_loss(all_trackers, all_observed_boxes):
    try:
        all_observed_track_id = all_observed_boxes[:,0]
    except:
        # if no boxes observed, bbox anomaly score is 0
        return 0
    L_bbox = 0
    # for each observed car, find the previous prediction of the car's box at the current time
    # e.g., X_{t}{t-1}, X_{t}{t-2}, X_{t}{t-3},...
    for track_id in all_observed_track_id:
        if track_id not in all_trackers.trackers.keys():
            # skip if the track id has already been forgotten
            continue
        # # use average prediction as the compare metrics
        pred_boxes_t = all_trackers.trackers[track_id].pred_boxes_t
        if len(pred_boxes_t) == 0:
            continue
        pred_box_t = torch.mean(pred_boxes_t, dim=0)#.view(1, 4)
        observed_box_t = all_trackers.trackers[track_id].bbox.squeeze()
        iou = compute_IOU(pred_box_t, observed_box_t)
        # if iou == 0:
        #     print("Warning: observed object location is very anomalous!")
        L_bbox += iou
    L_bbox /= len(all_observed_track_id)
    return 1 - float(L_bbox)


class EgoTracker():
    def __init__(self, args, pred_model, ego_motion):
        '''
        Params:
            args: cmd line arguments
            pred_model: link to the prediction function (not the predictor model!!)
            track_id: int
            box:(1,4)
            image_flow: (1,320,192,2)
            feature: None
        '''
        self.args = args
        self.ego_motion = ego_motion

        self.predictor = pred_model

        self.ego_h = torch.zeros(1, self.args.ego_enc_size).to(device)

        self.all_predictions = []
        self.prev_ego_motions = torch.zeros_like(torch.as_tensor(self.ego_motion)).float().to(device) # save the previous 5 steps ego_motion
        
        # update for the first time
        _ = self.update(self.ego_motion)

    def update(self, ego_motion):
        '''
        Params:
            ego_motion: (1,3) as (yaw, x, z)
        Returns:
            pred_ego_motion_chanegs: (1,pred_timesteps, 3) as changes of (yaw, x, z)
        '''
        self.ego_motion = torch.FloatTensor(ego_motion).to(device)

        model_input = self.ego_motion - self.prev_ego_motions[-1,:]
        
        
        self.pred_ego_motion_chanegs, self.ego_h = self.predictor(model_input, self.ego_h)
        
        # future ego motions are curren ego motion + predicted changes
        # shape is (1, 5, 3), need to convert to (5,3)
        self.pred_ego_motion = self.ego_motion + self.pred_ego_motion_chanegs[0]
        
        # update the past prediction queue 
        self.all_predictions.append(self.pred_ego_motion)
        if len(self.all_predictions) > self.args.pred_timesteps + 1:
            self.all_predictions.pop(0)#popleft()

        self.find_pred_ego_at_t()
        
        #update the previous motion tensor
        self.prev_ego_motions = torch.cat((self.prev_ego_motions, self.ego_motion), dim=0)

        return self.pred_ego_motion_chanegs

    def find_pred_ego_at_t(self):
        '''
        Find the previous predicted ego motion of current time
        pred_ego_t = [X_{t}{t-1}, 
                      X_{t}{t-2}, 
                      X_{t}{t-3},...]
        '''
        prev_predictions = self.all_predictions[:-1]
        prev_steps = len(prev_predictions)
        # skip if the car is firstly observed
        self.pred_ego_t = []
        if prev_predictions:
            for i in range(prev_steps):
                if len(self.pred_ego_t) == 0:
                    self.pred_ego_t = prev_predictions[prev_steps-i-1][i:i+1,:] 
                else:
                    self.pred_ego_t = torch.cat((self.pred_ego_t,
                                            prev_predictions[prev_steps-i-1][i:i+1,:]),
                                            dim=0)
    def compute_ego_loss(self):
        '''
        Compute the difference between the predicted ego motion and the observed ego motion of current time
            self.ego_motion: (1, 3)
            self.pred_ego_t: (n, 3)
        return:
            L_ego
        '''
        # use mean prediction as default
        if len(self.pred_ego_t) == 0:
            return 0
        else:
            L_ego = torch.sqrt(torch.sum((torch.mean(self.pred_ego_t, dim=0) - self.ego_motion) ** 2))
            return L_ego.detach().cpu().numpy()