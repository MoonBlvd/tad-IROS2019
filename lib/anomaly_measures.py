'''
Measures that can be used for anomaly detection.
Inputs are currrently observed bounding boxes, 
    each bounding box's prediction, 
    currently observed ego motion and 
    each bounxing box's ego motion

Feb 21 2019
'''
import numpy as np
from lib.utils.eval_utils import compute_IOU
from lib.utils.data_prep_utils import bbox_denormalize, cxcywh_to_x1y1x2y2

def boxes_union(boxes):
    '''
    Givent an array where each row is a box (Cx, Cy, w, h)
    Return:
        intersection: Box intersection area
        inter_box: the intersected rectangle box in (cx, cy, w, h)
    '''
    new_boxes = cxcywh_to_x1y1x2y2(boxes)
    xmin = np.min(new_boxes[:,0]) 
    ymin = np.min(new_boxes[:,1])
    xmax = np.max(new_boxes[:,2])
    ymax = np.max(new_boxes[:,3])


    union = np.sum(boxes[:,2] * boxes[:,3])
    union_box = np.array([(xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin])
    
    return union, union_box

def boxes_intersection(boxes):
    '''
    Givent an array where each row is a box (Cx, Cy, w, h)
    Return:
        intersection: Box intersection area
        inter_box: the intersected rectangle box in (cx, cy, w, h)
    '''
    boxes = cxcywh_to_x1y1x2y2(boxes)
    xmin = np.max(boxes[:,0]) 
    ymin = np.min(boxes[:,1])
    xmax = np.max(boxes[:,2])
    ymax = np.min(boxes[:,3])

    w_inter = np.max([0, xmax - xmin])
    h_inter = np.max([0, ymax - ymin])
    intersection = w_inter * h_inter

    if intersection == 0:
        inter_box = np.array([[0,0,0,0]])
    else:
        inter_box = np.array([(xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin])

    return intersection



def iou_metrics(all_pred_boxes_t, 
                all_observed_boxes, 
                multi_box='average', 
                iou_type='average'):
    '''
    given the boxes of observed objects, 
    compute the average iou between each box and its several previous predictions
    Params:
        all_pred_boxes_t: a dictionary saves all pred_box_t
        all_observed_boxes: a dictionary saves all bbox
        multi_box: The method to combine the five predictions: 'average', 'union' or 'intersection'
        iou_type: The method to compute across all objects: 'average', 'min'
    '''
    try:
        all_observed_track_id = all_observed_boxes.keys()#[:,0]
    except:
        # if no boxes observed, bbox anomaly score is 0
        return 0
    L_bbox = []
    # for each observed car, find the previous prediction of the car's box at the current time
    # e.g., X_{t}{t-1}, X_{t}{t-2}, X_{t}{t-3},...
    for track_id in all_observed_track_id:
        if track_id not in all_pred_boxes_t.keys():
            # skip if the track id has already been forgotten
            continue
        # # use average prediction as the compare metrics
        pred_boxes_t = all_pred_boxes_t[track_id]
        if len(pred_boxes_t) == 0:
            continue

        if multi_box == 'average':
            pred_box_t = np.mean(pred_boxes_t, axis=0)#.view(1, 4)
        elif multi_box == 'union':
            _, pred_box_t = boxes_union(pred_boxes_t)
        elif multi_box == 'intersection':
            _, pred_box_t = boxes_intersection(pred_boxes_t)
        else:
            raise NameError("Multi_box type: " + multi_box + " is unknown!")
        observed_box_t = all_observed_boxes[track_id].squeeze()
        iou = compute_IOU(pred_box_t, observed_box_t)
        L_bbox.append(iou)
    if len(L_bbox) <= 0:
        return 0
    if iou_type == 'average':
        L_bbox = np.mean(L_bbox)
  
    elif iou_type == 'min':
        L_bbox = np.min(L_bbox)
    else:
        raise NameError("IOU measure type: " + iou_type + " is unknown!")

    return 1 - float(L_bbox)

def mask_iou_metrics(all_pred_boxes_t, 
                        all_observed_boxes, 
                        W, H, 
                        multi_box='latest'):
    '''
    Compute a 0/1 mask for all tracked object and another mask for all observed boxes,
    return the difference of the mask as anomaly measures
    Params:
        all_pred_boxes_t: a dictionary saves all pred_box_t
        all_observed_boxes: a dictionary saves all bbox
        multi_box: 'latest', 'mverage', 'union'
    Return:
        mask_iou:
    '''
    pred_mask = np.zeros([H, W])
    observed_mask = np.zeros([H, W])
    
    if len(all_observed_boxes) > 0:
        for tarcker_id, bbox in all_observed_boxes.items():
            converted_box = cxcywh_to_x1y1x2y2(bbox)
            converted_box = bbox_denormalize(converted_box, W=W, H=H)[0].astype(int)
            observed_mask[converted_box[1]:converted_box[3], converted_box[0]:converted_box[2]] = 1
    if len(all_pred_boxes_t) > 0:
        for tracker_id, pred_boxes_t in all_pred_boxes_t.items():
            if len(pred_boxes_t) <= 0:
                continue
            if multi_box == 'union':
                converted_boxes = cxcywh_to_x1y1x2y2(pred_boxes_t)
                converted_boxes = bbox_denormalize(converted_boxes)
                converted_boxes = converted_boxes.astype(int)
                for box in converted_boxes:
                    pred_mask[box[1]:box[3], box[0]:box[2]] = 1
            elif multi_box == 'latest':
                pred_box_t = pred_boxes_t[0:1,:]
                converted_box = cxcywh_to_x1y1x2y2(pred_box_t)
                converted_box = bbox_denormalize(converted_box, W=W, H=H)
                converted_box = converted_box.astype(int)
                box = converted_box[0,:]
                pred_mask[box[1]:box[3], box[0]:box[2]] = 1
    
#     mask_diff = np.srqrt(np.mean((pred_mask - observed_mask)**2))
    mask_iou = compute_mask_iou(np.expand_dims(pred_mask, 0), 
                                np.expand_dims(observed_mask,0))
    return 1-mask_iou, pred_mask, observed_mask #mask_diff

def compute_mask_iou(inputs, targets):
    result = 0.0

    if type(inputs) == type(targets) == np.ndarray:
        for input, target in zip(inputs, targets):
            interest = float(np.sum((target != 0) & (target == input)))
            union = float(np.sum((target != 0) | (input !=0)))
            if union == 0:
                return 1
            result += interest / union
        return result / inputs.shape[0]
    else:
        raise(RuntimeError('Usage: IoU(inputs, targets), and inputs and targets '
                           'should be either torch.autograd.Variable or numpy.ndarray.'))

def prediction_std(all_pred_boxes_t, normalize=False):
    '''
    Find the mean pred_box_t std and the max pred_box_t std as the anomaly score
    '''
    max_score = 0
    anomalous_object = None
    all_scores = []
    score_of_all_objects = {}
    for track_id, pred_boxes_t in all_pred_boxes_t.items():
        if len(pred_boxes_t) > 0:
            score = np.std(pred_boxes_t, axis=0)
            if normalize:
                scale = np.mean(pred_boxes_t[:,2:], axis=0)
                W = scale[0]
                H = scale[1]
                score[0] /= W
                score[1] /= H
                score[2] /= W
                score[3] /= H
            
            score = np.mean(score)
            all_scores.append(score)
            if score > max_score:
                max_score = score
                anomalous_object = track_id
            score_of_all_objects[track_id] = score
    if len(all_scores) == 0:
        all_scores = [0]
    return max_score, np.mean(all_scores), anomalous_object, score_of_all_objects

def ego_motion_loss(ego_motion_obs, ego_motion_pred):
    '''
    Compute the difference between the predicted ego motion and the observed ego motion of current time
    Params:    
        ego_motion_obs: (1, 3)
        ego_motion_pred: (1, 3)
    Returns:
        L_ego: a scalar
    '''
    # use mean prediction as default
    if len(ego_motion_pred) == 0:
        return 0
    else:
        L_ego = np.sqrt(np.sum((np.mean(ego_motion_pred, axis=0) - ego_motion_obs) ** 2))
        return L_ego


def bbox_loss(all_trackers, all_observed_boxes):
    '''
    Compute the bbox loss using IOU between predicted boxes and observed boxes
    '''
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