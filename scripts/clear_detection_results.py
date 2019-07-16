'''
Feb 19 2018
1.
Clear the Mask-RCNN detection results before running deep-sort.
Original reults include following classes
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench']

Following rules are applied:
    1. Object classes we care abou： 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
        Their corresponding ids are: [1, 2, 3, 4, 6, 8]
    2. Objects with bounding box min(H,W) < 20 are removed
    3. Boxes locate at the image bottom with width close to 720 is ego car box and should be removed
2. 
Clear the deep-sort results
Following rules are applied:
    1. Trackings with less than 5 time steps are removed

'''
import os
import numpy as np
import glob
import argparse

CLASSES_KEEP = [1,2,3,4,6,8]
WIDTH_THRESH = 20
def clear_detection(detections):
    '''
    only keep boxesthat are specific classes
    det: [frame_ids, track_ids, x, y, w, h, classes, scores, features]
    '''
    new_det = []
    for det in detections:
        if det[6] not in CLASSES_KEEP:
            '''box class not interesting'''
            continue
        elif min(det[4:6]) < WIDTH_THRESH:
            '''box too small'''
            continue
        elif det[4] > 800 and abs((det[2] + det[4]/2) - 640) < 100 and (det[3] + det[5]/2) > 450:
            '''Highly possible that the box is ego car'''
            continue
        else:
            new_det.append(det)

    return np.array(new_det)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--detection_dir", required=True, 
                            help="directory where all maskrcnn numpy arries saved")
    parser.add_argument("-o", "--output_dir", required=True,
                            help="directory to save cleared detection numpy arries")
    args = parser.parse_args()

    # detection_dir = '/media/DATA/AnAnAccident_Detection_Dataset/mask_rcnn_detections'
    # output_dir = '/media/DATA/AnAnAccident_Detection_Dataset/mask_rcnn_detections_clear'
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    all_detection_files = sorted(glob.glob(os.path.join(args.detection_dir, '*.npy')))
    print("Number of detection files found: ", len(all_detection_files))
    for detection_file in all_detection_files:
        print(detection_file)
        numpy_name = detection_file.split('/')[-1]
        detections = np.load(detection_file)
        cleared_detections = clear_detection(detections)

        np.save(os.path.join(args.output_dir, numpy_name), cleared_detections)