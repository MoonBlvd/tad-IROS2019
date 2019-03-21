import numpy as np
# import cv2
import pickle as pkl
import os
import glob
import copy
import pickle as pkl


def bbox_normalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] /= W
    new_bbox[:,1] /= H
    new_bbox[:,2] /= W
    new_bbox[:,3] /= H
    
    return new_bbox

def bbox_denormalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] *= W
    new_bbox[:,1] *= H
    new_bbox[:,2] *= W
    new_bbox[:,3] *= H
    
    return new_bbox

# FLow loading code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

def load_flow(flow_folder):
    '''
    Given video key, load the corresponding flow file
    '''
    flow_files = sorted(glob.glob(flow_folder + '*.flo'))
    flows = []
    for file in flow_files:
        flow = read_flo(file)
        flows.append(flow)
    return flows

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = int(np.fromfile(f, np.int32, count=1))
    h = int(np.fromfile(f, np.int32, count=1))
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()

    return flow

def load_vo(odo_file, filter='moving average'):
    '''
    Params:
        odo_file: the directory to the .txt odometry file. The file must be in ORBSLAM2 format
    Returns:
        Tcws: (time, 4, 4) an array of 4X4 transition matrix at each timestep
    '''
    Tcws = []
    num_matched_odo = 0
    
    with open(odo_file) as file:
        num_lines = sum(1 for line in open(odo_file))
        start = None
        end = None
        for idx, line in enumerate(file):
            matrix = np.zeros((4,4))
            if line is '\n':
                if start is None:
                    start = idx
                    end = None
            else:
                end = idx
                for i, element in enumerate(line.split()):
                    matrix[int(i/4), int(i%4)] = float(element)
            Tcws.append(matrix)

            if start is not None and (end is not None or idx+1 == num_lines):
                if end is None:
                    end = idx+1
                    ''''Filter the pervious Tcws before expanding to the end'''
                    Tcws = np.array(Tcws)
                    if filter == 'moving average':
                        Tcws[:start] = ego_motion_filtering(Tcws[:start], 
                                                            decimals=3, 
                                                            window_size=5,
                                                            filter=filter)
                    else:
                        pass

#                 print("fixing lines from " + str(start+1) + " to " + str(end+1))
                Tcws = fix_odo_data(start, end, num_lines, Tcws)
                start = None
    return np.array(Tcws)

def fix_odo_data(start, end, num_lines, Tcws):
    '''There are some missed values so that the vo data needs to be fixed.'''
    if start == 0:
        for i in range(end):
            Tcws[i] = np.eye(4)
    else:
        # compute the average derivative 
        avg_derivative = np.zeros_like(Tcws[0])
        for i in range(1,2):
            try:
                derivative = np.linalg.inv(Tcws[start-i-1]).dot(Tcws[start-i])
                avg_derivative += derivative
            except:
                break
        avg_derivative = avg_derivative/(i)
        
        # expand from the current time using derivative
        for i in range(start, end):
            Tcws[i] = Tcws[i-1].dot(avg_derivative)
    return Tcws

def shift_odo_data(Tcws):
    '''
    project a set of Tcws to the first frame
    :Params:
        Tcws: a (time, 4, 4) numpy array of transformation matrics, 
        or (time,6) with x
    :Return:
        new_Tcws: a (time, 4, 4) numpy array of shifted transformation matrics
    '''
    if Tcws.shape[1] == 4:
        compensation = np.linalg.inv(Tcws[0])
        for i in range(Tcws.shape[0]):
            Tcws[i] = compensation.dot(Tcws[i])
    elif Tcws.shape[1] == 6:
        Tcws = Tcws-Tcws[0,:]
    else:
        raise ValueError("Wrong odometry data shape!")
        
    return Tcws

def vector_from_T(Tcws):
    '''
    Compute ego motion vector from given tranformation matrix
    :Params:
        Tcws: a (time, 4, 4) numpy array of transformation matrics
    :Return:
        ego_motion: a (time,6) numpy array of ego motion vector [yaw, pitch, roll, x, y, z]
    '''
    ego_motion = np.zeros((Tcws.shape[0],6))
    for i in range(Tcws.shape[0]):
        sy = np.sqrt(Tcws[i][0,0] * Tcws[i][0,0] +  Tcws[i][1,0] * Tcws[i][1,0])
        
        singular = sy < 1e-6
 
        if  not singular :
    #         yaw = -np.arcsin(Tcws[i][2,0]) # ccw is positive, in [-pi/2, pi/2]
            yaw = np.arctan2(-Tcws[i][2,0], sy) # ccw is positive, in [-pi/2, pi/2]
            pitch = np.arctan2(Tcws[i][2,1], Tcws[i][2,2]) # small value, in [-pi/2, pi/2]
            roll = np.arctan2(Tcws[i][1,0], Tcws[i][0,0]) # small value, in [-pi/2, pi/2]
        else:
            yaw = np.atan2(-Tcws[i][2,0], sy)
            pitch = np.atan2(-Tcws[i][1,2], Tcws[i][1,1])
            roll = 0
        
        ego_motion[i,0] = yaw
        ego_motion[i,1] = pitch
        ego_motion[i,2] = roll
        
        ego_motion[i,3:] = -Tcws[i,:3,3]     
        
    return ego_motion

def Tcws_to_xyz(Tcws, decimals=None):
    '''
    Tcws is s a rigid body transformation that transforms points from the world to the camera coordinate system
    '''
    all_xyz = []
    for i in range(len(Tcws)):
        car_xyz = np.linalg.inv(Tcws[i]).dot(np.array([0,0,0,1]))
        all_xyz.append(car_xyz)
    all_xyz = np.array(all_xyz)
    return np.around(all_xyz, decimals)

def Tcws_to_yaw(Tcws, decimals=None):
    '''
    Tcws is s a rigid body transformation that transforms points from the world to the camera coordinate system
    '''
    all_yaws = []
    for i in range(len(Tcws)):
        sy = np.sqrt(Tcws[i][0,0] * Tcws[i][0,0] +  Tcws[i][1,0] * Tcws[i][1,0])
        singular = sy < 1e-6
        if not singular :
    #         yaw = -np.arcsin(Tcws[i][2,0]) # ccw is positive, in [-pi/2, pi/2]
            yaw = np.arctan2(-Tcws[i][2,0], sy) # ccw is positive, in [-pi/2, pi/2]
            all_yaws.append(yaw)
    all_yaws = np.expand_dims(np.array(all_yaws), axis=1)
    return np.around(all_yaws, decimals)

def ego_motion_filtering(ego_motion, decimals=3, window_size=5,filter='moving average'):
    '''
    Smooth the computed ego motion vector
    Params: 
        ego_motion: (timesteps, 3) # yaw, x, z
    Returns:
        filtered_ego_motion: (timesteps, 3)
    '''    
    if filter == 'moving average':
        width = int((window_size-1)/2)
        for i in range(len(ego_motion)):
            # if i >= width and i+width < len(ego_motion):
            start = int(max([0, i - width]))
            end = int(min([i + width, len(ego_motion)]))

            ego_motion[i,:] = np.mean(ego_motion[start:end,:], axis=0)
        
    return np.round(ego_motion, decimals=decimals)

def shrink_flow_box(bbox):
    '''
        Shrink the bbox and compute the mean optical flow
        :Param: flow size is (h,w,2) containing two direction dense flow
        :Param: bbox format is:  [cx,cy,w,h]
        :return: [cx,cy,w,h]
    '''
    bbox_shrink = 1.2#0.8
#     cx = (bbox[1]+bbox[3])/2
#     cy = (bbox[0]+bbox[2])/2
#     w = bbox[3]-bbox[1]
#     h = bbox[2]-bbox[0]
    cx = bbox[0]
    cy = bbox[1]
    w = bbox[2]
    h = bbox[3]

    
    return np.array([[cx,cy,w*bbox_shrink,h*bbox_shrink]])

def roi_pooling_opencv(boxes,image,size=[5,5]):
    """Crop the image given boxes and resize with bilinear interplotation.
    :Params:
        image: Input image of shape (1, image_height, image_width, n_channels).
                The shape can be bigger or smaller than tehe 
        boxes: ROI of shape (num_boxes, 4) in range of [0,1]
                each row [cx, cy, w, h]
        size: Fixed size [h, w], e.g. [7, 7], for the output slices.
        W, H: width and height or original image
    :Returns:
        4D Tensor (number of regions, slice_height, slice_width, channels)
    """

    w = image.shape[2]
    h = image.shape[1]
    n_channels = image.shape[3]

    xmin = boxes[:,0]-boxes[:,2]/2
    xmax = boxes[:,0]+boxes[:,2]/2
    ymin = boxes[:,1]-boxes[:,3]/2
    ymax = boxes[:,1]+boxes[:,3]/2 
    

    ymin = np.max([0, int(h * ymin)])
    ymax = np.min([h, int(h * ymax)])
    
    xmin = np.max([0, int(w * xmin)])
    xmax = np.min([w, int(w * xmax)])
    

    size = (size[0], size[1])
    return np.expand_dims(cv2.resize(image[0,ymin:ymax, xmin:xmax,:], size), axis=0)

    #     # print(boxes)
    #     # raise ValueError('ymin:%d, ymax:%d, xmin:%d, xmax:%d'%(ymin, ymax, xmin, xmax))
        # print('ymin:%d, ymax:%d, xmin:%d, xmax:%d'%(ymin, ymax, xmin, xmax))
    #     # print("boxes: ",boxes)
    #     size = [size[0],size[1], n_channels]
    #     return None #np.expand_dims(np.zeros(size), axis=0)
        

