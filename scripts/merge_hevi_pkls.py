import os
import pickle as pkl
import numpy as np
import glob


INPUT_DIR = '../../../fvl2019icra-keras/data/val_nw'
OUTPUT_DIR = '/media/DATA/HEVI_dataset/fvl_data/val'
ego_data_root = '/media/DATA/HEVI_dataset/ego_motion/val'
stride = 1
all_segments =  sorted(glob.glob(os.path.join(INPUT_DIR,'*.pkl')))
print("Num all pkls: ", len(all_segments))


segment_data = {}
num_clips = 0
prev_segment_name = None
for idx, segment_file in enumerate(all_segments):
    video_name = segment_file.split('/')[-1].split('_')[0]
    track_id = segment_file.split('/')[-1].split('_')[1]
    segment_name = video_name + '_' + track_id
    # print(segment_file)

    # save all and go to next car if the segment changes    
    if (segment_name != prev_segment_name and num_clips != 0) or idx == len(all_segments)-1:
        '''append future observations to the sequence as well, note that we didnt have future flow!!!'''
        segment_data['bbox'] = np.vstack([segment_data['bbox'], data['target']])
        # segment_data['ego_motion'] = np.vstack([segment_data['ego_motion'], data['future_ego_motion']])
        try:
            # segment_data['ego_motion'] =  ego_motion_session[segment_data['frame_id'],:]
            extended_ego_motion_id = np.append(segment_data['frame_id'], 
                                                np.arange(segment_data['frame_id'][-1]+1, segment_data['frame_id'][-1]+11))
            segment_data['ego_motion'] =ego_motion_session[extended_ego_motion_id,:]
            output_file = os.path.join(OUTPUT_DIR, prev_segment_name+'.pkl')
            pkl.dump(segment_data, open(output_file,'wb'))
            # print("One session saved!")
        except:
            # usable_id = np.where(segment_data['frame_id'] < len(ego_motion_session))[0]
            # segment_data['frame_id'] = segment_data['frame_id'][usable_id]
            # segment_data['bbox'] = segment_data['bbox'][usable_id,:]
            # segment_data['flow'] = segment_data['flow'][usable_id,:,:,:]
            # segment_data['ego_motion'] =  ego_motion_session[segment_data['frame_id'],:]
            # print(video_name)
            # raise NameError(video_name)
            print('skiped one car')
            print(video_name)
            # output_file = os.path.join(OUTPUT_DIR, prev_segment_name+'.pkl')
            # pkl.dump(segment_data, open(output_file,'wb'))
        
        segment_data = {}
        num_clips = 0
    # segment_data['target'] = []
    with open(segment_file,'rb') as f:
        '''load one car's one segment'''
        data = pkl.load(f)
    '''load corresponding ego motion file, it is a long ego motion file'''    
    ego_session_file = os.path.join(ego_data_root, video_name+'.npy')
    ego_motion_session = np.load(ego_session_file) # a (time, 3) numpy array of [yaw, x, z]
    
    # data['ego_motion'] = np.array(data['ego_motion'])
    if num_clips == 0:
        segment_data['bbox'] = data['bbox'] # (10,4)
        segment_data['flow'] = data['flow'] # (10,5,5,2)
        # segment_data['ego_motion'] = data['ego_motion'] # (10, 6)
        segment_data['frame_id'] = data['frame_id']
        
    else:
        segment_data['bbox'] = np.vstack([segment_data['bbox'], data['bbox'][-1:,:]])
        segment_data['flow'] = np.vstack([segment_data['flow'], data['flow'][-1:,:]])
        # segment_data['ego_motion'] = np.vstack([segment_data['ego_motion'], data['ego_motion'][-1:,:]])
        segment_data['frame_id'] = np.append(segment_data['frame_id'], data['frame_id'][-1:])
    
    # data['future_ego_motion']
    # data['target']

    prev_segment_name = segment_name
    num_clips += 1    
