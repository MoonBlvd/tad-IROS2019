'''
Load odometry files (.txt) and save (yaw, x, z) to .npy files
The loaded odometry data are transition matrixs Tcws at each time frame
This script contains follwoing prcoess:
    1. Read Tcws from txt file and convert to (tiem, 4, 4) numpy array
    2. Filter the Tcws by moving average
    3. From the Tcws, compute the ego cars (x,z) coordinate as well as yaw angle
    4. yaw, x, z are filtered again 
    5, numpy array containing (yaw, x, z) are saved as .npy files

Feb 22 2019
'''

import sys
sys.path.append('../')
import numpy as np
from lib.utils.data_prep_utils import *
import argparse
import glob

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file", default=None, help="directory to the file where all sample names are saved")
    
    parser.add_argument("-d","--odo_root",help="root directory that all odometry .txt files are saved")
    parser.add_argument("-o","--output_dir",help="directory to save the ego motion .npy file")
    args = parser.parse_args()

    if args.input_file is not None:
        with open(args.input_file, 'r') as f:
            for row in f:
                video_name = row.strip('\n')
                odo_file_name = video_name + '.txt'
                output_name = video_name + '.npy'

                # filter the training file but not the testing?
                Tcws = load_vo(os.path.join(args.odo_root, odo_file_name),
                                filter='moving average')
                
                all_xyz = Tcws_to_xyz(Tcws, decimals=3)
                # all_xyz = ego_motion_filtering(all_xyz, window_size=11)
                
                all_yaws =Tcws_to_yaw(Tcws, decimals=3)
                # all_yaws = ego_motion_filtering(all_yaws, window_size=11)

                ego_motion = np.hstack([all_yaws, all_xyz[:,[0,2]]])

                np.save(os.path.join(args.output_dir,output_name), ego_motion)
    
    else:
        all_odo_files = sorted(glob.glob(os.path.join(args.odo_root, '*.txt')))
        print("Number of odometry files: ", len(all_odo_files))

        for odo_file in all_odo_files:
            video_name = odo_file.split('/')[-1].split('.')[0]
            output_name = video_name + '.npy'

            # filter the training file but not the testing?
            Tcws = load_vo(os.path.join(odo_file),
                            filter='moving average')
            
            all_xyz = Tcws_to_xyz(Tcws, decimals=3)
            # all_xyz = ego_motion_filtering(all_xyz, window_size=11)
            
            all_yaws =Tcws_to_yaw(Tcws, decimals=3)
            # all_yaws = ego_motion_filtering(all_yaws, window_size=11)

            ego_motion = np.hstack([all_yaws, all_xyz[:,[0,2]]])

            np.save(os.path.join(args.output_dir,output_name), ego_motion)