'''
Read the pickle file that saves label of A3D dataset.
Save the images of each short video clip separately and prepare to run MaskRCNN and flownet2

Feb 19 2019

Assume each video's frames are saved in: ROOT_DIR + '/images/xxxxxx'

'''
import os
import pickle as pkl
import shutil
import argparse

parser = argparse.ArgumentParser(description='A3D video split parameters.')
parser.add_argument('--root_dir', required=True, help='the root directory of the dataset')
parser.add_argument('--label_dir', required=True, help='the pkl label file')
args = parser.parse_args()

data = pkl.load(open(args.label_dir,'rb'))
for key, value in data.items():
    video_name = key
    video_dir = os.path.join(args.root_dir, 'images', value['video_name'])

    out_dir = os.path.join(args.root_dir, 'frames', video_name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.join(out_dir, 'images')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    start = value['clip_start']
    end = value['clip_end']
    for new_i, old_i in enumerate(range(int(start), int(end)+1)):
        img_name = str(old_i).zfill(6) + '.jpg'
        old_img_path = os.path.join(video_dir, img_name)
        new_img_path = os.path.join(out_dir, str(new_i + 1).zfill(6) + '.jpg')
        shutil.copy(old_img_path, new_img_path)

    
