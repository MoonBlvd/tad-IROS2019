import sys
import os
import glob
import argparse
import yaml
import cv2
import youtube_dl

parser = argparse.ArgumentParser(description='AnAnXingChe video downloader parameters.')
parser.add_argument('--download_dir', required=True, help='target directory to save downloaded videos')
parser.add_argument('--url_file', required=True, help='a .txt file saving urls to all videos')
parser.add_argument('--to_images', type=bool, default=False, help='downsample the video and save image frames, default is false')
parser.add_argument('--img_dir', help='target directory to save downsampled')
parser.add_argument('--img_ext', default='jpg', help='image extension')
parser.add_argument('--downsample_rate', default=3.0, type=float, help='downsample rate')

args = parser.parse_args()

DOWNLOAD_DIR = args.download_dir


# Download videos
if not os.path.isdir(DOWNLOAD_DIR):
    print("The indicated download directory does not exist!")
    print("Directory made!")
    os.makedirs(DOWNLOAD_DIR)

'''Download videos'''
ydl_opt = {'outtmpl': DOWNLOAD_DIR + '%(id)s.%(ext)s',
           'format': 'mp4'}
ydl = youtube_dl.YoutubeDL(ydl_opt)
'''
with ydl:
    result = ydl.extract_info(
        'https://www.youtube.com/channel/UC-Oa3wml6F3YcptlFwaLgDA',
        download=True # We just want to extract the info
    )
'''
url_list = open(args.url_file,'r').readlines()
ydl.download(url_list)
print("Download finished!")

all_videos = sorted(glob.glob(DOWNLOAD_DIR + '*.mp4'))
print("Number of videos: ", len(all_videos))

if args.to_images:
    IMAGE_DIR = args.img_dir
    # Downsample the saved videos and save images to another directory
    try:
        os.stat(IMAGE_DIR)
    except:
        print("The indicated image directory does not exist!")
        print("Directory made!")
        os.mkdir(IMAGE_DIR)

    downsample_rate = args.downsample_rate
    for video_idx, file_name in enumerate(all_videos):
        video_name = file_name.split('/')[-1][:-4]
        image_dir = OUT_DIR + video_name + '/'
        try:
            os.stat(image_dir)
        except:
            os.mkdir(image_dir)

        cap = cv2.VideoCapture(file_name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        print("Number of frames: ", length)
        print("FPS: ", fps)
        i = 0
        j = 0
        while True:
            # Capture frame-by-frame
            ret, image = cap.read()
            if not ret:
                break
            if i%downsample_rate == 0:
                img_name = str(format(j,'06')) + '.' + args.img_ext
                j += 1
                cv2.imwrite(image_dir + img_name, image)
            i += 1
