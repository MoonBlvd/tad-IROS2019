import os
import subprocess
import argparse
import glob

class FFMPEGFrames:
    def __init__(self, output, ext):
        self.output = output
        self.ext = ext
    def extract_frames(self, input, fps):
        output = input.split('/')[-1].split('.')[0]
        
        output_dir = os.path.join(self.output, output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            print(input + " has been processed!")
            return 

        #output_dir = os.path.join(output_dir, 'images')
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

        query = "ffmpeg -i " + input + " -vf fps=" + str(fps) + " -qscale:v 0 " + output_dir + "/%06d." + self.ext
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        s = str(response).encode('utf-8')

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_dir", required=True, help="the directory to the video or videos")
ap.add_argument("--video_key_file", help="the directory to the file of a list of video keys")
ap.add_argument("-f", "--fps", required=True, help="the target fps of the extracted frames")
ap.add_argument("-o", "--out_dir", required=True, help="the output directory")
ap.add_argument("-e", "--ext", required=True, help="the extension of the saved image")

args = vars(ap.parse_args())

input_video_dir = args["video_dir"]
fps = args["fps"]
out_root = args["out_dir"]
ext = args["ext"]
f = FFMPEGFrames(out_root, ext)
print("video key file: ", args['video_key_file'])
try:
    all_video_names = []
    file = open(args['video_key_file']) 
    for line in file:
        all_video_names.append(input_video_dir + line[:-1])
except:
    all_video_names = sorted(glob.glob(os.path.join(input_video_dir, '*')))

print("Number of video: ", len(all_video_names))
assert len(all_video_names) > 0

for video_name in all_video_names:
#    out_name = video_name.split('/')[-1][:-4]
#    out_dir = out_root + out_name + '/' 

    f.extract_frames(video_name, fps)
