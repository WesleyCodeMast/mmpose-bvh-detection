from pose_estimator_3d import estimator_3d
from utils import smooth, camera
from bvh_skeleton import cmu_skeleton

import cv2
import numpy as np
import os
from pathlib import Path
import json
import pathlib
import shutil
import subprocess
import argparse
from animated_drawings import render

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--video", type=str, help="The input video file.")
parser.add_argument("-i", "--image", type=str, help="The input image file.")

args = parser.parse_args()

video_file = args.video
image_file = args.image

json_dir = Path(f'output_json')

output_dir = Path(f'output')
if output_dir.exists():
    shutil.rmtree(output_dir)
    
os.makedirs(output_dir)
if json_dir.exists():
    shutil.rmtree("output_json")

os.system(rf'annotate.exe -i {image_file}')
os.system(rf'bin\OpenPoseDemo.exe --video {video_file} --write_json output_json/')

cap = cv2.VideoCapture(str(video_file))
keypoints_list = []
img_width, img_height = None, None
ret, frame = cap.read()
img_height = frame.shape[0]
img_width = frame.shape[1]
cap.release()

keypoints_list = []
for json_files in os.listdir("output_json"):
    with open(rf'output_json\{json_files}', 'r') as file:
        data = json.load(file)["people"][0]['pose_keypoints_2d']
    keypoints_list.append(data)
keypoints_list = np.array(keypoints_list).reshape((len(keypoints_list), 25, 3)).astype("float32")

keypoints_list = smooth.filter_missing_value(
    keypoints_list=keypoints_list,
    method='ignore'
)

pose2d = np.stack(keypoints_list)[:, :, :2]
pose2d_file = Path(output_dir / '2d_pose.npy')
np.save(pose2d_file, pose2d)

e3d = estimator_3d.Estimator3D(
    config_file='models/openpose_video_pose_243f/video_pose.yaml',
    checkpoint_file='models/openpose_video_pose_243f/best_58.58.pth'
)

pose2d = np.load(pose2d_file)
pose3d = e3d.estimate(pose2d, image_width=img_width, image_height=img_height)

subject = 'S1'
cam_id = '55011271'
cam_params = camera.load_camera_params('./cameras.h5')[subject][cam_id]
R = cam_params['R']
T = 0

pose3d_world = camera.camera2world(pose=pose3d, R=R, T=T)
pose3d_world[:, :, 2] -= np.min(pose3d_world[:, :, 2])

pose3d_file = output_dir / '3d_pose.npy'
np.save(pose3d_file, pose3d_world)

bvh_file = output_dir / f'result.bvh'
cmu_skel = cmu_skeleton.CMUSkeleton()
channels, header = cmu_skel.poses2bvh(pose3d_world, output_file=bvh_file)

render.start('./config/config/mvc/interactive_window_example.yaml')
