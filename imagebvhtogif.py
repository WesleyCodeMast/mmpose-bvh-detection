import cv2
import numpy as np
import os
from pathlib import Path
import json
import pathlib
import shutil
from annotation import annotation
import argparse
import sys
import os


# Add the _internal directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '_internal'))

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

json_dir = Path(f'output_json')

# image_dir = Path(f'input_image')
# file = '1.png'
# image_file = os.path.join(image_dir, file)

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", type=str, help="The input image file.")

parser.add_argument("-b", "--bvh", type=str, help="The input bvh file.")

args = parser.parse_args()

image_file = args.image

bvh_file = args.bvh

annotation_dir = Path(f'character')

output_dir = Path(f'output')
if output_dir.exists():
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
if json_dir.exists():
    shutil.rmtree("output_json")

annotation.run(image_file, annotation_dir)

shutil.copy(bvh_file, os.path.join(output_dir, 'result.bvh'))
from animated_drawings import render
render.start('./config/config/mvc/interactive_window_example.yaml')

