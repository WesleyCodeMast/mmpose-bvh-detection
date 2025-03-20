from annotation import annotation
import shutil
from pathlib import Path
import argparse
import sys
import os

# Add the _internal directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '_internal'))

annotation_dir = Path(f'character')
if annotation_dir.exists():
    shutil.rmtree(annotation_dir)

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", type=str, help="The input image file.")
args = parser.parse_args()
image_file = args.image

annotation.run(image_file, annotation_dir)
