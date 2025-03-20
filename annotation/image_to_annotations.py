# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import sys
# import cv2
# import numpy as np
# from skimage import measure
# from scipy import ndimage
# from pathlib import Path
# import yaml
# import logging
# from .handler import detect, pose

# def image_to_annotations(img_fn: str, out_dir: str) -> None:
#     """
#     Given the RGB image located at img_fn, runs detection, segmentation, and pose estimation for drawn character within it.
#     Crops the image and saves texture, mask, and character config files necessary for animation. Writes to out_dir.

#     Params:
#         img_fn: path to RGB image
#         out_dir: directory where outputs will be saved
#     """

#     # create output directory
#     outdir = Path(out_dir)
#     outdir.mkdir(exist_ok=True)

#     # read image
#     img = cv2.imread(img_fn)

#     # copy the original image into the output_dir
#     cv2.imwrite(str(outdir/'image.png'), img)

#     # ensure it's rgb
#     if len(img.shape) != 3:
#         msg = f'image must have 3 channels (rgb). Found {len(img.shape)}'
#         logging.critical(msg)
#         assert False, msg

#     # resize if needed
#     if np.max(img.shape) > 1000:
#         scale = 1000 / np.max(img.shape)
#         img = cv2.resize(img, (round(scale * img.shape[1]), round(scale * img.shape[0])))

#     detection_results = detect(img)
#     # # convert to bytes and send to torchserve
#     # img_b = cv2.imencode('.png', img)[1].tobytes()
#     # request_data = {'image': img_b}
#     # resp = requests.post("http://localhost:5000/detect", files=request_data, verify=False)
#     # if resp is None or resp.status_code >= 300:
#     #     raise Exception(f"Failed to get bounding box, please check if the 'docker_torchserve' is running and healthy, resp: {resp}")

#     # detection_results = json.loads(resp.content)
#     print("detection_result ==> ", detection_results)
#     # error check detection_results
#     if isinstance(detection_results, dict) and 'code' in detection_results.keys() and detection_results['code'] == 404:
#         assert False, f'Error performing detection. Check that drawn_humanoid_detector.mar was properly downloaded. Response: {detection_results}'

#     # order results by score, descending
#     detection_results.sort(key=lambda x: x['score'], reverse=True)

#     # if no drawn humanoids detected, abort
#     if len(detection_results) == 0:
#         msg = 'Could not detect any drawn humanoids in the image. Aborting'
#         logging.critical(msg)
#         assert False, msg

#     # otherwise, report # detected and score of highest.
#     msg = f'Detected {len(detection_results)} humanoids in image. Using detection with highest score {detection_results[0]["score"]}.'
#     logging.info(msg)

#     # calculate the coordinates of the character bounding box
#     bbox = np.array(detection_results[0]['bbox'])
#     l, t, r, b = [round(x) for x in bbox]
#     r = r + 2
#     # dump the bounding box results to file
#     with open(str(outdir/'bounding_box.yaml'), 'w') as f:
#         yaml.dump({
#             'left': l,
#             'top': t,
#             'right': r,
#             'bottom': b
#         }, f)
#     r = r + 2
#     # crop the image
#     cropped = img[t:b, l:r]

#     # get segmentation mask
#     # mask = segment(cropped)
#     mask = segment(img, t, b, l, r)

#     pose_results = pose(cropped)
#     # # send cropped image to pose estimator
#     # data_file = {'image': cv2.imencode('.png', cropped)[1].tobytes()}
#     # resp = requests.post("http://localhost:5000/pose", files=data_file, verify=False)
#     # if resp is None or resp.status_code >= 300:
#     #     raise Exception(f"Failed to get skeletons, please check if the 'docker_torchserve' is running and healthy, resp: {resp}")

#     # pose_results = json.loads(resp.content)
#     print("pose_result ==> ", pose_results)
#     # error check pose_results
#     if isinstance(pose_results, dict) and 'code' in pose_results.keys() and pose_results['code'] == 404:
#         assert False, f'Error performing pose estimation. Check that drawn_humanoid_pose_estimator.mar was properly downloaded. Response: {pose_results}'

#     # if no skeleton detected, abort
#     if len(pose_results) == 0:
#         msg = 'Could not detect any skeletons within the character bounding box. Expected exactly 1. Aborting.'
#         logging.critical(msg)
#         assert False, msg

#     # if more than one skeleton detected,
#     if 1 < len(pose_results):
#         msg = f'Detected {len(pose_results)} skeletons with the character bounding box. Expected exactly 1. Aborting.'
#         logging.critical(msg)
#         assert False, msg

#     # get x y coordinates of detection joint keypoints
#     kpts = np.array(pose_results[0]['keypoints'])[:, :2]

#     # use them to build character skeleton rig
#     skeleton = []
#     skeleton.append({'loc' : [round(x) for x in (kpts[11]+kpts[12])/2], 'name': 'root'          , 'parent': None})
#     skeleton.append({'loc' : [round(x) for x in (kpts[11]+kpts[12])/2], 'name': 'hip'           , 'parent': 'root'})
#     skeleton.append({'loc' : [round(x) for x in (kpts[5]+kpts[6])/2  ], 'name': 'torso'         , 'parent': 'hip'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[0]             ], 'name': 'neck'          , 'parent': 'torso'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[6]             ], 'name': 'right_shoulder', 'parent': 'torso'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[8]             ], 'name': 'right_elbow'   , 'parent': 'right_shoulder'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[10]            ], 'name': 'right_hand'    , 'parent': 'right_elbow'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[5]             ], 'name': 'left_shoulder' , 'parent': 'torso'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[7]             ], 'name': 'left_elbow'    , 'parent': 'left_shoulder'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[9]             ], 'name': 'left_hand'     , 'parent': 'left_elbow'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[12]            ], 'name': 'right_hip'     , 'parent': 'root'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[14]            ], 'name': 'right_knee'    , 'parent': 'right_hip'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[16]            ], 'name': 'right_foot'    , 'parent': 'right_knee'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[11]            ], 'name': 'left_hip'      , 'parent': 'root'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[13]            ], 'name': 'left_knee'     , 'parent': 'left_hip'})
#     skeleton.append({'loc' : [round(x) for x in  kpts[15]            ], 'name': 'left_foot'     , 'parent': 'left_knee'})

#     # create the character config dictionary
#     char_cfg = {'skeleton': skeleton, 'height': cropped.shape[0], 'width': cropped.shape[1]}

#     # convert texture to RGBA and save
#     cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
#     cv2.imwrite(str(outdir/'texture.png'), cropped)

#     # save mask
#     cv2.imwrite(str(outdir/'mask.png'), mask)

#     # dump character config to yaml
#     with open(str(outdir/'char_cfg.yaml'), 'w') as f:
#         yaml.dump(char_cfg, f)

#     # create joint viz overlay for inspection purposes
#     joint_overlay = cropped.copy()
#     for joint in skeleton:
#         x, y = joint['loc']
#         name = joint['name']
#         cv2.circle(joint_overlay, (int(x), int(y)), 5, (0, 0, 0), 5)
#         cv2.putText(joint_overlay, name, (int(x), int(y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)
#     cv2.imwrite(str(outdir/'joint_overlay.png'), joint_overlay)


# def segment(img: np.ndarray, top, bottom, left, right):
#     """ threshold """
#     img = np.min(img, axis=2)
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
#     img = cv2.bitwise_not(img)

#     stack = [(0, 0)]
#     real_stack = [(0, 0)]
#     h, w = img.shape
#     mask = np.full(img.shape, 255, np.uint8)
#     # Directions for 8-connected neighbors
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (0, -1),          (0, 1),
#                   (1, -1), (1, 0), (1, 1)]

#     # Process the stack
#     while stack:
#         x, y = stack.pop()
#         if img[x, y] == 0:  # If the current pixel is black
#             img[x, y] = 255  # Mark it white

#             # Add valid neighbors to the stack
#             for dx, dy in directions:
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < h and 0 <= ny < w and img[nx, ny] == 0:
#                     real_stack.append((x, y))

#                     stack.append((nx, ny))
    
#     for x, y in real_stack:
#         mask[x, y] = 0

#     # cv2.imshow("mig", mask[top:bottom, left:right])

#     mask = mask[top:bottom, left:right]

#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return mask


# if __name__ == '__main__':
#     log_dir = Path('./logs')
#     log_dir.mkdir(exist_ok=True, parents=True)
#     logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

#     img_fn = sys.argv[1]
#     out_dir = sys.argv[2]
#     image_to_annotations(img_fn, out_dir)








# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from pathlib import Path
import yaml
import logging
from .handler import detect, pose

def image_to_annotations(img_fn: str, out_dir: str) -> None:
    """
    Given the RGB image located at img_fn, runs detection, segmentation, and pose estimation for drawn character within it.
    Crops the image and saves texture, mask, and character config files necessary for animation. Writes to out_dir.

    Params:
        img_fn: path to RGB image
        out_dir: directory where outputs will be saved
    """

    # create output directory
    outdir = Path(out_dir)
    outdir.mkdir(exist_ok=True)

    # read image
    img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)
    # copy the original image into the output_dir
    if img is None:
        print("Error: Image not loaded. Check the file path.")
        return
    cv2.imwrite(str(outdir/'image.png'), img)

    # resize if needed
    if np.max(img.shape) > 1000:
        scale = 1000 / np.max(img.shape)
        img = cv2.resize(img, (round(scale * img.shape[1]), round(scale * img.shape[0])))
    
    white_background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
    if len(cv2.split(img)) != 4 and len(cv2.split(img)) != 3:
        msg = f'image must have 3 or 4 channels (rgb). Found {len(img.shape)}'
        logging.critical(msg)
        assert False, msg

    # ensure it's rgb
    if len(cv2.split(img)) == 4:

        b, g, r, a = cv2.split(img)
        print(f" here is image shape ************{img.shape} {white_background.shape}")
        a = a / 255.0
        for c in range(3):
            white_background[:, :, c] = (1 - a) * white_background[:, :, c] + a * img[:, :, c]
    elif len(cv2.split(img)) == 3:
        for c in range(3):
            white_background[:, :, c] = img[:, :, c]
    
    cv2.imwrite(str(outdir/'image_with_bg.png'), white_background)

    detection_results = detect(white_background)

    print("detection_result ==> ", detection_results)
    # error check detection_results
    if isinstance(detection_results, dict) and 'code' in detection_results.keys() and detection_results['code'] == 404:
        assert False, f'Error performing detection. Check that drawn_humanoid_detector.mar was properly downloaded. Response: {detection_results}'

    # order results by score, descending
    detection_results.sort(key=lambda x: x['score'], reverse=True)

    # if no drawn humanoids detected, abort
    if len(detection_results) == 0:
        msg = 'Could not detect any drawn humanoids in the image. Aborting'
        logging.critical(msg)
        assert False, msg

    # otherwise, report # detected and score of highest.
    msg = f'Detected {len(detection_results)} humanoids in image. Using detection with highest score {detection_results[0]["score"]}.'
    logging.info(msg)

    # calculate the coordinates of the character bounding box
    bbox = np.array(detection_results[0]['bbox'])
    l, t, r, b = [round(x) for x in bbox]

    # dump the bounding box results to file
    with open(str(outdir/'bounding_box.yaml'), 'w') as f:
        yaml.dump({
            'left': l,
            'top': t,
            'right': r,
            'bottom': b
        }, f)

    # crop the image
    cropped = img[t:b, l:r]
    cropped1 = white_background[t:b, l:r]

    # get segmentation mask
    mask = segment(cropped1)

    pose_results = pose(cropped1)

    print("pose_result ==> ", pose_results)
    # error check pose_results
    if isinstance(pose_results, dict) and 'code' in pose_results.keys() and pose_results['code'] == 404:
        assert False, f'Error performing pose estimation. Check that drawn_humanoid_pose_estimator.mar was properly downloaded. Response: {pose_results}'

    # if no skeleton detected, abort
    if len(pose_results) == 0:
        msg = 'Could not detect any skeletons within the character bounding box. Expected exactly 1. Aborting.'
        logging.critical(msg)
        assert False, msg

    # if more than one skeleton detected,
    if 1 < len(pose_results):
        msg = f'Detected {len(pose_results)} skeletons with the character bounding box. Expected exactly 1. Aborting.'
        logging.critical(msg)
        assert False, msg

    # get x y coordinates of detection joint keypoints
    kpts = np.array(pose_results[0]['keypoints'])[:, :2]

    # use them to build character skeleton rig
    skeleton = []
    skeleton.append({'loc' : [round(x) for x in (kpts[11]+kpts[12])/2], 'name': 'root'          , 'parent': None})
    skeleton.append({'loc' : [round(x) for x in (kpts[11]+kpts[12])/2], 'name': 'hip'           , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in (kpts[5]+kpts[6])/2  ], 'name': 'torso'         , 'parent': 'hip'})
    skeleton.append({'loc' : [round(x) for x in  kpts[0]             ], 'name': 'neck'          , 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  kpts[6]             ], 'name': 'right_shoulder', 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  kpts[8]             ], 'name': 'right_elbow'   , 'parent': 'right_shoulder'})
    skeleton.append({'loc' : [round(x) for x in  kpts[10]            ], 'name': 'right_hand'    , 'parent': 'right_elbow'})
    skeleton.append({'loc' : [round(x) for x in  kpts[5]             ], 'name': 'left_shoulder' , 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  kpts[7]             ], 'name': 'left_elbow'    , 'parent': 'left_shoulder'})
    skeleton.append({'loc' : [round(x) for x in  kpts[9]             ], 'name': 'left_hand'     , 'parent': 'left_elbow'})
    skeleton.append({'loc' : [round(x) for x in  kpts[12]            ], 'name': 'right_hip'     , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in  kpts[14]            ], 'name': 'right_knee'    , 'parent': 'right_hip'})
    skeleton.append({'loc' : [round(x) for x in  kpts[16]            ], 'name': 'right_foot'    , 'parent': 'right_knee'})
    skeleton.append({'loc' : [round(x) for x in  kpts[11]            ], 'name': 'left_hip'      , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in  kpts[13]            ], 'name': 'left_knee'     , 'parent': 'left_hip'})
    skeleton.append({'loc' : [round(x) for x in  kpts[15]            ], 'name': 'left_foot'     , 'parent': 'left_knee'})

    # create the character config dictionary
    char_cfg = {'skeleton': skeleton, 'height': cropped.shape[0], 'width': cropped.shape[1]}

    # convert texture to RGBA and save
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(str(outdir/'texture.png'), cropped)

    # save mask
    cv2.imwrite(str(outdir/'mask.png'), mask)

    # dump character config to yaml
    with open(str(outdir/'char_cfg.yaml'), 'w') as f:
        yaml.dump(char_cfg, f)

    # create joint viz overlay for inspection purposes
    joint_overlay = cropped.copy()
    for joint in skeleton:
        x, y = joint['loc']
        name = joint['name']
        cv2.circle(joint_overlay, (int(x), int(y)), 5, (0, 0, 0), 5)
        cv2.putText(joint_overlay, name, (int(x), int(y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)
    cv2.imwrite(str(outdir/'joint_overlay.png'), joint_overlay)


def segment(img: np.ndarray):
    """ threshold """
    img = np.min(img, axis=2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
    img = cv2.bitwise_not(img)

    """ morphops """
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=2)

    """ floodfill """
    mask = np.zeros([img.shape[0]+2, img.shape[1]+2], np.uint8)
    mask[1:-1, 1:-1] = img.copy()

    # im_floodfill is results of floodfill. Starts off all white
    im_floodfill = np.full(img.shape, 255, np.uint8)

    # choose 10 points along each image side. use as seed for floodfill.
    h, w = img.shape[:2]
    for x in range(0, w-1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h-1), 0)
    for y in range(0, h-1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w-1, y), 0)

    # make sure edges aren't character. necessary for contour finding
    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    im_floodfill = cv2.morphologyEx(im_floodfill, cv2.MORPH_ERODE, kernel, iterations=1)

    """ retain largest contour """
    mask2 = cv2.bitwise_not(im_floodfill)
    mask = None
    biggest = 0

    contours = measure.find_contours(mask2, 0.0)
    for c in contours:
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        if size > biggest:
            mask = x
            biggest = size
    if mask is None:
        msg = 'Found no contours within image'
        logging.critical(msg)
        assert False, msg

    mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = 255 * mask.astype(np.uint8)

    kernel = np.array([[0, 1, 0],
                    [1, 1, 0],
                    [0, 1, 0]], np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)

    return mask.T

if __name__ == '__main__':
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

    img_fn = sys.argv[1]
    out_dir = sys.argv[2]
    image_to_annotations(img_fn, out_dir)
