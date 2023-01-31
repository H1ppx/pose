import argparse
import math

import cv2
import numpy as np
import torch
from alive_progress import alive_bar

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose

def infer_fast(net, img, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = 256 / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = np.array(scaled_img, dtype=np.float32)
    scaled_img = (scaled_img - img_mean) * img_scale

    min_dims = [256, max(scaled_img.shape[1], 256)]

    h, w, _ = scaled_img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(scaled_img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def run(net, img, cpu=True):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, stride, upsample_ratio, cpu)
    total_keypoints_num = 0
    all_keypoints_by_type = []
    
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
    
    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
    
    for pose in current_poses:
        pose.draw(img)
    
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                      (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=''' Pose Est Video Test''')
    parser.add_argument('--video-path', required=True, help='path to input image')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    load_state(net, checkpoint)

    cap = cv2.VideoCapture(args.video_path)
    output = cv2.VideoWriter('test.mp4',
                cv2.VideoWriter_fourcc('F','M','P','4'),
                int(cap.get(5)),(int(cap.get(3)),int(cap.get(4))))

    with alive_bar(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as bar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            output.write(run(net, frame))
            bar()
            
cap.release()
output.release()
cv2.destroyAllWindows()
