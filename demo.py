import argparse
import math

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses

def infer_fast(net, img, stride, upsample_ratio, device,
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
    
    if str(device).isdigit():
        tensor_img = tensor_img.cuda() # Set tensor to GPU
    elif device == 'mps':
        tensor_img = tensor_img.to(torch.device('mps')) # Set tensor to Metal Performance Shaders

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=''' Pose Est Test''')
    parser.add_argument('--source', default=0, required=True, help='Video source, or webcam index.')
    parser.add_argument('--device', default='cpu', help='Device to use for inference. cpu, cuda(device number), mps')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    load_state(net, checkpoint)

    cap = cv2.VideoCapture(args.source)

    net = net.eval() # Set model to evaluation mode
    if str(args.device).isdigit():
        net = net.cuda() # Set model to GPU
    elif args.device == 'mps':
        net = net.to(torch.device('mps')) # Set model to Metal Performance Shaders

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == False:
            break

        img = frame.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, stride, upsample_ratio, args.device)
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

        track_poses(previous_poses, current_poses, smooth=False)
        
        # if there are no previous poses, set the current poses as the previous poses
        if not len(previous_poses):
                previous_poses = current_poses  

        # compare current poses with previous poses, 
        # if id is the same, update the previous pose with the current pose
        # else add the current pose to the previous poses
        for current_pose in current_poses: 
            loopBroken = False 
            for previous_pose in previous_poses:
                if current_pose.id == previous_pose.id:
                    previous_pose.keypoints = current_pose.keypoints
                    previous_pose.bbox = current_pose.bbox
                    loopBroken = True
                    break
            if not loopBroken:
                previous_poses.append(current_pose)
            

        for pose in current_poses:
            print(pose.keypoints)
            pose.draw(img)
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

            # Display the resulting frame
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()