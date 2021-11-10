#!/bin/bash python3
import argparse
import os
from os import path
import json

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_plot_image(netvlad_positions, optimizer_positions, gt_positions, idx):
    netvlad_positions = np.array(netvlad_positions)[:,:2]
    optimizer_positions = np.array(optimizer_positions)[:,:2]
    gt_positions = np.array(gt_positions)[:, :2]
    x_max = np.max(gt_positions[:,0]) * 1.00001
    y_max = np.max(gt_positions[:,1]) * 1.00001
    x_min = np.min(gt_positions[:,0]) * 0.99999
    y_min = np.min(gt_positions[:,1]) * 0.99999

    fig = plt.figure(figsize=(1200/96, 1200/96), dpi=96)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(left=x_min, right=x_max)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.scatter(gt_positions[:idx+1, 0], gt_positions[:idx+1, 1], 
               color='#10d010', label="Ground truth", marker='o')
    ax.scatter(netvlad_positions[:idx+1, 0], netvlad_positions[:idx+1, 1],
               color='#ffa0a0', label="Patch-NetVLAD position", marker='+')
    ax.scatter(optimizer_positions[:idx+1, 0], optimizer_positions[:idx+1, 1],
               color='#ef0000', label="Optimized position", marker='x')
    ax.legend()
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape((1200, 1200, 3))

    plt.close()

    return img


def main(opt):
    print("\n>>>>>\n\tResults visualization tool\n>>>>>\n")

    base_dir = opt.result_dir
    if base_dir[-1] == '/':
        base_dir = base_dir[:-1]
    if not path.exists(base_dir):
        raise FileNotFoundError(base_dir, " directory does not exist!")

    optimizer = opt.optimizer
    if optimizer == 'teaser':
        optimizer_name = 'TEASER++'
    elif optimizer == 'icp':
        optimizer_name = 'ICP'
    else:
        optimizer_name = 'unknown optimizer'

    print("\n===> Looking for files...")

    image_retrieval_dir = path.join(base_dir, 'image_retrieval')
    keypoints_dir = path.join(base_dir, 'keypoints')
    optimizer_dir = path.join(base_dir, 'pose_optimization')

    if not path.exists(image_retrieval_dir):
        raise FileNotFoundError(image_retrieval_dir, " directory not found!")
    if not path.exists(keypoints_dir):
        raise FileNotFoundError(keypoints_dir, " directory not found!")
    if not path.exists(optimizer_dir):
        raise FileNotFoundError(optimizer_dir, " directory not found!")

    patchnetvlad_pairs_filename = 'patchnetvlad_' + opt.netvlad_config + '_top1.txt'
    patchnetvlad_pairs_filename = path.join(image_retrieval_dir, patchnetvlad_pairs_filename)
    if not path.exists(patchnetvlad_pairs_filename):
        raise FileNotFoundError(patchnetvlad_pairs_filename, " file does not exist!")

    pntr_result_filename = path.join(optimizer_dir, f'PNTR_{optimizer}.json')
    if not path.exists(pntr_result_filename):
        raise FileNotFoundError(pntr_result_filename, " file does not exist!")
    pose_gt_kitti_filename = path.join(optimizer_dir, 'gt_kitti.txt')
    if not path.exists(pose_gt_kitti_filename):
        raise FileNotFoundError(pose_gt_kitti_filename, " file does not exist!")

    dump_match_pairs_dir = opt.dump_match_pairs_dir
    if not path.exists(dump_match_pairs_dir):
        raise FileNotFoundError(dump_match_pairs_dir, " directory does not exist!")

    query_name = base_dir.split('/')[-1]
    dump_match_pairs_listdir = os.listdir(dump_match_pairs_dir)
    dump_match_pairs_imgs = []
    for filename in dump_match_pairs_listdir:
        if filename.endswith('.png') and filename.startswith(query_name):
            dump_match_pairs_imgs.append(filename)
    dump_match_pairs_imgs = sorted(dump_match_pairs_imgs)
    if len(dump_match_pairs_imgs) == 0:
        raise FileNotFoundError(dump_match_pairs_dir,
                                " - no images for given results in directory!")

    result_dict = {}
    result_netvlad_positions = []
    result_optimizer_positions = []
    gt_positions = []
    with open(pose_gt_kitti_filename, 'r') as gt_kitti:
        for line in gt_kitti.readlines():
            pose = list(map(float, line[:-1].split(' ')))
            gt_positions.append([pose[3], pose[7], pose[11]])
    with open(pntr_result_filename) as result_f:
        result_dict = json.load(result_f)
        for result in result_dict.values():
            netvlad_position = result['netvlad_pose']['position']
            result_netvlad_positions.append(netvlad_position)
            optimizer_position = result['optimizer_pose']['position']
            result_optimizer_positions.append(optimizer_position)

    if len(result_dict) != len(dump_match_pairs_imgs):
        raise RuntimeError("Lengths of poses list and pairs images list does not match!")
    if len(result_dict) != len(gt_positions):
        raise RuntimeError("Lengths of estimated and gt poses does not match!")

    print("\tFound all files...")
    print("\n===> Creating video...")

    output_dir = opt.output_dir
    if path.isfile(output_dir):
        raise FileExistsError("You must specify output directory path, not file!")
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    video = cv2.VideoWriter(path.join(output_dir, f'video_{optimizer}.avi'),
                            fourcc, 2, (1446, 1080))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    blackColor = (0,0,0)
    greenColor = (0,127,0)
    redColor = (0,0,127)
    lineType = 3

    for idx, (query_imagename, result) in tqdm(enumerate(result_dict.items())):
        superglue_match_pair_img = cv2.imread(path.join(dump_match_pairs_dir, 
                                                        dump_match_pairs_imgs[idx]))
        plot_img = make_plot_image(result_netvlad_positions, result_optimizer_positions,
                                   gt_positions, idx)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

        output_frame = np.full((1800, 2410, 3), 255)
        output_frame[:600, :, :] = superglue_match_pair_img
        output_frame[600:, 1210:, :] = plot_img
        output_frame = np.array(output_frame, dtype=np.uint8)

        query_imagename_text = f"Query image: {query_imagename}"
        matched_imagename_text = f"Matched image: {result['db_match']}"

        cv2.putText(output_frame, query_imagename_text, 
                    (50, 700), font, fontScale, blackColor, lineType)
        cv2.putText(output_frame, matched_imagename_text, 
                    (50, 800), font, fontScale, blackColor, lineType)

        netvlad_dist_error = result['netvlad_dist_error']
        optimizer_dist_error = result['optimizer_dist_error']

        netvlad_angle_error = result['netvlad_angle_error']
        optimizer_angle_error = result['optimizer_angle_error']
        
        if optimizer_dist_error < netvlad_dist_error:
            netvlad_dist_error_text_color = redColor
            optimizer_dist_error_text_color = greenColor
        else:
            netvlad_dist_error_text_color = greenColor
            optimizer_dist_error_text_color = redColor

        if optimizer_angle_error < netvlad_angle_error:
            netvlad_angle_error_text_color = redColor
            optimizer_angle_error_text_color = greenColor
        else:
            netvlad_angle_error_text_color = greenColor
            optimizer_angle_error_text_color = redColor

        netvlad_error_text_title = "Patch-NetVLAD error:"
        netvlad_error_text_dist = f"distance = {netvlad_dist_error:.4f}"
        netvlad_error_text_angle = f"angle = {result_dict[query_imagename]['netvlad_angle_error']:.4f}"
        optimizer_error_text_title = f"Optimizer ({optimizer_name}) error:"
        optimizer_error_text_dist = f"distance = {optimizer_dist_error:.4f}"
        optimizer_error_text_angle = f"angle = {result_dict[query_imagename]['optimizer_angle_error']:.4f}"
            
        cv2.putText(output_frame, netvlad_error_text_title, 
                    (50,900), font, fontScale, blackColor, lineType)
        cv2.putText(output_frame, netvlad_error_text_dist, 
                    (70,1000), font, fontScale, netvlad_dist_error_text_color, lineType)
        cv2.putText(output_frame, netvlad_error_text_angle, 
                    (70,1100), font, fontScale, netvlad_angle_error_text_color, lineType)
        
        cv2.putText(output_frame, optimizer_error_text_title, 
                    (50,1300), font, fontScale, blackColor, lineType)
        cv2.putText(output_frame, optimizer_error_text_dist, 
                    (70,1400), font, fontScale, optimizer_dist_error_text_color, lineType)
        cv2.putText(output_frame, optimizer_error_text_angle, 
                    (70,1500), font, fontScale, optimizer_angle_error_text_color, lineType)

        output_frame_resized = cv2.resize(output_frame, (1446, 1080))
        output_frame_resized = np.array(output_frame_resized, dtype=np.uint8)

        video.write(output_frame_resized)

        # cv2.imwrite(path.join(output_dir, str(idx)+'.png'), output_frame)
    
    cv2.destroyAllWindows()
    video.release()

    # NOTE: dump_match_pairs_imgs.shape = (600, 2410, 3)
    # NOTE: plot shape will be 1200x1200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for making video visualization"+\
                                                 " of pipeline results")
    parser.add_argument('--result_dir', type=str, required=True,
                        help="Full path to result folder (example: ./result/query_00)")
    parser.add_argument('--netvlad_config', type=str, default='performance',
                        choices=['performance', 'speed', 'storage'],
                        help="Configuration for Patch-NetVLAD used for this results evaluation"+\
                             " (default: performance)")
    parser.add_argument('--optimizer', type=str, default='teaser',
                        choices=['teaser', 'icp'],
                        help="Optimizer which results will be visualized (default: teaser)")
    parser.add_argument('--dump_match_pairs_dir', type=str,
                        default='./3rd/SuperGluePretrainedNetwork/dump_match_pairs',
                        help="Path to directory with dumped matched SuperGlue pairs"+\
                             " (default: ./3rd/SuperGluePretrainedNetwork/")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Full path for output directory (example: ./result/query_00)")

    args = parser.parse_args()

    main(args)
