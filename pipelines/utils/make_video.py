#!/bin/bash python3
import argparse
import os
from os import path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_plot_image(kitti_positions, idx):
    kitti_positions = np.array(kitti_positions)[:,:2]
    x_max = np.max(kitti_positions[:,0]) * 1.001
    y_max = np.max(kitti_positions[:,1]) * 1.001
    kitti_positions[:,0] = x_max - kitti_positions[:,0]
    kitti_positions[:,1] = y_max - kitti_positions[:,1]

    fig = plt.figure(figsize=(1200/96, 1200/96), dpi=96)
    ax = fig.add_subplot(1,1,1)
    ax.scatter(kitti_positions[:idx+1, 0], kitti_positions[:idx+1, 1])
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape((1200, 1200, 3))

    return img


def main(opt):
    print("\n>>>>>\n\tResults visualization tool\n>>>>>\n")

    base_dir = opt.result_dir
    if base_dir[-1] == '/':
        base_dir = base_dir[:-1]
    if not path.exists(base_dir):
        raise FileNotFoundError(base_dir, " directory does not exist!")

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

    patchnetvlad_pairs_filename = 'patchnetvlad_' + opt.netvlad_config + 'top1.txt'
    patchnetvlad_pairs_filename = path.join(image_retrieval_dir, patchnetvlad_pairs_filename)
    if not path.exists(patchnetvlad_pairs_filename):
        raise FileNotFoundError(patchnetvlad_pairs_filename, " file does not exist!")

    pose_result_kitti_filename = path.join(optimizer_dir, 'result_kitti.txt')
    if not path.exists(pose_result_kitti_filename):
        raise FileNotFoundError(pose_result_kitti_filename, " file does not exist!")

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

    kitti_positions = []
    with open(pose_result_kitti_filename, 'r') as result_kitti:
        for line in result_kitti.readlines():
            pose = list(map(float, line[:-1].split(' ')))
            kitti_positions.append([pose[3], pose[7], pose[11]])
    if len(kitti_positions) != len(dump_match_pairs_imgs):
        raise RuntimeError("Lengths of poses list and pairs images list does not match!")

    print("\tFound all files...")
    print("\n===> Creating video...")

    output_dir = opt.output_dir
    if path.isfile(output_dir):
        raise FileExistsError("You must specify output directory path, not file!")
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in range(len(kitti_positions)):
        superglue_match_pair_img = cv2.imread(dump_match_pairs_dir+dump_match_pairs_imgs[idx])
        plot_img = make_plot_image(kitti_positions, idx)

        output_frame = np.zeros((1800, 2410, 3))
        output_frame[:600, :, :] = superglue_match_pair_img
        output_frame[600:, 605:1805, :] = plot_img

        cv2.imwrite(path.join(output_dir, str(idx)+'.png'), output_frame)

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
    parser.add_argument('--dump_match_pairs_dir', type=str,
                        default='./3rd/SuperGluePretrainedNetwork/dump_match_pairs',
                        help="Path to directory with dumped matched SuperGlue pairs"+\
                             " (default: ./3rd/SuperGluePretrainedNetwork/")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Full path for output directory (example: ./result/query_00)")

    args = parser.parse_args()

    main(args)
