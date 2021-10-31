#!/bin/bash python3
import os
from os.path import join
import argparse
import shutil
import json

import numpy as np
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def compute_pose(init_position, init_orientation_quat, kitti_pose):
    init_4x4 = np.eye(4)
    init_orientation_quat_xyzw = np.array([init_orientation_quat[1],
                                           init_orientation_quat[2],
                                           init_orientation_quat[3],
                                           init_orientation_quat[0]])
    init_rotation = R.from_quat(init_orientation_quat_xyzw)
    init_4x4[:3, :3] = init_rotation.as_matrix()
    init_4x4[:3, 3] = init_position
    transformation_4x4 = np.eye(4)
    transformation_4x4[:3, :] = kitti_pose.reshape(3,4)
    new_pose_4x4 = init_4x4 @ transformation_4x4
    position = new_pose_4x4[:3,3]
    orientation_xyzw = R.from_matrix(new_pose_4x4[:3, :3]).as_quat()
    orientation = [orientation_xyzw[3],
                   orientation_xyzw[0],
                   orientation_xyzw[1],
                   orientation_xyzw[2]]
    return {'position': list(position), 'orientation': list(orientation)}


def main(args):
    print("\n===> Loading data...")
    subset = args.subset_name
    orig_imgs_dir = args.images_dir
    orig_depths_dir = args.depths_dir
    imgs_list = sorted(os.listdir(orig_imgs_dir))
    depths_list = sorted(os.listdir(orig_depths_dir))
    poses_list = [line[:-1] for line in open(args.poses_txt, 'r').readlines()]
    assert len(imgs_list) == len(depths_list) == len(poses_list)
    initial_pose_data = yaml.load(open(args.initial_pose), Loader=yaml.FullLoader)['atlans_initial_pose']
    extraction_frequency = args.extraction_frequency
    start_idx = args.start_frame
    output_dir = args.output_dir

    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    print("\n===> Creating output directories...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, 'images'), exist_ok=True)
    os.makedirs(join(output_dir, 'depths'), exist_ok=True)
    
    init_position = np.array([initial_pose_data['position']['x'],
                              initial_pose_data['position']['y'],
                              initial_pose_data['position']['z']])
    init_orientation_quat = np.array([initial_pose_data['orientation']['w'],
                                      initial_pose_data['orientation']['x'],
                                      initial_pose_data['orientation']['y'],
                                      initial_pose_data['orientation']['z']])
    poses_json_file_path = join(output_dir, 'poses.json')
    poses = {}
    if not os.path.exists(poses_json_file_path):
        with open(poses_json_file_path, 'w'): pass
    else:
        with open(poses_json_file_path) as poses_json_file:
            poses = json.load(poses_json_file)

    print("\n===> Extracting and preprocessing data...")
    output_dir_last = output_dir.split('/')[-1]
    imagenames_file_path = join(output_dir, f"{output_dir_last}_imagenames.txt")
    imagenames_full_path = join(output_dir, f"{output_dir_last}_imagenames_full.txt")

    for idx in tqdm(range(start_idx, len(imgs_list), extraction_frequency)):
        img_name = subset + imgs_list[idx]
        name = img_name.split('.')[0]
        img_relative_path = join('images', img_name)
        with open(imagenames_file_path, 'a') as imagenames_file, open(imagenames_full_path, 'a') as imagenames_full:
            imagenames_file.write(img_relative_path + '\n')
            imagenames_full.write(output_dir_last + '/' + img_relative_path + '\n')
        
        depth_name = name + '.' + depths_list[idx].split('.')[-1]
        depth_path = join(output_dir, 'depths', depth_name)
        
        shutil.copy(join(orig_imgs_dir, imgs_list[idx]), join(output_dir, img_relative_path))
        shutil.copy(join(orig_depths_dir, depths_list[idx]), depth_path)
        
        kitti_pose = np.array(list(map(float, poses_list[idx].split(' '))))
        poses[name] = compute_pose(init_position, init_orientation_quat, kitti_pose)
        
    with open(poses_json_file_path, 'w') as poses_json_file:
        json.dump(poses, poses_json_file)

    print("===> Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data extracted from bagfile for future use in PNTR pipeline")
    parser.add_argument("--subset_name", type=str, default="husky",
                        help="The name of the subset in case there wil be a mix of multiple bagfiles in dataset")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to folder with images")
    parser.add_argument("--depths_dir", type=str, required=True,
                        help="Path to folder with depth images")
    parser.add_argument("--poses_txt", type=str, required=True,
                        help="Path to poses.txt file with poses in KITTI format relative to initial pose")
    parser.add_argument("--initial_pose", type=str, required=True,
                        help="Path to meta.yaml file with initial absolute pose in format (x, y, z) and quaternion")
    parser.add_argument("--extraction_frequency", type=int, default=10,
                        help="Extract every N-th frame (default: 10)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Number of starting frame (default: 0)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output folder (will be created if not exist)")

    args = parser.parse_args()

    main(args)
