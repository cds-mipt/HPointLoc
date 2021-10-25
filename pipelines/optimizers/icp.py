from tqdm import tqdm
import h5py
from utils.functions import quaternion_to_rotation_matrix, clouds3d_from_kpt, is_invertible
from scipy.spatial.transform import Rotation as R
import re
from pathlib import Path
import json
import os
from os.path import join, exists, isfile
import open3d as o3d
import numpy as np

def icp(dataset_root, path_image_retrieval, path_loc_features_matches, output_dir, topk=1):
    trans_init = np.asarray([[1,0,0,0],   
                            [0,1,0,0],   
                            [0,0,1,0], 
                            [0,0,0,1]])
    results = {
        "(5m, 20°)": 0,
        "(1m, 10°)": 0,
        "(0.5m, 5°)": 0,
        "(0.25m, 2°)": 0,
        "(5m)": 0,
        "(1m)": 0,
        "(0.5m)": 0,
        "(0.25m)": 0
    }
    icp_numbers = 0
    query_numbers = 0
    threshold = 1.4
    os.makedirs(output_dir, exist_ok = True)
    path_result_poses = join(output_dir, 'PNTR.json')
    path_result_kitti_poses = join(output_dir, 'result_kitti.txt')
    
    q_poses_file_path = join(dataset_root, 'query/poses.json')
    db_poses_file_path = join(dataset_root, 'database/poses.json')
    
    q_poses = {}
    with open(q_poses_file_path) as f:
        q_poses = json.load(f)

    db_poses = {}
    with open(db_poses_file_path) as f:
        db_poses = json.load(f)

    final_res = {}
    estimated_kitti = []

    dist_errors = []
    angle_errors = []

    with open(path_image_retrieval, 'r') as f:
        for pair in f.readlines():
            query_numbers += 1
            q_img_file_path, db_img_file_path, score = pair.split(', ')

            q_name = q_img_file_path.split('/')[-1].split('.')[0]
            db_name = db_img_file_path.split('/')[-1].split('.')[0]
            
            q_pose = q_poses[q_name]
            db_pose = db_poses[db_name]

            gt_q_position = np.array(q_pose['position'])
            db_position = np.array(db_pose['position'])

            gt_q_orientation_quat = np.array(q_pose['orientation'])
            gt_q_orientation_quat_xyzw = [gt_q_orientation_quat[1], 
                                          gt_q_orientation_quat[2], 
                                          gt_q_orientation_quat[3], 
                                          gt_q_orientation_quat[0]]
            gt_q_orientation_r = R.from_quat(gt_q_orientation_quat_xyzw)
            
            db_orientation_quat = np.array(db_pose['orientation'])

            estimated_orientation_quat = db_orientation_quat
            estimated_orientation_quat_xyzw = [estimated_orientation_quat[1], 
                                               estimated_orientation_quat[2], 
                                               estimated_orientation_quat[3], 
                                               estimated_orientation_quat[0]]
            estimated_orientation_r = R.from_quat(estimated_orientation_quat_xyzw)
            estimated_position = db_position
        
            pairpath = q_name + '_' + db_name +'.json'
            fullpath = join(path_loc_features_matches, pairpath)

            points_3d_query, points_3d_db = clouds3d_from_kpt(fullpath)

            if points_3d_db.shape[1] > 1:
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(points_3d_query.transpose())  
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(points_3d_db.transpose())  

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())

                rotation = reg_p2p.transformation[:3,:3]
                translation = reg_p2p.transformation[:3,3]

                db_4x4 = np.eye(4)
                db_4x4[:3, :3] = estimated_orientation_r.as_matrix()
                db_4x4[:3, 3] = estimated_position

                transformation_4x4 = np.eye(4)
                transformation_4x4[:3,:3] = rotation
                transformation_4x4[:3,3] = translation

                predict_4x4 = db_4x4 @ np.linalg.inv(transformation_4x4)
                predict_quat_xyzw = R.from_matrix(predict_4x4[:3,:3]).as_quat()
                predict_quat = [predict_quat_xyzw[3], predict_quat_xyzw[0], 
                                predict_quat_xyzw[1], predict_quat_xyzw[2]]
                predict_position = predict_4x4[:3,3]

                if not np.isnan(predict_quat).any():
                    icp_numbers += 1
                    estimated_position = predict_position
                    estimated_orientation_quat = predict_quat
                    estimated_orientation_quat_xyzw = predict_quat_xyzw
                    estimated_orientation_r = R.from_quat(estimated_orientation_quat_xyzw)

            pose_estimated = np.eye(4)
            pose_gt = np.eye(4)

            pose_estimated[:3, :3] = estimated_orientation_r.as_matrix()
            pose_estimated[:3, 3] = estimated_position

            pose_gt[:3, :3] = gt_q_orientation_r.as_matrix()
            pose_gt[:3, 3] = gt_q_position

            error_pose = np.linalg.inv(pose_estimated) @ pose_gt

            dist_error = np.sum(error_pose[:3, 3]**2) ** 0.5

            r = R.from_matrix(error_pose[:3, :3])
            rotvec = r.as_rotvec()
            angle_error = (np.sum(rotvec**2)**0.5) * 180 / 3.14159265353
            angle_error = abs(90 - abs(angle_error-90))

            print(f"DEBUG: | {query_numbers} | dist_error = {dist_error} | angle_error = {angle_error}")

            dist_errors.append(dist_error)
            angle_errors.append(angle_error)

            if  dist_error < 0.25:
                results["(0.25m)"] += 1
            if  dist_error < 0.5:
                results["(0.5m)"] += 1
            if  dist_error < 1:
                results["(1m)"] += 1
            if  dist_error < 5:
                results["(5m)"] += 1    

            if angle_error < 2 and dist_error < 0.25:
                results["(0.25m, 2°)"] += 1
            if angle_error < 5 and dist_error < 0.5:
                results["(0.5m, 5°)"] += 1
            if angle_error < 10 and dist_error < 1:
                results["(1m, 10°)"] += 1
            if angle_error < 20 and dist_error < 5:
                results["(5m, 20°)"] += 1

    for key in results.keys():
        results[key] = results[key] / query_numbers

    print('\n>>>>\n', results, '\n>>>>')
    print(f'Mean dist error: {np.mean(dist_errors)}')
    print(f'Mean angle error: {np.mean(angle_errors)}\n>>>>')
    print('Proportion of optimized:', icp_numbers/query_numbers, '\n>>>>\n')

    # outpath = join(output_dir, q + '_' + m + '.json')    
    # with open(outpath, 'w') as outfile:
    # json.dump(str(dictionary_kpt), outfile)