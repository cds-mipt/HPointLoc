import teaserpp_python
import os
import h5py
from utils.functions import quaternion_to_rotation_matrix, clouds3d_from_kpt,\
                            is_invertible, compute_errors, print_results
import numpy as np
from scipy.spatial.transform import Rotation as R
import re
from pathlib import Path
from os.path import join
import json

os.environ["OMP_NUM_THREADS"] = "12"

def teaser(dataset_root, query, path_image_retrieval, path_loc_features_matches, output_dir, topk=1):
    NOISE_BOUND = 0.5 # 0.05
    N_OUTLIERS = 1700
    N_INLIERS = 400
    OMP_NUM_THREADS=12
    OUTLIER_TRANSLATION_LB = 5
    OUTLIER_TRANSLATION_UB = 10
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 0.5  # 0.05
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

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

    netvlad_results = {
        "(5m, 20°)": 0,
        "(1m, 10°)": 0,
        "(0.5m, 5°)": 0,
        "(0.25m, 2°)": 0,
        "(5m)": 0,
        "(1m)": 0,
        "(0.5m)": 0,
        "(0.25m)": 0
    }

    teaser_numbers = 0
    query_numbers = 0
    
    os.makedirs(output_dir, exist_ok = True)
    path_result_poses = join(output_dir, 'PNTR_teaser.json')
    path_transformations = join(output_dir, 'transformations_teaser.json')
    path_result_kitti_poses = join(output_dir, 'result_kitti_teaser.txt')
    path_gt_kitti_poses = join(output_dir, 'gt_kitti.txt')
    
    q_poses_file_path = join(dataset_root, f'{query}/poses.json')
    db_poses_file_path = join(dataset_root, 'database/poses.json')
    
    q_poses = {}
    with open(q_poses_file_path) as f:
        q_poses = json.load(f)

    db_poses = {}
    with open(db_poses_file_path) as f:
        db_poses = json.load(f)

    final_res = {}
    estimated_kitti = []
    gt_kitti = []
    transformations_4x4 = {}

    netvlad_dist_errors = []
    netvlad_angle_errors = []
    optimizer_dist_errors = []
    optimizer_angle_errors = []

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
            fullpath = os.path.join(path_loc_features_matches, pairpath)

            points_3d_query, points_3d_db = clouds3d_from_kpt(fullpath)

            assert points_3d_query.shape == points_3d_db.shape 

            transformation_4x4 = np.eye(4)

            if points_3d_db.shape[0] > 0 and points_3d_db.shape[1] > 1:  
                solver = teaserpp_python.RobustRegistrationSolver(solver_params)
                solver.solve(points_3d_db, points_3d_query)
                rotation = solver.getSolution().rotation
                translation = solver.getSolution().translation
                # scale = solver.getSolution().scale

                db_4x4 = np.eye(4)
                db_4x4[:3, :3] = estimated_orientation_r.as_matrix()
                db_4x4[:3, 3] = db_position

                transformation_4x4[:3,:3] = rotation
                transformation_4x4[:3,3] = translation
                
                if is_invertible(transformation_4x4):
                    predict_4x4 = db_4x4 @ np.linalg.inv(transformation_4x4)
                    predict_quat_xyzw = R.from_matrix(predict_4x4[:3,:3]).as_quat()
                    predict_quat = [predict_quat_xyzw[3], predict_quat_xyzw[0], 
                                    predict_quat_xyzw[1], predict_quat_xyzw[2]]
                    predict_position = predict_4x4[:3,3]

                    if not np.isnan(predict_quat).any():
                        teaser_numbers += 1
                        estimated_position = predict_position
                        estimated_orientation_quat = predict_quat
                        estimated_orientation_quat_xyzw = predict_quat_xyzw
                        estimated_orientation_r = R.from_quat(estimated_orientation_quat_xyzw)

            transformations_4x4[q_name] = transformation_4x4.tolist()
            
            pose_estimated = np.eye(4)
            pose_gt = np.eye(4)

            pose_estimated[:3, :3] = estimated_orientation_r.as_matrix()
            pose_estimated[:3, 3] = estimated_position

            pose_gt[:3, :3] = gt_q_orientation_r.as_matrix()
            pose_gt[:3, 3] = gt_q_position

            optimizer_dist_error, optimizer_angle_error = compute_errors(pose_estimated, pose_gt)
            netvlad_dist_error, netvlad_angle_error = compute_errors(db_4x4, pose_gt)
            
            optimizer_dist_errors.append(optimizer_dist_error)
            optimizer_angle_errors.append(optimizer_angle_error)
            netvlad_dist_errors.append(netvlad_dist_error)
            netvlad_angle_errors.append(netvlad_angle_error)

            if  optimizer_dist_error < 0.25:
                results["(0.25m)"] += 1
                if optimizer_angle_error < 2:
                    results["(0.25m, 2°)"] += 1
            if  optimizer_dist_error < 0.5:
                results["(0.5m)"] += 1
                if optimizer_angle_error < 5:
                    results["(0.5m, 5°)"] += 1
            if  optimizer_dist_error < 1:
                results["(1m)"] += 1
                if optimizer_angle_error < 10:
                    results["(1m, 10°)"] += 1
            if  optimizer_dist_error < 5:
                results["(5m)"] += 1    
                if optimizer_angle_error < 20:
                    results["(5m, 20°)"] += 1

            if  netvlad_dist_error < 0.25:
                netvlad_results["(0.25m)"] += 1
                if netvlad_angle_error < 2:
                    netvlad_results["(0.25m, 2°)"] += 1
            if  netvlad_dist_error < 0.5:
                netvlad_results["(0.5m)"] += 1
                if netvlad_angle_error < 5:
                    netvlad_results["(0.5m, 5°)"] += 1
            if  netvlad_dist_error < 1:
                netvlad_results["(1m)"] += 1
                if netvlad_angle_error < 10:
                    netvlad_results["(1m, 10°)"] += 1
            if  netvlad_dist_error < 5:
                netvlad_results["(5m)"] += 1    
                if netvlad_angle_error < 20:
                    netvlad_results["(5m, 20°)"] += 1

            final_res[q_name] = {'db_match': db_name,
                                 'optimizer_pose_kitti': list(pose_estimated[:3, :].flatten()),
                                 'optimizer_pose': {'position': list(estimated_position),
                                                    'orientation': list(estimated_orientation_quat)},
                                 'netvlad_pose': {'position': list(db_position),
                                                  'orientation': list(db_orientation_quat)},
                                 'optimizer_dist_error': optimizer_dist_error,
                                 'optimizer_angle_error': optimizer_angle_error,
                                 'netvlad_dist_error': netvlad_dist_error,
                                 'netvlad_angle_error': netvlad_angle_error,
                                }

            estimated_pose_kitty = ' '.join(map(str, list(pose_estimated[:3, :].flatten()))) + '\n'
            gt_pose_kitty = ' '.join(map(str, list(pose_gt[:3, :].flatten()))) + '\n'
            estimated_kitti.append(estimated_pose_kitty)
            gt_kitti.append(gt_pose_kitty)
    
    with open(path_result_poses, 'w') as result_poses_file:
        json.dump(final_res, result_poses_file)

    with open(path_transformations, 'w') as transformations_file:
        json.dump(transformations_4x4, transformations_file)

    with open(path_result_kitti_poses, 'w') as result_kitti_file,\
         open(path_gt_kitti_poses, 'w') as gt_kitti_file:
        result_kitti_file.write(''.join(estimated_kitti))
        gt_kitti_file.write(''.join(gt_kitti))
            
    for key in results.keys():
        results[key] = results[key] / query_numbers

    for key in netvlad_results.keys():
        netvlad_results[key] = netvlad_results[key] / query_numbers

    print_results(netvlad_results, results, optimizer_type="TEASER++")
    print('Mean dist error:')
    print(f'\twithout optimization: {np.mean(netvlad_dist_errors)}')
    print(f'\tafter optimization: {np.mean(optimizer_dist_errors)}')
    print(f'Mean angle error:')
    print(f'\twithout optimization: {np.mean(netvlad_angle_errors)}')
    print(f'\tafter optimization: {np.mean(optimizer_angle_errors)}\n>>>>')
    print('Proportion of optimized:', teaser_numbers / query_numbers, '\n>>>>\n')
