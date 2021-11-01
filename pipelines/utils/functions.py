import json
from scipy.spatial.transform import Rotation as R
import ast
import math
import numpy as np


MAX_DEPTH = np.inf

CAMERA_INTRINSICS = {'query_00': [859.7086033959424, 859.7086033959424, 
                                  559.0216447990299, 309.07840227628515],
                     'query_01': [867.1920188938037, 867.1920188938037, 
                                  554.9548231998494, 308.58383552157835],
                     'query_17': [830.93694071386, 830.93694071386, 
                                  572.169579679035, 309.80029849197297],
                     'database': [830.93694071386, 830.93694071386, 
                                  572.169579679035, 309.80029849197297]
                    }


def camera_center_to_translation(c, qvec):
    R = quaternion_to_rotation_matrix(qvec)
    return (-1) * np.matmul(R, c)


def quaternion_to_rotation_matrix(quaternion_wxyz):
    r = R.from_quat([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    matrix = r.as_matrix()
    matrix[:3,2] = -matrix[:3,2]
    matrix[:3,1] = -matrix[:3,1]
    return matrix


def get_point_3d(x, y, depth, fx, fy, cx, cy, cam_center_world, R_world_to_cam, w_in_quat_first = True):
    if depth <= 0:
        return 0
    new_x = (x - cx)*depth/fx
    new_y = (y - cy)*depth/fy
    new_z = depth
    coord_3D_world_to_cam = np.array([new_x, new_y, new_z], float)
    if len(R_world_to_cam) == 4:
        if w_in_quat_first:
            matrix = quaternion_to_rotation_matrix(R_world_to_cam)
        else:
            R_world_to_cam = [R_world_to_cam[3], R_world_to_cam[0], R_world_to_cam[1], R_world_to_cam[2]]
            matrix = quaternion_to_rotation_matrix(R_world_to_cam)
    coord_3D_cam_to_world = np.matmul(matrix, coord_3D_world_to_cam) + cam_center_world
    return coord_3D_cam_to_world


def clouds3d_from_kpt(path):
    query_name = ''
    query_num = ''
    if 'query_00' in path:
        query_name = 'query_00'
        query_num = '00'
    elif 'query_01' in path:
        query_name = 'query_01'
        query_num = '01'
    elif 'query_17' in path:
        query_name = 'query_17'
        query_num = '17'

    q_fx, q_fy, q_cx, q_cy = CAMERA_INTRINSICS[query_name]
    db_fx, db_fy, db_cx, db_cy = CAMERA_INTRINSICS['database']

    filedict = open(path, 'r')
    dict_string = json.load(filedict)
    ast.literal_eval(dict_string)  
    kpt_dict = ast.literal_eval(dict_string)

    # находим точные ключи в словаре для query и database
    dict_keys = list(kpt_dict.keys())
    if dict_keys[0].startswith(query_num):
        query_key = dict_keys[0]
        db_key = dict_keys[1]
    else:
        query_key = dict_keys[1]
        db_key = dict_keys[0]

    query_points = kpt_dict[query_key]
    db_points = kpt_dict[db_key]

    assert len(query_points) == len(db_points)  # в json словаре должны быть попарные матчи точек

    query_3d_points = []
    db_3d_points = []
    for query_point, db_point in zip(query_points, db_points):
        q_x = query_point[0]
        q_y = query_point[1]
        q_depth = query_point[2]

        db_x = db_point[0]
        db_y = db_point[1]
        db_depth = db_point[2]

        if q_depth <= 0 or db_depth <= 0 or \
           q_depth > MAX_DEPTH or db_depth > MAX_DEPTH:
            continue

        query_3d_point = cloud_3d_cam(q_x, q_y, q_depth,
                                      q_fx, q_fy, q_cx, q_cy)
        db_3d_point = cloud_3d_cam(db_x, db_y, db_depth,
                                   db_fx, db_fy, db_cx, db_cy)

        query_3d_points.append(query_3d_point)
        db_3d_points.append(db_3d_point)

    query_3d_points = np.array(query_3d_points).T
    db_3d_points = np.array(db_3d_points).T
    
    return query_3d_points, db_3d_points


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));


def compute_errors(pose_estimated, pose_gt):
    error_pose = np.linalg.inv(pose_estimated) @ pose_gt
    dist_error = np.sum(error_pose[:3, 3]**2) ** 0.5
    r = R.from_matrix(error_pose[:3, :3])
    rotvec = r.as_rotvec()
    angle_error = (np.sum(rotvec**2)**0.5) * 180 / 3.14159265353
    angle_error = abs(90 - abs(angle_error-90))
    return dist_error, angle_error


def cloud_3d_cam(x, y, depth, 
                 fx = CAMERA_INTRINSICS['database'][0],
                 fy = CAMERA_INTRINSICS['database'][1],
                 cx = CAMERA_INTRINSICS['database'][2],
                 cy = CAMERA_INTRINSICS['database'][3]):
    if depth <= 0:
        return 0
    new_x = depth  # roll
    new_y = - (x - cx)*depth/fx  # pitch
    new_z = - (y - cy)*depth/fy  # yaw
    coord_3D_world_to_cam = np.array([new_x, new_y, new_z], float)
    return coord_3D_world_to_cam


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def print_results(netvlad_results, optimizer_results, optimizer_type = ""):
    print("\n\n>>>> Metrics without optimization:")
    print("\t(5m, 20°): {:.4f},\t(1m, 10°): {:.4f},".format(netvlad_results['(5m, 20°)'],
                                                            netvlad_results['(1m, 10°)']),
          "\t(0.5m, 5°): {:.4f},\t(0.25m, 2°): {:.4f}".format(netvlad_results['(0.5m, 5°)'],
                                                              netvlad_results['(0.25m, 2°)']))
    print("\t(5m): {:.4f},\t\t(1m): {:.4f},".format(netvlad_results['(5m)'],
                                                    netvlad_results['(1m)']),
          "\t\t(0.5m): {:.4f},\t\t(0.25m): {:.4f}".format(netvlad_results['(0.5m)'],
                                                          netvlad_results['(0.25m)']))
    
    print(f"\n>>>> Metrics after optimization ({optimizer_type}):")
    print("\t(5m, 20°): {:.4f},\t(1m, 10°): {:.4f},".format(optimizer_results['(5m, 20°)'],
                                                            optimizer_results['(1m, 10°)']),
          "\t(0.5m, 5°): {:.4f},\t(0.25m, 2°): {:.4f}".format(optimizer_results['(0.5m, 5°)'],
                                                              optimizer_results['(0.25m, 2°)']))
    print("\t(5m): {:.4f},\t\t(1m): {:.4f},".format(optimizer_results['(5m)'],
                                                    optimizer_results['(1m)']),
          "\t\t(0.5m): {:.4f},\t\t(0.25m): {:.4f}".format(optimizer_results['(0.5m)'],
                                                          optimizer_results['(0.25m)']))
    print('\n>>>>')
