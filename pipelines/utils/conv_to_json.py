from tqdm import tqdm
import h5py
from os.path import join
import os
import numpy as np
from pathlib import Path
import json
import numpy as np
import cv2

MAXDEPTH = 10

def conv_to_json(dataset_root, path_to_npz_folder, output_dir):
    pairs_npz = os.listdir(path_to_npz_folder) 
    os.makedirs(output_dir, exist_ok = True)
    for pair_npz in tqdm(pairs_npz):
        if not pair_npz.endswith('.npz'):
            continue
        npz = np.load(join(path_to_npz_folder, pair_npz))
        q_folder = pair_npz.split('_')[0]
        q_name = pair_npz.split('_')[2]
        db_folder = pair_npz.split('_')[3]
        db_name = pair_npz.split('_')[5] 

        q_depth_file_path = join(dataset_root, q_folder, 'depths', q_name+'.exr')
        db_depth_file_path = join(dataset_root, db_folder, 'depths', db_name+'.exr')

        print("\n\nDEBUG\n", q_depth_file_path, db_depth_file_path)
        
        q_depth = cv2.imread(q_depth_file_path, cv2.IMREAD_UNCHANGED)
        db_depth = cv2.imread(db_depth_file_path, cv2.IMREAD_UNCHANGED)

        # q_depth = np.squeeze(depth[int(q_name)])  # *MAXDEPTH  # what?
        # m_depth = np.squeeze(depth_base[int(m_name)])  # *MAXDEPTH

        q_coord_frame = []
        db_coord_frame = []

        for kpt in range(min(npz['keypoints1'].shape[0], npz['matches'].shape[0])): 
            if npz['matches'][kpt] != -1:
                x_q, y_q = map(int, npz['keypoints0'][kpt])
                x_db, y_db = map(int, npz['keypoints1'][npz['matches'][kpt]])
                
                depth_q = float(q_depth[y_q, x_q])
                if np.isnan(depth_q) or np.isinf(depth_q):
                    depth_q = 0

                depth_db = float(db_depth[y_db, x_db])
                if np.isnan(depth_db) or np.isinf(depth_db):
                    depth_db = 0

                q_coord_frame.append((x_q, y_q, depth_q))
                db_coord_frame.append((x_db, y_db, depth_db))
    
        dictionary_kpt = {q_name: q_coord_frame, db_name: db_coord_frame}
        outpath = join(output_dir, q_name + '_' + db_name + '.json')
        with open(outpath, 'w') as outfile:
            json.dump(str(dictionary_kpt), outfile)
