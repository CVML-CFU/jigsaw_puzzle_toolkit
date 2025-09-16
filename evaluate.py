from evaluation import Evaluation
import os 
import numpy as np
import natsort 
import yaml 
import pandas as pd 
from datetime import datetime
import json

def main():

    #data_dir = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca"
    data_dir = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=m.khoroshiltseva/home/ssd/datasets/RePAIR_ReLab_marina/"
    test_set = np.loadtxt(os.path.join(data_dir, 'PAD_v2', 'test.txt'), dtype=str)
    # test_set = test_set[10:]
    experiments_folder = os.path.join(data_dir, 'experiments')
    preprocessing_folder = os.path.join(data_dir, 'preprocessing')
    Q_pos_list = []
    Q_pos_best_list = []
    xy_num_points_list = []
    theta_num_points_list = []
    no_rotations_list = []
    RMSE_r_list = []
    RMSE_t_list = []
    CM_type_list = []
    anchor_ids_list = []
    evaluated_test_set = []

    for test_puzzle in test_set:
        print(test_puzzle)
        test_puzzle_folder = os.path.join(experiments_folder, test_puzzle)

        if os.path.exists (test_puzzle_folder):
            runs = os.listdir(test_puzzle_folder)
            #runs = runs[4:]
            for run in runs:
                # breakpoint()
                run_path = os.path.join(test_puzzle_folder, run)
                folders_of_this_run = [folder for folder in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, folder)) == True]
                # print(folders_of_this_run)
                # breakpoint()
                # folders_of_this_run = folders_of_this_run[-1:]
                for folder in folders_of_this_run:
                    if "sol" in folder:
                        sol_path = os.path.join(test_puzzle_folder, run, folder, 'solution.txt')
                        if os.path.exists(sol_path):
                            print("solution found")
                            # solution as numpy array
                            solution = np.loadtxt(sol_path, dtype=str)
                            # gt as numpy array
                            ground_truth_xyz = np.loadtxt(os.path.join(preprocessing_folder, test_puzzle, 'ground_truth.txt'))
                            with open(os.path.join(preprocessing_folder, test_puzzle, 'ground_truth.json'), 'r') as jgtp:
                                ground_truth_json = json.load(jgtp)
                            # Create the dictionaries with piece id as key and the values
                            results = {}
                            ground_truth = {}
                            path_lists = {}
                            pieces_fn = natsort.natsorted(os.listdir(os.path.join(preprocessing_folder, test_puzzle, 'images')))
                            pieces_names = [piece_n[:-4] for piece_n in pieces_fn ]
                            transform_matrix = ground_truth_json['transform']
                            scaling_xyz = np.asarray([transform_matrix[0][0], transform_matrix[1][1], transform_matrix[2][2]])
                            # print(scaling_xyz)
                            for j in range(solution.shape[0]):
                                pid = solution[j][0]
                                piece_id_str = str(pid).split("_")[0]
                                # print(f'taking piece {piece_id_str}')
                                gt_piece = ground_truth_json['pieces'][piece_id_str]
                                results[pid] = solution[j][1:4].astype(float)
                                ground_truth[pid] = np.asarray([gt_piece['x'], gt_piece['y'], gt_piece['theta']])
                                piece_path = os.path.join(preprocessing_folder, test_puzzle, 'images', f"{pid}.png")
                                path_lists[pid] = piece_path

                            # results[pid] = solution[j][1:4].astype(float)
                            # ground_truth[pid] = ground_truth_xyz[j] * scaling_xyz #3.9 #
                            # piece_path = os.path.join(preprocessing_folder, test_puzzle, 'images', f"{pid}.png")
                            # path_lists[pid] = piece_path                        
                            
                        eval = Evaluation()
                        ## Evaluate
                        print("#" * 50)
                        print(f"Evaluating run {run}, sol {folder} on {test_puzzle}")    
                        avg_q_pos, avg_q_pos_best, avg_rmse_rot, avg_rmse_translation = eval.evaluate(pieces_names, results, ground_truth, path_lists, transform_matrix)
                        ## print stats
                                   
                        print(f"Q_pos: {avg_q_pos:.03f}")
                        print(f"Q_pos_Best: {avg_q_pos_best:.03f}")
                        print(f"RMSE_rot: {avg_rmse_rot:.03f}")
                        print(f"RMSE_t: {avg_rmse_translation:.03f}")
                        print("-" * 30)
                        # print params so I understand what I am evaluating
                        with open(os.path.join(test_puzzle_folder, run, folder, 'solution_output_params.yaml'), 'r') as yf:
                            params = yaml.safe_load(yf)
                        print("Parameters:")
                        print(f"CM: \t{params['aggregation']['method']}")
                        print(f"Solver:")
                        if 'anchor_index' in params['solver'].keys():
                            anchor_idx = params['solver']['anchor_index']
                        elif 'anchor_idx' in params['solver'].keys():
                            anchor_idx = params['solver']['anchor_idx']
                        else:
                            anchor_idx = params['input_params']['solver']['anchor_index']
                        print(f"\tAnchor: {anchor_idx}")
                        print(f"\tTmax: {params['solver']['T_max']}")                
                        # print(f"\tPQ_mode: {params['solver']['PQ_mode']}")
                        print("#" * 50)

                        Q_pos_list.append(avg_q_pos)
                        Q_pos_best_list.append(avg_q_pos_best)
                        RMSE_r_list.append(avg_rmse_rot)
                        RMSE_t_list.append(avg_rmse_translation)
                        CM_type_list.append(params['aggregation']['method'])
                        xy_num_points_list.append(params['grid_params']['xy_num_points'])
                        theta_num_points_list.append(params['grid_params']['theta_num_points'])
                        no_rotations_list.append(params['solver']['no_rotations'])
                        anchor_ids_list.append(anchor_idx)
                        evaluated_test_set.append(test_puzzle)
                
    eval_df = pd.DataFrame()
    eval_df['puzzle'] = evaluated_test_set
    eval_df['CM'] = CM_type_list
    eval_df['anchor'] = anchor_ids_list
    eval_df['xy_num_points'] = xy_num_points_list
    eval_df['theta_num_points'] = theta_num_points_list
    eval_df['without_rotations'] = no_rotations_list
    eval_df['Q_pos'] = Q_pos_list
    eval_df['Q_pos_Best'] = Q_pos_best_list
    eval_df['RMSE (r)'] = RMSE_r_list
    eval_df['RMSE (t)'] = RMSE_t_list
    timestamp = datetime.now().strftime('%Y%m%d')
    eval_df.to_csv(os.path.join(data_dir, f'evaluation_{timestamp}.csv'))
        

if __name__ == '__main__':
    main()