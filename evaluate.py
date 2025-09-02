from evaluation import Evaluation
import os 
import numpy as np
import natsort 
import yaml 
import pandas as pd 
from datetime import datetime


def main():

    data_dir = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca"
    #data_dir = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=m.khoroshiltseva/home/ssd/datasets/RePAIR_ReLab_marina/"
    test_set = np.loadtxt(os.path.join(data_dir, 'PAD_v2', 'test.txt'), dtype=str)
    experiments_folder = os.path.join(data_dir, 'experiments')
    preprocessing_folder = os.path.join(data_dir, 'preprocessing')
    Q_pos_list = []
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
        runs = os.listdir(test_puzzle_folder)
        for run in runs:
            # breakpoint()
            run_path = os.path.join(test_puzzle_folder, run)
            folders_of_this_run = [folder for folder in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, folder)) == True]
            # print(folders_of_this_run)
            # breakpoint()
            for folder in folders_of_this_run:
                if "sol" in folder:
                    sol_path = os.path.join(test_puzzle_folder, run, folder, 'solution.txt')
                    if os.path.exists(sol_path):
                        print("solution found")
                        # solution as numpy array
                        solution = np.loadtxt(sol_path, dtype=str)
                        # gt as numpy array
                        ground_truth_xyz = np.loadtxt(os.path.join(preprocessing_folder, test_puzzle, 'ground_truth.txt'))
                        # Create the dictionaries with piece id as key and the values
                        results = {}
                        ground_truth = {}
                        path_lists = {}
                        pieces_fn = natsort.natsorted(os.listdir(os.path.join(preprocessing_folder, test_puzzle, 'images')))
                        pieces = [piece_n[:-4] for piece_n in pieces_fn ]
                        for j in range(solution.shape[0]):
                            pid = solution[j][0]
                            results[pid] = solution[j][0:3].astype(float)
                            ground_truth[pid] = ground_truth_xyz[j] / 3.9
                            piece_path = os.path.join(preprocessing_folder, test_puzzle, 'images', f"{pid}.png")
                            path_lists[pid] = piece_path                        eval = Evaluation()
                        ## Evaluate
                        avg_q_pos, avg_rmse_rot, avg_rmse_translation = eval.evaluate(pieces, results, ground_truth, path_lists)
                        ## print stats
                        print("#" * 50)
                        print(f"Evaluating run {run} on {test_puzzle}")                
                        print(f"Q_pos: {avg_q_pos:.03f}")
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
    eval_df['RMSE (r)'] = RMSE_r_list
    eval_df['RMSE (t)'] = RMSE_t_list
    timestamp = datetime.now().strftime('%Y%m%d')
    eval_df.to_csv(os.path.join(data_dir, f'evaluation_{timestamp}.csv'))
        

if __name__ == '__main__':
    main()