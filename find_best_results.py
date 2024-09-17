import os
import argparse

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs.envs_deterministic import find_envs as find_det_envs
from configs.algos import find_algos as find_algos
from configs.envs_stochastic import find_envs as find_stoch_envs


def find_envs(num_arms):
    # find the name of the environments.
    if args.stoch:
        environments = find_stoch_envs(num_arms, args.env_names)
    else:
        environments = find_det_envs(num_arms, args.env_names)
    
    return environments


def find_best_hyper_params(metric, env_name, experiment_name, algo, split_variables):
    results_dir = os.path.join(args.log_dir, experiment_name, env_name)
    log_files = glob.glob(f"{results_dir}/*.csv")

    # filter the logs to the current algo.
    filtered_log_files = []
    for file_name in log_files:
        df_algo = algo
        for split_variable in split_variables:
            df_algo = file_name.split('/')[-1].split('_{}'.format(split_variable))[0]
        if algo == df_algo:
            filtered_log_files.append(file_name)
    
    if len(filtered_log_files) == 0:
        return None
    
    # load the data.
    df = pd.concat([pd.read_csv(file_name, index_col=0) for file_name in filtered_log_files])

    # find each unique algo_name and seeds.
    algo_names = df['algo'].unique()
    print('*'*100)
    print(algo_names)
    seeds = df['instance_number'].unique()
    t = df[(df['algo'] == algo_names[0]) & (df['instance_number'] == seeds[0])][metric].values.shape[0]

    # for each algo and instance_number find the mean and std of the metric.
    best_hyper_params = {}
    for algo_name in algo_names:
        metric_arr = np.zeros((len(seeds), t))
        sub_opt_gap_arr = np.zeros((len(seeds), t))
        for seed in seeds:
            metric_arr[seed, :] = df[(df['algo'] == algo_name) & (df['instance_number'] == seed)][metric].values
            sub_opt_gap_arr[seed, :] = df[(df['algo'] == algo_name) & (df['instance_number'] == seed)]['sub_opt_gap'].values

        # find the mean and std of the metric.
        mean_metric = np.mean(metric_arr, axis=0)
        std_metric = np.std(metric_arr, axis=0)
        mean_sub_opt_gap = np.mean(sub_opt_gap_arr, axis=0)
        std_sub_opt_gap = np.std(sub_opt_gap_arr, axis=0)
        # print(mean_metric)

        auc = np.trapz(mean_metric)
        if auc > best_hyper_params.get('auc', -np.inf):
            best_hyper_params['auc'] = auc
            best_hyper_params['algo_name'] = algo_name
            best_hyper_params['mean_{}'.format(metric)] = mean_metric
            best_hyper_params['std_{}'.format(metric)] = std_metric
            best_hyper_params['mean_sub_opt_gap'] = mean_sub_opt_gap
            best_hyper_params['std_sub_opt_gap'] = std_sub_opt_gap

    return best_hyper_params


def main():
    # build the output dir if needed.
    output_dir = os.path.join('results', args.fig_folder)
    os.makedirs(output_dir, exist_ok=True)

    for num_arms in args.num_arms_list:
        for initialization in args.initialization_list:
            for experiment_name, exp_info in experiment_names.items():

                # build the directory for the current experiment.
                curr_experiment_name = '{}_arms_{}_init_{}'.format(experiment_name, num_arms, initialization)

                # load the environments.
                environments = find_envs(num_arms)
                # load the algorithms.
                algos = find_algos(num_arms, args.t, exp_info['algorithm_names'])

                for env in environments:
                    # create a directory for the current environment.
                    curr_env_name = env['environment_name']
                    curr_env_dir = os.path.join(output_dir, curr_experiment_name, curr_env_name)
                    os.makedirs(curr_env_dir, exist_ok=True)

                    for curr_algo in algos:
                        best_hyper_params = find_best_hyper_params(
                                                                    args.metric,
                                                                    curr_env_name,
                                                                    curr_experiment_name,
                                                                    algo=curr_algo['algo_name'],
                                                                    split_variables=exp_info['split_variables']
                                                                    )
                        if best_hyper_params is not None:
                            print(best_hyper_params['algo_name'])
                            print(best_hyper_params['mean_{}'.format(args.metric)][-10:])

                            # save the best hyper params.
                            output_path = os.path.join(curr_env_dir, '{}_best_results.npy'.format(curr_algo['algo_name']))
                            np.save(output_path, best_hyper_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='opt_action_pr')
    parser.add_argument('--stoch', type=bool, default=True)
    parser.add_argument('--split_variables', type=list, default=['eta'])
    parser.add_argument('--env_names', type=list, default=['all'])
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--num_arms_list', type=list, default=[2, 5, 10])
    parser.add_argument('--t', type=int, default=10**6)
    parser.add_argument('--initialization_list', type=str, default=['uniform', 'bad'])
    parser.add_argument('--fig_folder', type=str, default='SMDPO_VS_MDPO_PG_STOCH')

    args = parser.parse_args()
    # experiment_names = {
    # 'constant_eta': {'algorithm_names': ['mdpo_update', 'smdpo_update', 'pg_update'], 'split_variables': ['eta']},
    # 'delta_dependent_eta': {'algorithm_names': ['smdpo_delta_dependent_update'], 'split_variables': []},
    # }
    experiment_names = {
    'constant_eta_fine_gs': {'algorithm_names': ['mdpo_update', 'smdpo_update'], 'split_variables': ['eta']},
    'constant_eta': {'algorithm_names': ['spg_update'], 'split_variables': ['eta']},
    }

    main()


