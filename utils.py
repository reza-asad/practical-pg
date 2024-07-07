import os
import numpy as np
import pandas as pd
import json

def save_experiment(path, experiment_name, logs, env_def, algo_name, inital_policy, total_times):
    dir = f"{path}/{experiment_name}/{env_def['environment_name']}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    df = pd.DataFrame(logs)
    df.to_csv(f"{dir}/{algo_name}_{inital_policy}_bandit.csv")

    np.save(f"{dir}/{algo_name}_{inital_policy}_time.txt", np.array(total_times))
    print(f"average  time: {np.array(total_times).mean()}")

    # make local to not modify the original
    env_def = env_def.copy()
    # change the Bandit class to a string to be able to save it
    env_def['Bandit'] = env_def['Bandit'].name
    env_def['bandit_kwargs'].pop('mean_r', None)
    with open(f"{dir}/environment_def.json", 'w' ) as f:
        json.dump(env_def, f)