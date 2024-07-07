import jax
# opt in early to change in JAX's RNG generation
# https://github.com/google/jax/discussions/18480
jax.config.update("jax_threefry_partitionable", True)
# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platform_name", "cpu")

from bandit_environments import Bandit
from experiment import run_experiment

T = 1_000_000
TIME_TO_LOG = T // 10
NUM_ARMS = 10
LOG_DIR = f"logs"
EXP_NAME = f"bad_init"
INTIAL_POLICY = "bad"
NUM_INSTANCES = 50
ENV_SEED = 1337
EXP_SEED = 1337 + 42

environment_definitions = [
    {
        "Bandit": Bandit,
        "bandit_kwargs": {"K": NUM_ARMS},
        "min_reward_gap": 0.05,
        "max_reward_gap": 0.5,
        "environment_name": "Deterministic (0.05)",
    },
    {
        "Bandit": Bandit,
        "bandit_kwargs": {"K": NUM_ARMS},
        "min_reward_gap": 0.1,
        "max_reward_gap": 0.5,
        "environment_name": "Deterministic (0.1)",
    },
    {
        "Bandit": Bandit,
        "bandit_kwargs": {"K": NUM_ARMS},
        "min_reward_gap": 0.2,
        "max_reward_gap": 0.5,
        "environment_name": "Deterministic (0.2)",
    },
]

L = 5 / 2

algos = [
    {
        "algo_name": "det_pg",
        "algo_kwargs": {"eta": 1 / L},
    },
    {
        "algo_name": "det_pg_entropy",
        "algo_kwargs": {
            "tau": 0.1
        },  # step-size depends on `L^tau` which requires knowledge of number of arms and is computed later
    },
    {
        "algo_name": "det_pg_entropy_multistage",
        "algo_kwargs": {
            "tau": 0.1,
            "p": 1,
            "B_1": 0.01,
        },  # step-size depends on `L^tau` which requires knowledge of number of arms and is computed in the update
    },
]

run_experiment(
    environment_definitions,
    algos,
    T=T,
    environment_seed=ENV_SEED,
    experiment_seed=EXP_SEED,
    num_instances=NUM_INSTANCES,
    runs_per_instance=1,
    time_to_log=TIME_TO_LOG,
    log_dir=LOG_DIR,
    exp_name=EXP_NAME,
    intial_policy=INTIAL_POLICY,
)
