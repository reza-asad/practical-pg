import jax
from absl import app, flags


# opt in early to change in JAX's RNG generation
# https://github.com/google/jax/discussions/18480
jax.config.update("jax_threefry_partitionable", True)
# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from bandit_environments import BerBandit, BetaBandit, FixedBandit, GaussBandit
from experiment import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_integer("env_number", -1, "Environment number to run (-1 for all)")
flags.DEFINE_integer("algo_number", -1, "Environment number to run (-1 for all)")
flags.DEFINE_string("initial_policy", "uniform", "Initial policy to use {uniform, bad}")

flags.DEFINE_integer("t", 100, "Number of iterations")
flags.DEFINE_string("exp_name", "uni_init", "Experiment Name")
flags.DEFINE_string("save_dir", "./logs/", "Log directory")
flags.DEFINE_integer("runs_per_instance", 1, "Runs per instance")
flags.DEFINE_integer("num_instances", 1, "Number of instance")
flags.DEFINE_integer("env_seed", 1337, "Environment Seed")
flags.DEFINE_integer("exp_seed", 100, "Experiment Seed")

NUM_ARMS = 10


environment_definitions = [
    {
        "Bandit": BetaBandit,
        "bandit_kwargs": {"a_plus_b": 4, "K": NUM_ARMS},
        "max_reward_gap": 0.5,
        "environment_name": "Beta (easy)",
    },
    {
        "Bandit": BetaBandit,
        "bandit_kwargs": {"a_plus_b": 4, "K": NUM_ARMS},
        "max_reward_gap": 0.1,
        "environment_name": "Beta (hard)",
    },
    {
        "Bandit": GaussBandit,
        "bandit_kwargs": {"sigma": 0.1, "K": NUM_ARMS},
        "max_reward_gap": 0.5,
        "environment_name": "Gaussian (easy)",
    },
    {
        "Bandit": GaussBandit,
        "bandit_kwargs": {"sigma": 0.1, "K": NUM_ARMS},
        "max_reward_gap": 0.1,
        "environment_name": "Gaussian (hard)",
    },
    {
        "Bandit": BerBandit,
        "bandit_kwargs": {"K": NUM_ARMS},
        "max_gap": 0.5,
        "environment_name": "Bernoulli (easy)",
    },
    {
        "Bandit": BerBandit,
        "bandit_kwargs": {"K": NUM_ARMS},
        "max_gap": 0.1,
        "environment_name": "Bernoulli (hard)",
    },
]

L = 5 / 2
L_ENT = L + 0.1 * 5 * (1 + jnp.log(NUM_ARMS))


def main(_):
    envs = []
    if FLAGS.env_number == -1:
        envs = environment_definitions
    else:
        envs.append(environment_definitions[FLAGS.env_number])
    print(envs)

    ALGOS = [
        {
            "algo_name": "spg_ess_eta_0_1_18", # SPG-ESS
            "algo_kwargs": {"alpha": (1 / FLAGS.t) ** (1 / FLAGS.t), "eta_0": 1 / 18},
        },
        {
            "algo_name": "spg_multistage_ess", # SPG-ESS [D]
            "algo_kwargs": {"beta": 1, "eta_0": 1 / 18, "stage_length": 5000},
        },
        {
            "algo_name": "spg_gradient_step_size", # SPG-O-G
            "algo_kwargs": {"eta": None},
        },
        {
            "algo_name": "spg_delta_step_size", # SPG-O-D
            "algo_kwargs": {
                "eta": None  # step-size depends on problem dependent reward gap and is set later
            },
        },
        {
            "algo_name": "spg_entropy_multistage",
            "algo_kwargs": {"beta": 1, "stage_length": 5000, "tau": 0.5},
        },
        {
            "algo_name": f"spg_entropy_ess",
            "algo_kwargs": {
                "alpha": (1 / FLAGS.t) ** (1 / FLAGS.t),
                "eta_0": 1 / L_ENT,
                "tau": 0.1,
            },
        },
    ]

    algos = []
    if FLAGS.algo_number == -1:
        algos = ALGOS
    else:
        algos.append(ALGOS[FLAGS.algo_number])
    print(algos)

    run_experiment(
        envs,
        algos,
        T=FLAGS.t,
        environment_seed=FLAGS.env_seed,
        experiment_seed=FLAGS.exp_seed,
        num_instances=FLAGS.num_instances,
        runs_per_instance=FLAGS.runs_per_instance,
        time_to_log=FLAGS.t // 100,
        log_dir=FLAGS.save_dir,
        exp_name=FLAGS.exp_name,
        intial_policy=FLAGS.initial_policy,
    )


if __name__ in "__main__":
    app.run(main)
