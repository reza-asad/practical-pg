import jax
from absl import app, flags


# opt in early to change in JAX's RNG generation
# https://github.com/google/jax/discussions/18480
jax.config.update("jax_threefry_partitionable", True)
# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platform_name", "cpu")

from experiment import run_experiment
from experiment_functional import run_experiment_functional
from configs.envs_stochastic import find_envs
from configs.algos import find_algos

FLAGS = flags.FLAGS

flags.DEFINE_list("env_names", 'Gaussian (hard)', "Environment name to run")
flags.DEFINE_list("algo_names", 'smdpo_update', "Environment name to run ")
flags.DEFINE_string("initial_policy", "uniform", "Initial policy to use {uniform, bad}")

flags.DEFINE_integer("t", 10**5, "Number of iterations")
flags.DEFINE_string("exp_name", "debug", "Experiment Name")
flags.DEFINE_string("save_dir", "./logs/", "Log directory")
flags.DEFINE_integer("runs_per_instance", 1, "Runs per instance")
flags.DEFINE_integer("num_instances", 50, "Number of instance")
flags.DEFINE_integer("env_seed", 1337, "Environment Seed")
flags.DEFINE_integer("exp_seed", 100, "Experiment Seed")

# my flags
flags.DEFINE_string("functional_update", 'True', "Use functional update")
flags.DEFINE_float("eta", 0.001, "Step size for the update")
flags.DEFINE_integer("num_arms", 2, "Number of arms")


def main(_):
    FLAGS.functional_update = False if FLAGS.functional_update == 'False' else True

    # load the env using json.
    envs = find_envs(FLAGS.num_arms, FLAGS.env_names)
    print(envs)

    algos = find_algos(FLAGS.num_arms, FLAGS.t, FLAGS.algo_names)

    # concat arm number to the experiment name
    FLAGS.exp_name = f"{FLAGS.exp_name}_arms_{FLAGS.num_arms}"

    # concat initial policy to the experiment name
    FLAGS.exp_name = f"{FLAGS.exp_name}_init_{FLAGS.initial_policy}"

    # concat eta to algo name if needed.
    if FLAGS.eta != -1:
        for algo in algos:
            if "eta" in algo["algo_kwargs"]:
                algo["algo_kwargs"]["eta"] = FLAGS.eta
            algo["algo_name"] = f"{algo['algo_name']}_eta_{FLAGS.eta}"

    print(algos)

    if FLAGS.functional_update:
        run_experiment_functional(
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
    else:
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
