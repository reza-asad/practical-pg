import time
import jax
import jax.numpy as jnp
from jax.nn import softmax
from collections import namedtuple
from scipy.special import lambertw
import tqdm

from bandit_environments import make_envs
from updates import (
    mdpo_stoch,
    mdpo,
    smdpo_stoch,
    smdpo,
    smdpo_delta_dependent,
)
from utils import save_experiment

BanditData = namedtuple(
    "BanditData",
    [
        "iteration",
        "expected_reward",
        "sub_opt_gap",
        "opt_action_pr",
        "algo",
        "env_name",
        "instance_number",
        "run_number",
    ],
)

def log_data(pi, pistar, env, algo_name, optimal_action, t, run_number):
    sub_opt_gap = ((pistar - pi) @ env.mean_r).item()
    data = BanditData(
        iteration=t,
        expected_reward=(pi @ env.mean_r).item(),
        algo=algo_name,
        env_name=env.name,
        sub_opt_gap=sub_opt_gap,
        opt_action_pr=pi[optimal_action].item(),
        instance_number=env.instance_number,
        run_number=run_number,
    )

    return data


def run_bandit_experiment(
    algo_name, algo_kwargs, env, theta_0, key, T, time_to_log, run_number
):
    print(f"{theta_0}")
    print(env.mean_r)
    # map algo_name to update and specific any additional kwargs

    if "smdpo" in algo_name:
        if 'Deterministic' in env.name:
            if "delta_dependent" in algo_name:
                gradient_update = smdpo_delta_dependent
            else:
                gradient_update = smdpo
        else:
            gradient_update = smdpo_stoch
    elif "mdpo" in algo_name:
        if 'Deterministic' in env.name:
            gradient_update = mdpo
        else:
            gradient_update = mdpo_stoch
    else:
        assert False, f"Unknown algorithm: {algo_name}"

    optimal_action = env.mean_r.argmax()
    pistar = jax.nn.one_hot(optimal_action, len(env.mean_r))
    pi = softmax(theta_0)
    
    log = []
    log.append(
        log_data(
            pi, pistar, env, algo_name, optimal_action, t=0, run_number=run_number
        )
    )

    @jax.jit
    def bandit_update(key, pi, **algo_kwargs):
        key, reward_key, action_key = jax.random.split(key, 3)
        reward = env.randomize(reward_key)
        pi, eta = gradient_update(action_key, pi, reward, **algo_kwargs)
        return key, pi, eta

    @jax.jit
    def terminate_condition(theta):
        @jax.grad
        def df(theta):
            return jax.nn.softmax(theta) @ env.mean_r

        return jnp.linalg.norm(df(theta)) < 1e-8

    total_time = 0
    for t in tqdm.tqdm(range(1, T + 1), position=1, desc="T", leave=False):
        tik = time.time()

        # _, reward_key, action_key = jax.random.split(key, 3)
        # reward = env.randomize(reward_key)
        # action = jax.random.choice(action_key, len(reward), p=pi)
        # reward_hat = jax.nn.one_hot(action, len(reward)) / pi * reward
        # stoch_grad = pi * (reward_hat - pi.dot(reward_hat)) 
        # pi = pi + algo_kwargs['eta'] * stoch_grad
        # print('pi: {}, stoch_grad: {}, reward_hat: {}'.format(pi, stoch_grad, reward_hat))
        # print('*' * 100)

        key, pi, eta = bandit_update(key, pi, **algo_kwargs)
        elapsed_time = time.time() - tik
        total_time += elapsed_time

        # # step-size updates if needed
        # # outside of the jit-ed function since 'if' statements are sometimes needed
        # if "det_pg_ls_increasing" == algo_name:
        #     algo_kwargs = det_pg_ls_step_size_update(algo_kwargs, eta)
        # elif "det_pg_entropy_multistage" in algo_name:
        #     if t - stage_start >= algo_kwargs["stage_length"]:
        #         stage_start = t
        #         algo_kwargs = multistage_stage_update(algo_kwargs)
        # elif "spg_ess" in algo_name or "spg_entropy_ess" in algo_name:
        #     algo_kwargs = ess_step_size_update(algo_kwargs)
        # elif "spg_multistage_ess" == algo_name:
        #     algo_kwargs = ess_step_size_update(algo_kwargs)
        #     if t - stage_start >= algo_kwargs["stage_length"]:
        #         stage_start = t
        #         algo_kwargs = multistage_stage_update(algo_kwargs)
        # elif "spg_entropy_multistage" in algo_name:
        #     algo_kwargs = ess_step_size_update(algo_kwargs)
        #     if t - stage_start >= algo_kwargs["stage_length"]:
        #         stage_start = t
        #         algo_kwargs = multistage_stage_update(algo_kwargs)

        # if terminate_condition(theta):
        #     print()
        #     print(
        #         f"Gradient norm is small (< 1e-8) at iteration {t}, terminating, early"
        #     )
        #     log.append(
        #         log_data(
        #             theta,
        #             pistar,
        #             env,
        #             algo_name,
        #             optimal_action,
        #             t=t,
        #             run_number=run_number,
        #         )
        #     )
        #     return log, total_time

        if t % time_to_log == 0:
            log.append(
                log_data(
                    pi,
                    pistar,
                    env,
                    algo_name,
                    optimal_action,
                    t=t,
                    run_number=run_number,
                )
            )

    print(f"\nFinal pi: {pi}")
    return log, total_time


def run_experiment_functional(
    environment_definitions,
    algos,
    T,
    environment_seed,
    experiment_seed,
    num_instances,
    time_to_log,
    log_dir,
    exp_name,
    runs_per_instance,
    intial_policy="uniform",
):
    assert intial_policy in [
        "uniform",
        "bad",
    ], f"Unknown initial policy: {intial_policy}"

    for env_def in environment_definitions:
        print(f"Running experiment on: {env_def['environment_name']}")

        env_key = jax.random.PRNGKey(environment_seed)
        # generate a new environment with random mean reward vector in [0, 1]^K for each run
        envs = make_envs(env_def, num_instances, env_key)

        key = jax.random.PRNGKey(experiment_seed)

        for algo in algos:
            print(f"Algorithm: {algo['algo_name']}")

            logs = []
            times = []
            for env in tqdm.tqdm(envs, position=0, desc="envs"):

                for run_number in range(runs_per_instance):
                    theta_0 = jnp.zeros_like(env.mean_r)

                    # let the worse action have a high probability of being selected
                    if intial_policy == "bad":
                        theta_0 = theta_0.at[0].set(12)

                    key, exp_key = jax.random.split(key)
                    log, total_time = run_bandit_experiment(
                        env=env,
                        run_number=run_number,
                        theta_0=theta_0,
                        key=exp_key,
                        time_to_log=time_to_log,
                        T=T,
                        algo_name=algo["algo_name"],
                        algo_kwargs=algo["algo_kwargs"].copy(),
                    )
                    logs.extend(log)
                    times.append(total_time)

            save_experiment(
                log_dir,
                exp_name,
                logs,
                env_def,
                algo["algo_name"],
                intial_policy,
                times,
            )
