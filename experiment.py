import time
import jax
import jax.numpy as jnp
from jax.nn import softmax
from collections import namedtuple
from scipy.special import lambertw
import tqdm

from bandit_environments import make_envs
from updates import (
    det_pg,
    det_pg_ls,
    det_pg_transformed_ls,
    det_gnpg,
    det_pg_entropy,
    det_pg_entropy_multistage,
)
from updates import (
    spg,
    spg_gradient_step_size,
    spg_entropy,
    spg_entropy_multistage,
    spg_multistage,
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


def log_data(theta, pistar, env, algo_name, optimal_action, t, run_number):
    pi = softmax(theta)

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
    if "det_pg" == algo_name:
        gradient_update = det_pg
    elif "det_pg_ls" == algo_name or "det_pg_ls_increasing" == algo_name:
        gradient_update = det_pg_ls
        if "det_pg_ls_increasing" == algo_name:
            eta_max_ratio = algo_kwargs.pop("r")

            def det_pg_ls_step_size_update(algo_kwargs, eta):
                if eta == algo_kwargs["eta_max"]:
                    algo_kwargs["eta_max"] *= eta_max_ratio
                return algo_kwargs

    elif "det_pg_transformed_ls" == algo_name:
        gradient_update = det_pg_transformed_ls
        algo_kwargs["eta_max"] = 1 / algo_kwargs["eps"]

    elif "det_gnpg" == algo_name:
        gradient_update = det_gnpg
    elif "det_pg_entropy" == algo_name:
        gradient_update = det_pg_entropy
        L_tau = 5 / 2 + algo_kwargs["tau"] * 5 * (1 + jnp.log(env.K))
        algo_kwargs["eta"] = 1 / L_tau
    elif "det_pg_entropy_multistage" == algo_name:
        gradient_update = det_pg_entropy_multistage
        stage_start = 1

        w_term = lambertw(env.K - 1).real / jnp.e
        B_4 = w_term + jnp.log(env.K)
        L_tau = 5 / 2 + algo_kwargs["tau"] * 5 * (1 + jnp.log(env.K))

        algo_kwargs['eta'] = 1 / L_tau

        algo_kwargs["stage_length"] = jnp.log(
            2 * (1 + B_4)
        ) / (algo_kwargs['eta'] * algo_kwargs["tau"] ** algo_kwargs["p"] * algo_kwargs["B_1"])

        def multistage_stage_update(algo_kwargs):
            algo_kwargs["tau"] /= 2

            L_tau = 5 / 2 + algo_kwargs["tau"] * 5 * (1 + jnp.log(env.K))
            algo_kwargs['eta'] = 1 / L_tau

            algo_kwargs["stage_length"] = jnp.log(2 * (1 + B_4)) / (
                algo_kwargs['eta'] * algo_kwargs["tau"] ** algo_kwargs["p"] * algo_kwargs["B_1"]
            )

            return algo_kwargs

    elif "spg_delta_step_size" in algo_name:
        gradient_update = spg
        # step-size is problem dependent
        Delta = jnp.min(jnp.abs(env.mean_r[1:] - env.mean_r[:-1]))
        R_max = 1
        algo_kwargs["eta"] = Delta**2 / (40 * len(env.mean_r) ** (3 / 2) * R_max)
    elif "spg_gradient_step_size" in algo_name:
        gradient_update = spg_gradient_step_size
    elif "spg_ess" in algo_name:
        gradient_update = spg

        alpha = algo_kwargs.pop("alpha")
        eta = algo_kwargs.pop("eta_0")
        algo_kwargs["eta"] = eta * alpha

        def ess_step_size_update(algo_kwargs):
            algo_kwargs["eta"] *= alpha
            return algo_kwargs

    elif "spg_multistage_ess" == algo_name:
        gradient_update = spg_multistage
        stage_start = 1

        beta = algo_kwargs.pop("beta")
        eta_0 = algo_kwargs.pop("eta_0")
        algo_kwargs["eta"] = eta_0 * (beta / algo_kwargs["stage_length"]) ** (
            1 / algo_kwargs["stage_length"]
        )

        def ess_step_size_update(algo_kwargs):
            alpha = (beta / algo_kwargs["stage_length"]) ** (
                1 / algo_kwargs["stage_length"]
            )
            algo_kwargs["eta"] *= alpha
            return algo_kwargs

        def multistage_stage_update(algo_kwargs):
            algo_kwargs["stage_length"] *= 2

            alpha = (beta / algo_kwargs["stage_length"]) ** (
                1 / algo_kwargs["stage_length"]
            )

            algo_kwargs["eta"] = eta_0 * alpha

            return algo_kwargs

    elif "spg_entropy_ess" in algo_name:
        gradient_update = spg_entropy
        alpha = algo_kwargs.pop("alpha")
        eta = algo_kwargs.pop("eta_0")
        algo_kwargs["eta"] = eta * alpha

        def ess_step_size_update(algo_kwargs):
            algo_kwargs["eta"] *= alpha
            return algo_kwargs

    elif "spg_entropy_multistage" in algo_name:
        gradient_update = spg_entropy_multistage

        stage_start = 1

        beta = algo_kwargs.pop("beta")
        alpha = (beta / algo_kwargs["stage_length"]) ** (
            1 / algo_kwargs["stage_length"]
        )
        L_tau = 5 / 2 + algo_kwargs["tau"] * 5 * (1 + jnp.log(env.K))
        eta_0 = 1 / L_tau
        algo_kwargs["eta"] = eta_0 * alpha

        def ess_step_size_update(algo_kwargs):
            algo_kwargs["eta"] *= (beta / algo_kwargs["stage_length"]) ** (
                1 / algo_kwargs["stage_length"]
            )
            return algo_kwargs

        def multistage_stage_update(algo_kwargs):
            algo_kwargs["tau"] /= 2
            algo_kwargs["stage_length"] *= 2

            L_tau = 5 / 2 + algo_kwargs["tau"] * 5 * (1 + jnp.log(env.K))
            eta_0 = 1 / L_tau

            alpha = (beta / algo_kwargs["stage_length"]) ** (
                1 / algo_kwargs["stage_length"]
            )

            algo_kwargs["eta"] = eta_0 * alpha

            return algo_kwargs

    else:
        assert False, f"Unknown algorithm: {algo_name}"

    optimal_action = env.mean_r.argmax()
    pistar = jax.nn.one_hot(optimal_action, len(env.mean_r))

    log = []
    log.append(
        log_data(
            theta_0, pistar, env, algo_name, optimal_action, t=0, run_number=run_number
        )
    )

    theta = theta_0

    @jax.jit
    def bandit_update(key, theta, **algo_kwargs):
        key, reward_key, action_key = jax.random.split(key, 3)
        reward = env.randomize(reward_key)
        theta, eta = gradient_update(action_key, theta, reward, **algo_kwargs)
        return key, theta, eta

    @jax.jit
    def terminate_condition(theta):
        @jax.grad
        def df(theta):
            return jax.nn.softmax(theta) @ env.mean_r

        return jnp.linalg.norm(df(theta)) < 1e-8

    total_time = 0
    for t in tqdm.tqdm(range(1, T + 1), position=1, desc="T", leave=False):
        tik = time.time()
        key, theta, eta = bandit_update(key, theta, **algo_kwargs)
        elapsed_time = time.time() - tik
        total_time += elapsed_time

        # step-size updates if needed
        # outside of the jit-ed function since 'if' statements are sometimes needed
        if "det_pg_ls_increasing" == algo_name:
            algo_kwargs = det_pg_ls_step_size_update(algo_kwargs, eta)
        elif "det_pg_entropy_multistage" in algo_name:
            if t - stage_start >= algo_kwargs["stage_length"]:
                stage_start = t
                algo_kwargs = multistage_stage_update(algo_kwargs)
        elif "spg_ess" in algo_name or "spg_entropy_ess" in algo_name:
            algo_kwargs = ess_step_size_update(algo_kwargs)
        elif "spg_multistage_ess" == algo_name:
            algo_kwargs = ess_step_size_update(algo_kwargs)
            if t - stage_start >= algo_kwargs["stage_length"]:
                stage_start = t
                algo_kwargs = multistage_stage_update(algo_kwargs)
        elif "spg_entropy_multistage" in algo_name:
            algo_kwargs = ess_step_size_update(algo_kwargs)
            if t - stage_start >= algo_kwargs["stage_length"]:
                stage_start = t
                algo_kwargs = multistage_stage_update(algo_kwargs)

        if terminate_condition(theta):
            print()
            print(
                f"Gradient norm is small (< 1e-8) at iteration {t}, terminating, early"
            )
            log.append(
                log_data(
                    theta,
                    pistar,
                    env,
                    algo_name,
                    optimal_action,
                    t=t,
                    run_number=run_number,
                )
            )
            return log, total_time

        if t % time_to_log == 0:
            log.append(
                log_data(
                    theta,
                    pistar,
                    env,
                    algo_name,
                    optimal_action,
                    t=t,
                    run_number=run_number,
                )
            )

    print(f"\nFinal theta: {theta}")
    return log, total_time


def run_experiment(
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
