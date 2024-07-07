# %%
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import time

jax.config.update("jax_threefry_partitionable", True)
# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platform_name", "cpu")

from mdp_environments import (
    FlatGrad,
    DeepSeaTreasure,
    CliffWorld,
)

from mdp_updates import (
    calc_pistar,
    calc_vpi,
    det_pg_ls,
    det_pg_transformed_ls,
    det_pg_A,
    mdp_gnpg,
    det_pg,
    policy_evaluation,
    policy_greedy,
)


envs_name = ["Cliff World", "Deep Sea Teasure", "Flat Grad"]
envs = [CliffWorld, DeepSeaTreasure, FlatGrad]

def plot(ax, env_name, sub_opt_gaps, label):
    start = sub_opt_gaps[0].item() + 0.01
    ax.plot(np.arange(len(sub_opt_gaps)), sub_opt_gaps, label=label)
    ax.set_xlabel("Log Iteration")
    ax.set_xscale("symlog")

    if env_name == "Flat Grad":
        ax.set_ylim(1.0e-5, 1)
    else:
        ax.set_ylim(1.0e-5, start)

    ax.set_ylabel("Log Suboptimality Gap")
    ax.set_yscale("log")


def plot_pg_ls_experiment(ax, env, env_name, sub_opt_gap):
    theta = jnp.zeros_like(env.r)
    sub_opt_gaps = [sub_opt_gap(env, theta)]

    eps = 1e-4
    for _ in range(10_000):
        theta, eta = det_pg_ls(key, theta, env, 1 / eps, c=0.5, beta=0.1)
        sub_opt_gaps.append(sub_opt_gap(env, theta))

    ax.set_title(env_name)
    plot(ax, env_name, sub_opt_gaps, "PG-LS")


def plot_pg_transformed_ls_experiment(ax, env, env_name, sub_opt_gap):
    theta = jnp.zeros_like(env.r)
    sub_opt_gaps = [sub_opt_gap(env, theta)]

    eps = 1e-4
    for _ in range(10_000):
        theta, eta = det_pg_transformed_ls(
            key, theta, env, beta=0.1, c=0.5, f_star=normalized_f_star, eta_max=1 / eps
        )
        sub_opt_gaps.append(sub_opt_gap(env, theta))

    plot(ax, env_name, sub_opt_gaps, "PG-Log-LS")


def plot_gnpg_experiment(ax, env, env_name, sub_opt_gap):
    theta = jnp.zeros_like(env.r)
    sub_opt_gaps = [sub_opt_gap(env, theta)]

    for _ in range(10000):
        theta, eta = mdp_gnpg(key, theta, env)
        sub_opt_gaps.append(sub_opt_gap(env, theta))

    plot(ax, env_name, sub_opt_gaps, "GNPG")


def plot_pg_A_experiment(ax, env, env_name, sub_opt_gap):
    theta = jnp.zeros_like(env.r)
    sub_opt_gaps = [sub_opt_gap(env, theta)]

    for _ in range(10_000):
        theta, eta = det_pg_A(key, theta, env)
        sub_opt_gaps.append(sub_opt_gap(env, theta))

    plot(ax, env_name, sub_opt_gaps, "PG-A")


def plot_pg_experiment(ax, env, env_name, sub_opt_gap):
    theta = jnp.zeros_like(env.r)
    sub_opt_gaps = [sub_opt_gap(env, theta)]

    L = 8 / (1 - env.gamma) ** 3
    for _ in range(10_000):
        theta, eta = det_pg(key, theta, env, 1 / L)
        sub_opt_gaps.append(sub_opt_gap(env, theta))

    plot(ax, env_name, sub_opt_gaps, "PG")


def plot_PI_experiment(ax, env, env_name, sub_opt_gap):
    pi = jnp.ones((env.S, env.A)) / env.A
    sub_opt_gaps = [sub_opt_gap(env, None, pi)]

    for _ in range(10):
        v_pi = policy_evaluation(pi, env)
        pi_hat_k = policy_greedy(v_pi, env)
        pi = pi_hat_k

        sub_opt_gaps.append(sub_opt_gap(env, None, pi))

    plot(ax, env_name, sub_opt_gaps, "PI")


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i in range(len(envs)):
    env = envs[i]
    env_name = envs_name[i]
    print('Running', env_name)

    pi_star = calc_pistar(env)
    v_pistar = calc_vpi(pi_star, env.P, env.r, env.gamma)
    normalized_f_star = v_pistar @ env.rho * (1 - env.gamma)

    key = jax.random.PRNGKey(0)
    theta = jnp.zeros_like(env.r)

    def sub_opt_gap(env, theta=None, pi=None):
        if theta is not None:
            pi = jax.nn.softmax(theta, axis=-1)
        v_pi = calc_vpi(pi, env.P, env.r, env.gamma)
        normalized_f = v_pi @ env.rho * (1 - env.gamma)
        return normalized_f_star - normalized_f

    print('\tPG-LS')
    plot_pg_ls_experiment(axes[i], env, env_name, sub_opt_gap)
    print('\tPG-Log-LS')
    plot_pg_transformed_ls_experiment(axes[i], env, env_name, sub_opt_gap)
    print('\tGNPG')
    plot_gnpg_experiment(axes[i], env, env_name, sub_opt_gap)
    print('\tPG-A')
    plot_pg_A_experiment(axes[i], env, env_name, sub_opt_gap)
    print('\tPG')
    plot_pg_experiment(axes[i], env, env_name, sub_opt_gap)
    # print('\tPI')
    # plot_PI_experiment(axes[i], env, env_name, sub_opt_gap)

lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc="lower center", ncol=9, bbox_to_anchor=[0.5, -0.10])
plt.tight_layout()
plt.savefig("plots/det_mdp.png", bbox_inches="tight", dpi=400)
# %%
