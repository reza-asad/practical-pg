import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm


@jax.jit
def vpi(env, pi):
    p_pi = jnp.einsum("xay,xa->xy", env.P, pi)
    r_pi = jnp.einsum("xa,xa->x", env.r, pi)
    v_pi = jnp.linalg.solve(jnp.eye(env.P.shape[0]) - env.gamma * p_pi, r_pi)
    return v_pi


@jax.jit
def qpi(env, pi):
    v_pi = vpi(env, pi)
    q_pi = env.r + env.gamma * jnp.einsum("xay, y-> xa", env.P, v_pi)
    return q_pi


@jax.jit
def dpi(env, pi):
    p_pi = jnp.einsum("xay,xa->xy", env.P, pi)
    d_pi = (1 - env.gamma) * jnp.linalg.solve(
        jnp.eye(env.P.shape[0]) - env.gamma * p_pi.T, env.rho
    )
    d_pi = d_pi / d_pi.sum()  # for addressing numerical errors
    return d_pi


@jax.jit
def J(env, pi):
    return env.rho @ vpi(env, pi)


@jax.jit
def grad_J(env, theta):
    pi = jax.nn.softmax(theta, axis=1)
    d_pi = dpi(env, pi)
    q_pi = qpi(env, pi)

    @jax.grad
    def gradJ(theta):
        return jnp.dot(
            d_pi, jnp.einsum("xa, xa-> x", jax.nn.softmax(theta, axis=1), q_pi)
        ) / (1 - env.gamma)

    return gradJ(theta)


def armijo_line_search(env, f, theta, eta_max, c, beta):
    gradient = grad_J(env, theta)

    def armijo_condition_not_satisfied(eta):
        return f(theta + eta * gradient) < f(theta) + c * eta * norm(gradient) ** 2

    def reduce_eta(eta):
        return eta * beta

    eta = jax.lax.while_loop(armijo_condition_not_satisfied, reduce_eta, eta_max)

    return eta


@jax.jit
def normalized_grad_J(env, theta):
    pi = jax.nn.softmax(theta, axis=1)
    d_pi = dpi(env, pi)
    q_pi = qpi(env, pi) * (1 - env.gamma)

    @jax.grad
    def gradJ(theta):
        return jnp.dot(
            d_pi, jnp.einsum("xa, xa-> x", jax.nn.softmax(theta, axis=1), q_pi)
        ) / (1 - env.gamma)

    return gradJ(theta)


def transformed_armijo_line_search(
    env, normalized_f, normalized_f_star, theta, eta_max, c, beta
):
    gradient = normalized_grad_J(env, theta)

    def armijo_condition_not_satisfied(eta):
        return jnp.log(normalized_f_star - normalized_f(theta + eta * gradient)) > (
            jnp.log(normalized_f_star - normalized_f(theta))
            - c * eta * norm(gradient) ** 2 / (normalized_f_star - normalized_f(theta))
        )

    def reduce_eta(eta):
        return eta * beta

    eta = jax.lax.while_loop(armijo_condition_not_satisfied, reduce_eta, eta_max)

    return eta


@jax.jit
def det_pg(key, theta, env, eta):

    grad = grad_J(env, theta)
    theta = theta + eta * grad

    return theta, eta


@jax.jit
def det_pg_ls(key, theta, env, eta_max, c, beta):

    def f(theta):
        pi = jax.nn.softmax(theta, axis=-1)
        return env.rho @ vpi(env, pi)

    eta = armijo_line_search(env, f, theta, eta_max, c, beta)
    grad = grad_J(env, theta)
    theta = theta + eta * grad

    return theta, eta


@jax.jit
def det_pg_transformed_ls(key, theta, env, eta_max, c, beta, f_star):

    def f(theta):
        pi = jax.nn.softmax(theta, axis=-1)
        return env.rho @ vpi(env, pi) * (1 - env.gamma)

    eta = transformed_armijo_line_search(env, f, f_star, theta, eta_max, c, beta)
    grad = normalized_grad_J(env, theta)
    theta = theta + eta * grad

    return theta, eta


def det_pg_A(key, theta, env):
    grad = grad_J(env, theta)

    pi = jax.nn.softmax(theta, axis=1)
    advantage = qpi(env, pi) - vpi(env, pi).reshape(-1, 1)
    a_hat = pi * jnp.abs(advantage)
    max_a_hat = a_hat.max(axis=1).flatten()
    non_zero_max_a_hat = max_a_hat[jnp.nonzero(max_a_hat)]

    # use very large step-sizes if \hat{S} is empty
    if non_zero_max_a_hat.size == 0:
        eta = 1e9
    else:
        eta = 1 / non_zero_max_a_hat.min()

    theta = theta + eta * grad

    return theta, eta


@jax.jit
def mdp_gnpg(key, theta, env):
    grad = grad_J(env, theta)

    # min_s \rho(s) = 1/S
    C_infty = 1 / env.S
    eta = (
        (1 - env.gamma)
        * env.gamma
        / (6 * (1 - env.gamma) * env.gamma + 4 * (C_infty - (1 - env.gamma)))
        * 1
        / jnp.sqrt(env.S)
    )

    theta = theta + eta / jnp.linalg.norm(grad) * grad
    return theta, eta / jnp.linalg.norm(grad)


def calc_qstar(env, num_iters=1000):
    q = jnp.zeros((env.P.shape[0], env.P.shape[1]))
    for i in range(num_iters):
        q_new = env.r + jnp.einsum("xay,y->xa", env.gamma * env.P, q.max(axis=1))
        q = q_new.copy()
    return q


def calc_pistar(env, num_iters=1000):
    q_star = calc_qstar(env, num_iters)
    pi_star = jnp.zeros_like(q_star)
    pi_star = pi_star.at[jnp.arange(env.P.shape[0]), q_star.argmax(axis=1)].set(1)
    return pi_star


def calc_vpi(pi, P, r, gamma):
    # p(s, s') = sum_{a} p(s' | s, a) pi(a | s)
    p_pi = jnp.einsum("xay,xa->xy", P, pi)
    # r(s) = sum_{a} r(s, a) * pi(a | s)
    r_pi = jnp.einsum("xa,xa->x", r, pi)
    # V  = R + gamma P^piV
    # (I - gamma P^pi) V = R
    # V = (I - gamma P^pi)^{-1} R
    v_pi = jnp.linalg.solve(jnp.eye(P.shape[0]) - gamma * p_pi, r_pi)

    return v_pi


def value_iteration(P, r, gamma, V, iters=1):
    n_states = P.shape[0]
    res = V
    for _ in range(iters):
        Vp = jnp.sum(P * res.reshape((1, 1, n_states)), axis=-1)
        res = jnp.max(r + gamma * Vp, axis=-1)
    return res


def get_optimal_V(P, r, gamma, iters=int(1e5)):
    return value_iteration(P, r, gamma, jnp.zeros(P.shape[0]), iters)


def policy_evaluation(pi, env):
    p_pi = jnp.einsum("xay,xa->xy", env.P, pi)
    r_pi = jnp.einsum("xa, xa -> x", env.r, pi)
    v_pi = jnp.linalg.solve(jnp.eye(env.S) - env.gamma * p_pi, r_pi)
    return v_pi


def policy_greedy(v, env):
    _pi_hat = jnp.argmax(env.r + env.gamma * jnp.einsum("xay, y->xa", env.P, v), axis=1)
    pi_hat = jnp.zeros((env.S, env.A))
    pi_hat = pi_hat.at[jnp.arange(env.S), _pi_hat].set(1)
    return pi_hat
