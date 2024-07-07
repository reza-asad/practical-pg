from flax.struct import dataclass
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

# NOTE: we assume that each update function is of the form
# >> def update(key, theta, reward, *args):
# >>     return theta, eta, action


def armijo_line_search(f, theta, eta_max, c, beta):
    gradient = jax.grad(f)(theta)

    def armijo_condition_not_satisfied(eta):
        return f(theta + eta * gradient) < f(theta) + c * eta * norm(gradient) ** 2

    def reduce_eta(eta):
        return eta * beta

    eta = jax.lax.while_loop(armijo_condition_not_satisfied, reduce_eta, eta_max)

    return eta

def transformed_armijo_line_search(f, theta, eta_max, c, beta, eps):

    gradient = jax.grad(f)(theta)
    theta_star = jnp.ones_like(theta) * -100
    theta_star = theta_star.at[0].set(100)
    f_star = f(theta_star)

    def armijo_condition_not_satisfied(eta):
        return jnp.log(f_star - f(theta + eta * gradient)) > (jnp.log(f_star - f(theta + eta * gradient)) - c * eta * norm(gradient) ** 2 / (f_star - f(theta)))

    def reduce_eta(eta):
        return eta * beta

    eta = jax.lax.while_loop(armijo_condition_not_satisfied, reduce_eta, eta_max)

    return eta

@jax.jit
def det_pg(key, theta, reward, eta):

    @jax.grad
    def df(theta):
        return jax.nn.softmax(theta) @ reward

    grad = df(theta)
    theta = theta + eta * grad

    return theta, eta


@jax.jit
def det_pg_ls(key, theta, reward, c, beta, eta_max):

    def f(theta):
        return jax.nn.softmax(theta) @ reward

    eta = armijo_line_search(f, theta, eta_max, c, beta)
    grad = jax.grad(f)(theta)
    theta = theta + eta * grad

    return theta, eta

@jax.jit
def det_pg_transformed_ls(key, theta, reward, c, beta, eta_max, eps):

    def f(theta):
        return jax.nn.softmax(theta) @ reward

    eta = transformed_armijo_line_search(f, theta, eta_max, c, beta, eps)
    grad = jax.grad(f)(theta)
    theta = theta + eta * grad

    return theta, eta

@jax.jit
def det_gnpg(key, theta, reward, eta=None):

    @jax.grad
    def df(theta):
        return jax.nn.softmax(theta) @ reward

    grad = df(theta)
    theta = theta + (1 / 6) / jnp.linalg.norm(grad) * grad

    return theta, eta


@jax.jit
def det_pg_entropy(key, theta, reward, eta, tau):

    @jax.grad
    def df(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        return jax.nn.softmax(theta) @ (reward - tau * jnp.log(pi))

    theta = theta + eta * df(theta)

    return theta, eta


@jax.jit
def det_pg_entropy_multistage(
    key, theta, reward, eta, tau, stage_length=None, p=None, B_1=None
):

    @jax.grad
    def df(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        return jax.nn.softmax(theta) @ (reward - tau * jnp.log(pi))

    theta = theta + eta * df(theta)

    return theta, eta


@jax.jit
def spg(key, theta, reward, eta):
    action = jax.random.categorical(key, theta)

    def stochastic_f(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        reward_hat = jax.nn.one_hot(action, len(reward)) / pi * reward
        return jax.nn.softmax(theta) @ reward_hat

    theta = theta + eta * jax.grad(stochastic_f)(theta)

    return theta, eta


@jax.jit
def spg_multistage(key, theta, reward, eta, stage_length=None):
    action = jax.random.categorical(key, theta)

    def stochastic_f(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        reward_hat = jax.nn.one_hot(action, len(reward)) / pi * reward
        return jax.nn.softmax(theta) @ reward_hat

    theta = theta + eta * jax.grad(stochastic_f)(theta)

    return theta, eta


@jax.jit
def spg_gradient_step_size(key, theta, reward, eta=None):
    action = jax.random.categorical(key, theta)

    def stochastic_f(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        reward_hat = jax.nn.one_hot(action, len(reward)) / pi * reward
        return jax.nn.softmax(theta) @ reward_hat

    @jax.grad
    def df(theta):
        pi = jax.nn.softmax(theta)
        return pi @ reward

    eta = 1 / 12 * norm(df(theta))

    theta = theta + eta * jax.grad(stochastic_f)(theta)

    return theta, eta


@jax.jit
def spg_entropy(key, theta, reward, eta, tau):
    action = jax.random.categorical(key, theta)

    def stochastic_f(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        reward_hat = jax.nn.one_hot(action, len(reward)) / pi * reward
        return jax.nn.softmax(theta) @ (reward_hat - tau * jnp.log(pi))

    theta = theta + eta * jax.grad(stochastic_f)(theta)

    return theta, eta


@jax.jit
def spg_entropy_multistage(key, theta, reward, eta, tau, stage_length=None):
    action = jax.random.categorical(key, theta)

    def stochastic_f(theta):
        pi = jax.lax.stop_gradient(jax.nn.softmax(theta))
        reward_hat = jax.nn.one_hot(action, len(reward)) / pi * reward
        return jax.nn.softmax(theta) @ (reward_hat - tau * jnp.log(pi))

    theta = theta + eta * jax.grad(stochastic_f)(theta)

    return theta, eta
