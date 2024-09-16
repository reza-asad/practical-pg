from flax.struct import dataclass
import jax
import jax.numpy as jnp


@dataclass
class Bandit:
    mean_r: jnp.array
    K: int
    best_arm: int = 0
    name: str = "Bandit"
    d: int = None
    features: jnp.array = None
    instance_number: int = 0

    @classmethod
    def create(cls, mu, instance_number, env_name, **kwargs):
        best_arm = jnp.argmax(mu)
        return cls(mean_r=mu, instance_number=instance_number, best_arm=best_arm, name=env_name, **kwargs)

    def randomize(self, key):
        return self.mean_r

    @staticmethod
    def regret(arm, best_arm, reward):
        return reward[best_arm] - reward[arm]


@dataclass
class BerBandit(Bandit):
    def randomize(self, key):
        rt = jax.random.uniform(key, shape=(self.K,))
        return jnp.array(rt < self.mean_r).astype(jnp.float32)


@dataclass
class GaussBandit(Bandit):
    sigma: float = 0.1
    name = "Gaussian Bandit"

    def randomize(self, key):
        rt = jax.random.normal(key, shape=(self.K,))
        return jnp.clip(self.mean_r + self.sigma * rt, 0, 1)


@dataclass
class BetaBandit(Bandit):
    a_plus_b: float = 0
    name = "Beta Bandit"

    def randomize(self, key):
        return jax.random.beta(
            key, self.a_plus_b * self.mean_r, self.a_plus_b * (1 - self.mean_r)
        )


def randomize_beta(key, mean_r, a_plus_b):
    return jax.random.beta(key, a_plus_b * mean_r, a_plus_b * (1 - mean_r))


def randomize_bernoulli(key, mean_r):
    return jnp.array(jax.random.uniform(key, shape=(len(mean_r),)) < mean_r).astype(
        jnp.float32
    )


def randomize_gaussian(key, mean_r, sigma=0.1):
    return jnp.clip(mean_r + sigma * jax.random.normal(key, shape=(len(mean_r),)), 0, 1)