from flax.struct import dataclass
import jax
import jax.numpy as jnp


@dataclass
class Bandit:
    mean_r: jnp.array
    K: int = -1
    best_arm: int = 0
    instance_number: int = 0
    name: str = "Bandit"

    @classmethod
    def create(cls, mean_reward, instance_number, name, **kwargs):
        best_arm = jnp.argmax(mean_reward)
        return cls(
            mean_r=mean_reward,
            instance_number=instance_number,
            best_arm=best_arm,
            name=name,
            **kwargs,
        )

    def randomize(self, key):
        return self.mean_r


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


@dataclass
class FixedBandit(Bandit):
    name = "Fixed Bandit"


def make_bandit(
    env_key,
    instance_number,
    bandit_class,
    bandit_kwargs,
    environment_name,
    min_reward_gap=None,
    max_reward_gap=0.5,
):
    """
    Generates a bandit environment with K arms, with a random mean reward vector in [0, 1]^K

    To vary the difficutly of the bandit, the reward vectors are sampled in the range of `[0.5 - max_reward_gap/2, 0.5 + max_reward_gap/2]`

    To ensure that the bandit is not too easy/hard, we subtract the `min_reward_gap`, then clip the rewards to be in [0, 1]^K

    Note: if `min_reward_gap` is not None, then we cannot guarentee that the mean reward are in the range of `[0.5 - max_reward_gap/2, 0.5 + max_reward_gap/2]`
    """
    mean_reward = max_reward_gap * jax.random.uniform(
        env_key, (bandit_kwargs["K"],)
    ) + (0.5 - max_reward_gap / 2)
    mean_reward = mean_reward.sort()
    if min_reward_gap is not None:
        mean_reward = mean_reward.at[:-1].add(-min_reward_gap)
        mean_reward = jnp.clip(mean_reward, 0.0, 1.0)
    print(f"reward gap: {mean_reward[-1] - mean_reward[0]}")
    bandit = bandit_class.create(
        mean_reward, instance_number, environment_name, **bandit_kwargs
    )
    return bandit


def make_envs(env_def, num_runs, key):
    envs = []
    for instance_number in range(num_runs):
        key, env_key = jax.random.split(key)

        if env_def["Bandit"] is FixedBandit:
            env = env_def["Bandit"].create(
                env_def["bandit_kwargs"]["mean_r"],
                instance_number,
                env_def["environment_name"],
                K=env_def["bandit_kwargs"]["K"],
            )
        else:
            env = make_bandit(
                env_key,
                instance_number,
                env_def["Bandit"],
                env_def["bandit_kwargs"],
                env_def["environment_name"],
                min_reward_gap=env_def.get("min_reward_gap", None),
                max_reward_gap=env_def.get("max_reward_gap", 0.5),
            )

        envs.append(env)
    return envs
