import jax.numpy as jnp
import numpy as np
import copy
from flax.struct import dataclass
import jax
import jax.numpy as jnp


def rescale_reward(r):
    return (r - r.min()) / (r.max() - r.min())


# src: https://github.com/svmgrg/fma-pg/blob/main/current_code/environments.py
# ======================================================================
# Simplified CliffWorld
# ----------------------------------------------------------------------
# -------------------         4 is the goal state
# | 4 | 9 | 14 | 19 |         20 is terminal state reached only via the state 4
# -------------------         3, 2, 1 are chasms
# | 3 | 8 | 13 | 18 |         0 is the start state
# -------------------
# | 2 | 7 | 12 | 17 |         all transitions are deterministic
# -------------------         Actions: 0=down, 1=up, 2=left, 3=right
# | 1 | 6 | 11 | 16 |
# ------------------------    rewards are all zeros except at chasms (-100)
# | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
# ------------------------
# ----------------------------------------------------------------------

# environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
P = np.zeros((21, 4, 21))
for state_idx in range(21):
    for action_idx in range(4):
        if state_idx in [1, 2, 3]:  # chasms: reset to start state 0
            new_state_idx = 0
        elif state_idx == 4:  # goal state: agent always goes to 20
            new_state_idx = 20
        elif state_idx == 20:  # terminal state
            new_state_idx = 20
        else:  # move according to the deterministic dynamics
            x_new = x_old = state_idx // 5
            y_new = y_old = state_idx % 5
            if action_idx == 0:  # Down
                y_new = np.clip(y_old - 1, 0, 4)
            elif action_idx == 1:  # Up
                y_new = np.clip(y_old + 1, 0, 4)
            elif action_idx == 2:  # Left
                x_new = np.clip(x_old - 1, 0, 3)
            elif action_idx == 3:  # Right
                x_new = np.clip(x_old + 1, 0, 3)
            new_state_idx = 5 * x_new + y_new

        P[state_idx, action_idx, new_state_idx] = 1

r = np.zeros((21, 4))
r[1, :] = r[2, :] = r[3, :] = -100  # negative reward for falling into chasms
r[4, :] = +1  # positive reward for finding the goal terminal state

rho = np.zeros(21)
rho[0] = 1

terminal_states = [20]

P_CliffWorld = copy.deepcopy(P)
r_CliffWorld = rescale_reward(copy.deepcopy(r))
rho_CliffWorld = copy.deepcopy(rho)
terminal_states_CliffWorld = copy.deepcopy(terminal_states)
# ======================================================================

# ======================================================================
# Deep Sea Treasure
# ----------------------------------------------------------------------
# ------------------------    20 is the goal state
# | 4 | 9 | 14 | 19 | 24 |    0, 5, 10, 15, 20 are terminal states
# ------------------------    4 is the start state
# | 3 | 8 | 13 | 18 | 23 |
# ------------------------
# | 2 | 7 | 12 | 17 | 22 |    all transitions are deterministic
# ------------------------    Actions: 0=left, 1=right
# | 1 | 6 | 11 | 16 | 21 |
# ------------------------    rewards are all -0.01/5 except at state 20
# | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
# ------------------------
# ----------------------------------------------------------------------
terminal_states = [0, 5, 10, 15, 20]

# environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
P = np.zeros((25, 2, 25))
for state_idx in range(25):
    for action_idx in range(2):
        if state_idx in terminal_states:  # terminal states
            new_state_idx = state_idx
        else:  # move according to the deterministic dynamics
            x_new = x_old = state_idx // 5
            y_new = y_old = state_idx % 5
            if action_idx == 0:  # left
                x_new = np.clip(x_old - 1, 0, 4)
            elif action_idx == 1:  # right
                x_new = np.clip(x_old + 1, 0, 4)
            y_new = y_old - 1
            new_state_idx = 5 * x_new + y_new

        P[state_idx, action_idx, new_state_idx] = 1

r = (-0.01 / 5) * np.ones((25, 2))
r[16, 1] = r[21, 1] = +1  # positive reward for finding the goal terminal state
for s in terminal_states:
    r[s, :] = 0

rho = np.zeros(25)
rho[4] = 1

P_DeepSeaTreasure = copy.deepcopy(P)
r_DeepSeaTreasure = rescale_reward(copy.deepcopy(r))
rho_DeepSeaTreasure = copy.deepcopy(rho)
terminal_states_DeepSeaTreasure = copy.deepcopy(terminal_states)


# ======================================================================

# ======================================================================
# Penalized Gradient
def build_chain_mdp(H, n_states, n_actions, right_penalty=0):
    n_states = H + 2
    P = np.zeros((n_states, n_actions, n_states))  # (s, a, s')
    r = np.zeros((n_states, n_actions))

    # taking action 0 at absorbing state gives you reward 1
    r[-1, 0] = 1.0

    # optional penalty of going right at every state but the last
    r[:-1, 0] = right_penalty

    # populate the transition matrix
    # forward actions
    for s in range(n_states - 1):
        P[s, 0, s + 1] = 1.0
    P[n_states - 1, :, n_states - 1] = (
        1.0  # irrespective of the action, you end up in the last state forever
    )

    # backward actions take you backwards
    for s in range(1, n_states - 1):
        P[s, 1:, s - 1] = 1.0
    P[0, 1:, 0] = 1.0
    return P, r

H = 20
gamma_FlatGrad = H / (H + 1)
P_FlatGrad, r_FlatGrad = build_chain_mdp(H, n_states=20 + 2, n_actions=4)
r_FlatGrad = rescale_reward(r_FlatGrad)

@dataclass
class UniformTabularEnv:
    P: jnp.array
    r: jnp.array
    rho: jnp.array
    S: int
    A: int
    gamma: float

    @classmethod
    def create(cls, P, r, gamma):
        P = P.astype(jnp.int32)
        S = P.shape[0]
        A = P.shape[1]
        rho = jnp.ones(S) / S
        return cls(P, r, rho, S, A, gamma)

        
CliffWorld = UniformTabularEnv.create(
    jnp.array(P_CliffWorld),
    jnp.array(r_CliffWorld),
    0.9,
)

DeepSeaTreasure = UniformTabularEnv.create(
    jnp.array(P_DeepSeaTreasure),
    jnp.array(r_DeepSeaTreasure),
    0.9,
)

FlatGrad = UniformTabularEnv.create(
    jnp.array(P_FlatGrad),
    jnp.array(r_FlatGrad),
    gamma_FlatGrad
)
    