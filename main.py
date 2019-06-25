from World import World
import numpy as np
import pandas as pd


def part_a(world, p=0.8):

    nstates = world.get_nstates()
    nrows = world.get_nrows()
    obsacle_index = world.get_stateobstacles()
    terminal_index = world.get_stateterminals()
    bad_index = obsacle_index + terminal_index
    rewards = np.array([-0.04] * 4 + [0] + [-0.04] * 4 + [1, -1] + [-0.04])
    actions = ["N", "S", "E", "W"]
    transition_models = {}
    for action in actions:
        transition_model = np.zeros((nstates, nstates))
        for i in range(1, nstates + 1):
            if i not in bad_index:
                if action == "N":
                    if i + nrows <= nstates and (i + nrows) not in obsacle_index:
                        transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if 0 < i - nrows <= nstates and (i - nrows) not in obsacle_index:
                        transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if (i - 1) % nrows > 0 and (i - 1) not in obsacle_index:
                        transition_model[i - 1][i - 1 - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                if action == "S":
                    if i + nrows <= nstates and (i + nrows) not in obsacle_index:
                        transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if 0 < i - nrows <= nstates and (i - nrows) not in obsacle_index:
                        transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if 0 < i % nrows and (i + 1) not in obsacle_index and (i + 1) <= nstates:
                        transition_model[i - 1][i + 1 - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                if action == "E":
                    if i + nrows <= nstates and (i + nrows) not in obsacle_index:
                        transition_model[i - 1][i + nrows - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                    if 0 < i % nrows and (i + 1) not in obsacle_index and (i + 1) <= nstates:
                        transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if (i - 1) % nrows > 0 and (i - 1) not in obsacle_index:
                        transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                if action == "W":
                    if 0 < i - nrows <= nstates and (i - nrows) not in obsacle_index:
                        transition_model[i - 1][i - nrows - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                    if 0 < i % nrows and (i + 1) not in obsacle_index and (i + 1) <= nstates:
                        transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if (i - 1) % nrows > 0 and (i - 1) not in obsacle_index:
                        transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
            elif i in terminal_index:
                transition_model[i - 1][i - 1] = 1
        transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1), columns=range(1, nstates + 1))

    return transition_models, rewards


def max_action(transition_models, rewards, gamma, s, V, actions):

    maxs = {key: 0 for key in actions}
    max_a = ""
    for action in actions:
        for s_p, p in enumerate(transition_models[action].loc[1, :]):
            maxs[action] += rewards[s - 1] + gamma * p * V[s_p][0]
    maxi = -10 ** 10
    for key in maxs:
        if maxs[key] > maxi:
            max_a = key
            maxi = maxs[key]
    return maxi, max_a


def part_b(world, transition_models, rewards, gamma=1, theta=10 ** -4):

    nstates = world.get_nstates()
    V = np.zeros((nstates, 1))
    P = {}
    actions = ["N", "S", "E", "W"]
    delta = theta + 1
    while delta > theta:
        delta = 0
        for s in range(1, nstates + 1):
            v = V[s - 1][0]
            V[s - 1], P[s] = max_action(transition_models, rewards, gamma, s, V, actions)
            delta = max(delta, np.abs(v - V[s - 1]))
    return V, P


if __name__ == "__main__":

    world = World()
    # world.plot()
    # world.plot_value([np.random.random() for i in range(12)])
    # world.plot_policy(np.random.randint(1, world.nActions,(world.nStates, 1)))
    # part a
    transition_models, rewards = part_a(world)
    part_b(world, transition_models, rewards)

