from World import World
import numpy as np
import pandas as pd
import copy


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


def max_action(transition_models, rewards, gamma, s, V, actions, terminal_ind):

    maxs = {key: 0 for key in actions}
    max_a = ""
    action_map = {k: v for k, v in zip(actions, [1, 3, 2, 4])}
    for action in actions:
        if s not in terminal_ind:
            maxs[action] += rewards[s - 1] + gamma * np.dot(transition_models[action].loc[s, :].values, V)
        else:
            maxs[action] = rewards[s - 1]
    maxi = -10 ** 10
    for key in maxs:
        if maxs[key] > maxi:
            max_a = key
            maxi = maxs[key]
    return maxi, action_map[max_a]


def part_b(world, transition_models, rewards, gamma=1, theta=10 ** -4):

    nstates = world.get_nstates()
    terminal_ind = world.get_stateterminals()
    V = np.zeros((nstates, ))
    P = np.zeros((nstates, 1))
    actions = ["N", "S", "E", "W"]
    delta = theta + 1
    while delta > theta:
        delta = 0
        v = copy.deepcopy(V)
        for s in range(1, nstates + 1):
            V[s - 1], P[s - 1] = max_action(transition_models, rewards, gamma, s, v, actions, terminal_ind)
            delta = max(delta, np.abs(v[s - 1] - V[s - 1]))
    return V, P


if __name__ == "__main__":

    world = World()
    # world.plot()
    # world.plot_value([np.random.random() for i in range(12)])
    # world.plot_policy(np.random.randint(1, world.nActions,(world.nStates, 1)))
    # part a
    transition_models, rewards = part_a(world)
    # part b
    V, P = part_b(world, transition_models, rewards)
    world.plot_value(V)
    world.plot_policy(P)

