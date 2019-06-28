from World import World
import numpy as np
import pandas as pd
import copy


def construct_p(world, p=0.8, step=-0.04):

    nstates = world.get_nstates()
    nrows = world.get_nrows()
    obsacle_index = world.get_stateobstacles()
    terminal_index = world.get_stateterminals()
    bad_index = obsacle_index + terminal_index
    rewards = np.array([step] * 4 + [0] + [step] * 4 + [1, -1] + [step])
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


def value_iteration(world, transition_models, rewards, gamma=1.0, theta=10 ** -4):

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


def policy_iter(policy, world, transition_models, rewards, gamma=0.9, theta=10 ** -4):

    nstates = world.get_nstates()
    terminal_ind = world.get_stateterminals()
    # Initiate value function to zeros
    V = np.zeros((nstates,))
    a = ["N", "S", "E", "W"]
    while True:
        delta = 0
        # For each state, perform a backup
        for s in range(nstates):
            v = 0
            # Look at the policy actions and their probabilities
            for action, action_prob in enumerate(policy[s]):
                action = a[action]
                # For each action, calculate total gain
                if s not in terminal_ind:
                    v += rewards[s - 1] + action_prob * gamma * np.dot(transition_models[action].loc[s, :].values, V)
                else:
                    v = rewards[s - 1]
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
            print (V[s])
        # Stop evaluating once the value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


#  Helper function to calculate the value for all action in a given state
def lookfoword(s, V, transition_models, rewards, gamma = 0.9):

    nActions = world.get_nstates()
    terminal_ind = world.get_stateterminals()
    A = np.zeros(nActions)
    a = ["N", "S", "E", "W"]
    for i, action in enumerate(nActions):
        action = a[action]
        if s not in terminal_ind:
            A[i] += rewards[s - 1] + gamma * np.dot(transition_models[a].loc[s, :].values, V)
        else:
            A[i] = rewards[s - 1]
    return A


def policy_improvement(world, transition_models, rewards, gamma= 0.9):

    nstates = world.get_nstates()
    nActions = world.get_nactions()


    # Start with a uniform policy
    policy = np.ones((nstates, nActions)) / nActions

    while True:
        # Evaluate the current policy
        V = policy_iter(policy, world, transition_models, rewards, gamma)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        for s in range(nstates):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            action_values = lookfoword(s, V, transition_models, rewards, gamma)
            best_action = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_action:
                policy_stable = False
            policy[s] = np.eye(nActions)[best_action]

        if policy_stable:
            return V, policy


if __name__ == "__main__":

    world = World()
    # world.plot()
    # world.plot_value([np.random.random() for i in range(12)])
    # world.plot_policy(np.random.randint(1, world.nActions,(world.nStates, 1)))
    # part a
    # transition_models, rewards = construct_p(world)
    # part b
    transition_models, rewards = construct_p(world)
    V, P = value_iteration(world, transition_models, rewards)
    world.plot_value(V)
    world.plot_policy(P)
    # part c
    transition_models, rewards = construct_p(world)
    V, P = value_iteration(world, transition_models, rewards, gamma=0.9)
    world.plot_value(V)
    world.plot_policy(P)
    # part d
    transition_models, rewards = construct_p(world, step=-0.02)
    V, P = value_iteration(world, transition_models, rewards)
    world.plot_value(V)
    world.plot_policy(P)
    # part e
    # transition_models, rewards = construct_p(world)
    # V, P = policy_improvement(world, transition_models, rewards, gamma=0.9)
    # world.plot_value(V)
    # world.plot_policy(P)
