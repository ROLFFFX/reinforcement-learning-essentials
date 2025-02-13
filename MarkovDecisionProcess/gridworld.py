import numpy as np
import pprint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


'''
    [
        0  1  2  3  4
        5  6  7  8  9
        10 11 12 13 14 
        15 16 17 18 19
        20 21 22 23 24
    ]
    
    north = 0
    east  = 1
    south = 2
    west  = 3

    Any action taken from 1 (except for a = 0, in which r = -1): reward = +10, takes agent to 21
    Any action taken from 3 (except for a = 0, in which r = -1): reward = +5,  takes agent to 13
    
    Slip probability of 0.2: no location update, reward = 0
    Any action that causes agent to go off the grid: no location update, reward = -1
    Other actions: reward = 0
    
'''

def gridworld(slip_prob=0.2):
    # slip_prob is the probability the agent slips.
    '''
    P = {
    s1: {a1: [(p(s'_1|s1,a1), s'_1, reward(s'_1,s1,a1)),
    (p(s'_2|s1,a1), s'_2, reward(s'_2,s1,a1)),
    ...
    ],
    a2: ...,
    ...
    },
    s2: ...,
    ...
    }
    '''
    def check_off_grid(state, action):
        # helper function to determine if (s, a) goes off grid. returns true if goes off grid.
        if (state in [1, 3]):
            return False
        if state in range(5) and action == 0: return True
        if state in range(20, 25) and action == 2: return True
        if state in [0, 5, 10, 15, 20] and action == 3: return True
        if state in [4, 9, 14, 19, 24] and action == 1: return True
        return False 
    # step 1: initialize P dict, first layer with 25 states {}
    # nested layer for each state s there's 4 possible actions,
    # each action corresponds to a list of tuples containing:
    # (prob. of next state is s1 given s1, a2; next state; reward)
    P = { s : {} for s in range(25)}
    actions = [0, 1, 2, 3]
    for state in P:
        for action in actions:
            P[state][action] = []
    # step 2: populate P given following cases:
    # case 1: slipped. prob of 0.2 for every state, s' unchanged, reward 0.
    # case 2: jackpot. special cases for s = 1, 3. s' changed to 21, 13, reward 10, 5.
    # case 3: unslipped. action taken, but goes off grid. s' unchanged, reward -1.
    # case 4: unslipped. action taken, does not go off grid. s' changed to corresponding direction, reward 0.
    for state in P:
        for action in actions:
            # case 3: off grid
            if check_off_grid(state, action):
                P[state][action].append((1, state, -1))
                continue
            # case 1: slipped
            if (state not in [1, 3]):
                P[state][action].append((slip_prob, state, 0))
            # ----------------rest actions has probability of 0.8
            # case 2: jackpot
            if state == 1:
                P[state][action].append((1, 21, 10))
                continue
            if state == 3:
                P[state][action].append((1, 13, 5))
                continue
            # case 4: normal transition
            match action:
                case 0:
                    P[state][action].append((1 - slip_prob, state - 5, 0))
                case 1:
                    P[state][action].append((1 - slip_prob, state + 1, 0))
                case 2:
                    P[state][action].append((1 - slip_prob, state + 5, 0))
                case 3:
                    P[state][action].append((1 - slip_prob, state - 1, 0))
    return P

def policy_eval(P, policy="uniform_policy", theta=0.0001, gamma=0.9):
    '''
    P: as returned by your gridworld(slip=0.2).
    policy: probability distribution over actions for each states.
    Default to uniform policy.
    theta: stopping condition.
    gamma: the discount factor.
    V: 5 by 5 numpy array where each entry is the value of the
    corresponding location. Initialize V with zeros.
    '''
    # for each entry s, v pi(s) is the expected return for follwing pi at state s and thereafter.
    # bellman equation: (prob. of action = a given policy) * (sum over all (s, r) prob. of (s', r)) * (reward + gamma * V[s'])
    V = np.zeros(25)
    policy_distribution = {}
    # generate policy distribution matrix for probability of each action. for uniform policy, all of them are 1 / a
    if isinstance(policy, str) and policy == "uniform_policy":
        policy_distribution = {s : {} for s in range(25)}
        for state in policy_distribution:
            policy_distribution[state].update({a : 0.25 for a in range(4)})
    else:
        policy_distribution = {s : {} for s in range(25)}
        for state in policy_distribution:
            a = 0
            for prob in policy[state]:
                policy_distribution[state].update({a : prob})
                a+=1
    # print("########")
    # pprint.pp(policy_distribution)
    # print("########")
    # main loop
    while True:
        delta = 0
        for state in range(25):
            v = 0 
            for a, action_probability in policy_distribution[state].items():
                for probability, next_state, reward in P[state][a]:
                    # ∑(a) (π(a|s) * ∑(s', r) (p(s', r | s, a) * [r + γ * Vk(s')])) 
                    v += action_probability * probability * (reward + gamma * V[next_state])
                    # print(v)
            delta = max(delta, abs(v - V[state]))
            V[state] = v
        if delta < theta:
            break
    return np.round(V.reshape((5,5)), 2)
    
def visualize_policy(policy):
    policy_mat = []
    directions = {0: "n", 1: "e", 2: "s", 3: "w"}
    c = 0
    for s in policy:
        optimal_action = []
        for i in range(4):
            if s[i] != 0:
                optimal_action.append(directions[i])
              
        policy_mat.append(optimal_action)
        
        print("".join(optimal_action), end=" ")
        c += 1
        if c == 5:
            print("\n")
            c = 0
    return policy_mat
    
def policy_iter(P, theta=0.0001, gamma=0.9):
    '''
    policy: 25 by 4 numpy array where each row is a probability
    distribution over moves for a state. If it is
    deterministic, then the probability will be a one hot vector.
    
    If there is a tie between two actions, break the tie with
    equal probabilities.
    
    Initialize the policy with the uniformly random policy
    described in Part (b).
    '''
    # step 1: initialization
    # policy would be initialized as the uniform policy.
    # V would be initialized during step 2, policy evaluation.
    policy = np.full((25, 4), 0.25)
    # V = np.zeros(25)
    while (True):
        # step 2: policy eval
        V = policy_eval(P, policy=policy, theta=theta, gamma=gamma)
        V = V.reshape(25, 1)
        # step 3: policy improvement
        policy_stable = True
        for state in range(25):
            # old_action <- π(s)
            old_actions = policy[state].copy()
            action_values = np.zeros(4)
            for a in range(4):
            # for each action, get value for selecting that action
                for probability, next_state, reward in P[state][a]:
                    action_values[a] += probability * (reward + gamma * V[next_state])
            
            # get index of best action and record it
            best_action_index = None
            max_value = float('-inf')
            for i in range(len(action_values)):
                if action_values[i] > max_value:
                    max_value = action_values[i]
                    best_action_index = i
            
            # assign new probability to best action (1), creating new policy
            new_action_probs = np.zeros(4)
            new_action_probs[best_action_index] = 1.0
            
            best_actions = np.where(action_values == action_values[best_action_index])[0]
            new_action_probs[best_actions] = 1.0 / len(best_actions)
            policy[state] = new_action_probs
            
            if not np.array_equal(old_actions, new_action_probs):
                policy_stable = False
                
        if policy_stable:
            break
    
    # process V and policy
    V = np.round(V.reshape((5,5)), 2)
    policy_mat = visualize_policy(policy)
    return V, policy_mat
    
def value_iter(P, theta=0.0001, gamma=0.9):
    # value iteratino: a truncated version of policy iteration
    V = np.zeros((25, 1))
    policy = np.zeros((25, 4))
    while (True):
        delta = 0
        for state in range(25):
            v = V[state].copy()
            # V(s) <- max(a) ∑(s', r) * p(s', r | s, a) * [r + gamma * V(s')]
            action_values = np.zeros(4)
            # for each action, get value for selecting that action
            for action in range(4):
                for probability, next_state, reward in P[state][action]:
                    action_values[action] += probability * (reward + gamma * V[next_state])
            
            # set V[s] to max action value
            V[state] = np.max(action_values)
            delta = max(delta, abs(v - V[state]))
                
        if delta < theta:
            break
        
    # get optimal policy: π(s) = argmax(a) ∑(s', r) * p(s' , r | s, a) * [r + gamma * V(s')]
    for state in range(25):
        action_values = np.zeros(4)
        for action in range(4):
            for probability, next_state, reward in P[state][action]:
                action_values[action] += probability * (reward + gamma * V[next_state])
        
        # Assign the best action to the policy
        best_action = np.argmax(action_values)
        policy[state] = np.eye(4)[best_action]

    # process V and policy
    V = np.round(V.reshape((5,5)), 2)
    policy_mat = visualize_policy(policy)
    return V, policy_mat

def gridworld_ep(slip_prob=0.2):
    # Whenever the agent moves to A’ or B’, the episode ends.
    # conceptually, A' and B' becomes the terminal state.
    
    # when agent reaches A' and B', the transition probability 
    # of going out of A' and B' would be 0.
    def check_off_grid(state, action):
        # helper function to determine if (s, a) goes off grid. returns true if goes off grid.
        if (state in [1, 3]):
            return False
        if state in range(5) and action == 0: return True
        if state in range(20, 25) and action == 2: return True
        if state in [0, 5, 10, 15, 20] and action == 3: return True
        if state in [4, 9, 14, 19, 24] and action == 1: return True
        return False 
    # step 1: initialize P dict, first layer with 25 states {}
    # nested layer for each state s there's 4 possible actions,
    # each action corresponds to a list of tuples containing:
    # (prob. of next state is s1 given s1, a2; next state; reward)
    P = { s : {} for s in range(25)}
    actions = [0, 1, 2, 3]
    for state in P:
        for action in actions:
            P[state][action] = []
    # step 2: populate P given following cases:
    # case 1: slipped. prob of 0.2 for every state, s' unchanged, reward 0.
    # case 2: jackpot. special cases for s = 1, 3. s' changed to 21, 13, reward 10, 5.
    # case 3: unslipped. action taken, but goes off grid. s' unchanged, reward -1.
    # case 4: unslipped. action taken, does not go off grid. s' changed to corresponding direction, reward 0.
    
    
    for state in P:
        # added episodic end: 
        #  21: {0: [(1, 21, 0)], 1: [(1, 21, 0)], 2: [(1, 21, 0)], 3: [(1, 21, 0)]},
        #  13: {0: [(1, 13, 0)], 1: [(1, 13, 0)], 2: [(1, 13, 0)], 3: [(1, 13, 0)]},
        if state in [21, 13]:
            for action in actions:
                P[state][action].append((1, state, 0))
            continue
        for action in actions:
            # case 3: off grid
            if check_off_grid(state, action):
                P[state][action].append((1, state, -1))
                continue
            # case 1: slipped
            if (state not in [1, 3]):
                P[state][action].append((slip_prob, state, 0))
            # ----------------rest actions has probability of 0.8
            # case 2: jackpot
            if state == 1:
                P[state][action].append((1, 21, 10))
                continue
            if state == 3:
                P[state][action].append((1, 13, 5))
                continue
            # case 4: normal transition
            match action:
                case 0:
                    P[state][action].append((1 - slip_prob, state - 5, 0))
                case 1:
                    P[state][action].append((1 - slip_prob, state + 1, 0))
                case 2:
                    P[state][action].append((1 - slip_prob, state + 5, 0))
                case 3:
                    P[state][action].append((1 - slip_prob, state - 1, 0))
    return P

def grid_search():
    P_ep = gridworld_ep(slip_prob=0.2)
    V_ep, optimal_policy = value_iter(P = P_ep, gamma = 0.9)
    gamma_value = 0.9
    decrease_value = 0.001
    while (gamma_value > 0.8):
        gamma_value -= decrease_value
        V, new_policy = value_iter(P = P_ep, gamma = gamma_value)
        print("gamma value of f{gamma_value}:")
        pprint.pp(V)
        print("##############################")
        if (new_policy != optimal_policy):
            for i in range(25):
                if (new_policy[i] != optimal_policy[i]):
                    print(f"policy at state {i} is different.")
            break
    return gamma_value

if __name__ == "__main__":
    P = gridworld(slip_prob=0.2)
    # # print(P)
    # # pprint.pp(P)
    # V = policy_eval(P)
    # pprint.pp(V)
    # print("##############from policy iteration#################")
    V, policy = policy_iter(P)
    print(policy)
    # print(V)
    # # print(V)
    # print("##############from value iteration#################")
    # V, policy = value_iter(P)
    # # print(V)
    # P_ep = gridworld_ep(slip_prob=0.2)
    # # pprint.pp(P_ep)
    # print("############## episodic task #################")
    # # question (e)
    # V_ep, policy_ep = value_iter(P = P_ep, gamma=0.9)
    # print(V_ep)
    # critical_gamma = grid_search()
    # print(critical_gamma)
    