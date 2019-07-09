import numpy as np
import random
from MClearning.grid_world_multistate_index import *
import math
# NOTE: find optimal policy and value function

def random_action(g, s, a, eps=0.1):
    # choose given a with probability 1 - eps + eps/4
    # choose some other a' != a with probability eps/4
    print("in random action",eps)
    return a
    # p = np.random.random()
    # n=len(g.actions[s])
    # # print("n,p ",n,p)
    # if p < (1 - eps + (eps/n)):
    #     return a
    # else:
    #     i=random.randint(0,n-1)
    #     return g.actions[s][i]


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


# returns a list of states and corresponding returns
def play_game(g, policy, vStream_x, vStream_y,epsil):
    print("epsil in playgame ", epsil)
    s=g.start_state
    t,i,j=s
    g.set_state(s)
    a = random_action(g, s, policy[s],eps=epsil)


    # print("s,rand a", s,a)
    # be aware of the timing
    # each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s, a, 0)]
    while True:
        r = g.move( a, vStream_x[t,i,j], vStream_y[t,i,j] )
        s = g.current_state()
        t, i, j = s
        if g.is_terminal() or (not g.if_within_actionable_time()):
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(g, s, policy[s],eps=epsil) # the next state is stochastic
            states_actions_rewards.append((s, a, r))
        # print("s,rand a, r", s, a, r )

    if not g.is_terminal():
        return []

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
    # the value of the terminal state is 0 by definition
    # we should ignore the first state we encounter
    # and ignore the last G, which is meaningless since it doesn't correspond to any move
    #     print("sarg= ",s,a,r,G)

        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))

        G = r + G

    states_actions_returns.reverse() # we want it to be in order of state visited
    # print("saglist",states_actions_returns)
    return states_actions_returns


if __name__ == '__main__':

    threshold = 1e-3

    # set up stream velocity
    # vStream_x, vStream_y, xs, ys, t_list = velocity_field()
    nt = 2
    xs = np.arange(0, 10)
    ys = xs
    vStream_x = np.zeros((nt, len(ys), len(xs)))
    vStream_y = np.zeros((nt, len(ys), len(xs)))
    t_list = np.arange(0, nt)

    # vStream_x[:, 2:3, :] = 1

    X, Y = my_meshgrid(xs, ys)
    print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

    # set up grid
    # start and endpos are tuples of !!indices!!
    g = timeOpt_grid(t_list, xs, ys, (9, 7), (1, 2))


    # initialize a random policy
    policy = {}

    action_states=g.ac_state_space()

    # for s in action_states:
    #     print(s)
    #     i=random.randint(0,len(g.actions[s])-1)
    #     policy[s] = g.actions[s][i]

    Pi=math.pi
    for s in action_states:
        t,i,j=s
        print("start",s)
        i2,j2=g.endpos
        print("i2,j2",i2,j2)
        if j2==j:
            if i2>i:
                policy[s]=(1,1.5*Pi)
            elif i2<i:
                policy[s] = (1, 0.5 * Pi)
        elif j2>j:
            if i2>i:
                policy[s]=(1,1.75*Pi)
            elif i2<i:
                policy[s] = (1, 0.25 * Pi)
            elif i2==i:
                policy[s] = (1, 0)
        elif j2<j:
            if i2 > i:
                policy[s] = (1, 1.25 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.75 * Pi)
            elif i2 == i:
                policy[s] = (1, Pi)
        if i==9 or j==9:
            print("state, action ",s, policy[s])
    # actioncount=0
    #test random action
    # for i in range(1000):
    #     s=(0,1,1)
    #     a=(1,0)
    #     newa=random_action(g,s,a,0.1)
    #     if newa == a:
    #         actioncount += 1
    # print("a=policy: ",actioncount)

    # initialize Q(s,a) and returns
    Q = {}
    N = {}
    returns = {} # dictionary of state -> list of returns we've received
    for s in action_states:
        Q[s] = {}
        N[s] ={}
        for a in g.actions[s]:
            Q[s][a] = 0
            N[s][a] =0
            returns[(s,a)] = []


    # Repeat until convergence
    count = 0
    deltas = []
    iters=1

    for t in range(iters):
        if t % 1000 == 0:
            print(t)

        if t<0.9*iters:
            eps=0.5
        else:
            eps=1/(t+1)
        # generate an episode using pi
        biggest_change = 0
        print("eps in loop ",eps)
        states_actions_returns = play_game(g, policy, vStream_x, vStream_y,eps)

        if len(states_actions_returns)!=0:
            # calculate Q(s,a)
            # seen_state_action_pairs = set()
            for s, a, G in states_actions_returns:
                # check if we have already seen s
                # called "first-visit" MC policy evaluation
                sa = (s, a)
                # if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                # returns[sa].append(G)
                N[s][a] = N[s][a] + 1
                Q[s][a] = Q[s][a] + (1/N[s][a])*(G - Q[s][a])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                # seen_state_action_pairs.add(sa)
            deltas.append(biggest_change)

            # calculate new policy pi(s) = argmax[a]{ Q(s,a) }
            for s in action_states:
                # print("state, Q(s,a)", Q[s])
                a, _ = max_dict(Q[s])
                # print("best action")
                policy[s] = a
        else:
            count+=1

    plt.plot(deltas)
    plt.show()
    print("#empty lists returned= ",count)
    # find the optimal state-value function
    # V(s) = max[a]{ Q(s,a) }
    V = {}
    for s in policy.keys():
        V[s] = max_dict(Q[s])[1]

    print(V)

    outputfile = open('output.txt', 'w')
    print(policy, file=outputfile)
    outputfile.close()


    traj,(t,i,j),val=plot_trajectory(g, policy, xs, ys, X, Y, vStream_x,vStream_y)
    # traj, (t, i, j), val = plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y)

    # print("MDP solve time: ", end - start)
    print("time taken across optimal path", t * g.dt, val)
    print("Value function", val)
