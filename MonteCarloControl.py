
from MClearning.grid_world_multistate_index import *
from custom_functions import *
from MClearning.extract_velocity_field import velocity_field

def run_episode(g, policy, vStream_x, vStream_y, eps):

    s = g.start_state
    g.set_state(s)
    t, i, j = s
    a = stochastic_action(policy, s, g, eps)
    s_a_reward = [(s, a, 0)]
    count=0
    while True:
        count+=1
        # print(a, vStream_x[t, i, j], vStream_y[t, i, j])
        r = g.move(a, vStream_x[t, i, j], vStream_y[t, i, j])

        s = g.current_state()
        if g.is_terminal() or (not g.if_within_actionable_time()):
            s_a_reward.append((s, None, r))
            break
        else:
            a = stochastic_action(policy, s, g, eps)
            s_a_reward.append((s, a, r))

    state_action_return = []

    if not g.is_terminal():
        state_action_return = []

    else:
        G = 0
        first = True
        for s, a, r in reversed(s_a_reward):
            if first:
                first = False
            else:
                state_action_return.append((s, a, G))
            G = G + r
    state_action_return.reverse()

    return state_action_return


# set up grid

nt = 30
xs = np.arange(0, 10)
ys = xs
vStream_x = np.zeros((nt, len(ys), len(xs)))
vStream_y = np.zeros((nt, len(ys), len(xs)))
t_list = np.arange(0, nt)
# vStream_x[:, 6:12, :] = 2


# vStream_x, vStream_y, xs, ys, t_list = velocity_field()

X, Y = my_meshgrid(xs, ys)
print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

g = timeOpt_grid(t_list, xs, ys, (2, 1), (8, 8))

action_states = g.ac_state_space()

# initialise policy
policy = initialise_policy(g, action_states)

# initialise Q and N
Q, N = initialise_Q_N(action_states, g)

# Monte Carlo Control loop
delta_list = []
mcIters = 100000
delta = -1
count=0
stepsize=1000
numsteps=mcIters/stepsize
tk=0
for k in range(mcIters):

    eps=1-(tk/numsteps)

    s_a_return = run_episode(g, policy, vStream_x, vStream_y, eps)
    if len(s_a_return)==0:
        count+=1

    sl = al = None
    trajectory=[]
    # print("s_a_return", s_a_return)
    for s, a, G in s_a_return:
        # print("s, a , g=  ", s, a, G)
        trajectory.append(s)
        oldqsa = Q[s][a]
        N[s][a] += 1
        Q[s][a] = Q[s][a] + (1 / N[s][a]) * (G - Q[s][a])
        if delta != max(delta, np.abs(oldqsa - Q[s][a])):
            delta = max(delta, np.abs(oldqsa - Q[s][a]))
            sl=s
            al=a
    delta_list.append((sl,al,delta))

    if k % stepsize == 0:
        tk += 1
        print("Iteration", k, eps)
        print("trajectory length",len(trajectory))


    # print("----Qsa values----")
    # print("all states visited in the episode", k)
    # for s, a, G in s_a_return:
    #     if s[0]==0:
    #         print("state , action,  return")
    #         print(s, a, G)
    #         print(Q[s])
    #         print()

    # update policy
    for s, a, G  in s_a_return:
        newa , _ = max_dict(Q[s])
        policy[s] = newa

print("empty returns: ", count)
# print("delta list: ",delta_list)
# i=0
# for s,a,d in delta_list:
#     i+=1
#     plt.scatter(i,d)
# plt.show()

# write policy to file
writePolicytoFile(policy)

# plot trajectory
traj, (t, i, j), val = plot_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y,fname='MCapx')
traje, (te, ie, je), vale = plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y,fname='MC')

print("Value function", val)
print("time taken for optimal path")
print("with state apx " , t)
print("without state apx ", te)