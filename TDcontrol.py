from MClearning.grid_world_multistate_index import *
from custom_functions import *




def run_TD0_episode(g, vStream_x, vStream_y, eps, Q, N,alpha=0.1):
    s = g.start_state
    g.set_state(s)
    t, i, j = s
    # print(" start state", s)
    a = max_dict(Q[s])[0]
    # print("1. max a", a)
    a = random_action(a, s, g, eps)
    # print("2. randomized a",a)
    count=0
    while (not g.is_terminal())  and  g.if_within_TD_actionable_time():
        count+=1
        # print("count=", count)
        r = g.move(a, vStream_x[t,i,j], vStream_y[t,i,j])
        s2 = g.current_state()
        # print(" s2", s2)
        if (not g.is_terminal()) and g.if_within_actionable_time():
            a2 = max_dict(Q[s2])[0]
            # print("i. max a", a2)
            a2 = random_action(a2, s2, g, eps)
            # print("ii. randomized a", a2)

            N[s][a] += 0.005
            valpha=alpha/N[s][a]

            Q[s][a]=Q[s][a] + valpha*(r + Q[s2][a2] - Q[s][a])

            s=s2
            a=a2

    return Q, N



nt = 40
xs = np.arange(0, 20)
ys = xs
vStream_x = np.zeros((nt, len(ys), len(xs)))
vStream_y = np.zeros((nt, len(ys), len(xs)))
t_list = np.arange(0, nt)
# vStream_x[:, 4:7, :] = 1


# vStream_x, vStream_y, xs, ys, t_list = velocity_field()
X, Y = my_meshgrid(xs, ys)
print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

g = timeOpt_grid(t_list, xs, ys, (2, 1), (12, 12))

action_states = g.ac_state_space()
for s in action_states:
    print(s)
# initialise policy
policy = initialise_policy(g, action_states)

# initialise Q and N
Q, N = initialise_Q_N1(action_states, g)


MaxIters=150000
stepsize=1000
numsteps=MaxIters/stepsize
tk=0
#convergence loop
for k in range(MaxIters):

    eps = 1 - (tk / numsteps)

    Q,N= run_TD0_episode(g, vStream_x, vStream_y, eps, Q, N, alpha=0.1)

    if k % stepsize == 0:
        tk += 1
        print("iter, eps", k, eps)



#Compute Policy
for s in action_states:
    policy[s]=max_dict(Q[s])[0]

writePolicytoFile(policy)

#plot trajectory
traj, (t, i, j), val = plot_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y,fname='TD0apx')
traje, (te, ie, je), vale = plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y,fname='TD0')

print("Value function", val)
print("time taken for optimal path")
print("with state apx " , t)
print("without state apx ", te)