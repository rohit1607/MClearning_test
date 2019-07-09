#from TimeOpt_Det_VI_multist.extract_velocity_field import *
import matplotlib.pyplot as plt
from MClearning.grid_world_multistate_index import *
from custom_functions import *
import numpy as np

nt = 2
xs = np.arange(0, 4)
ys = xs
vStream_x = np.zeros((nt, len(ys), len(xs)))
vStream_y = np.zeros((nt, len(ys), len(xs)))
t_list = np.arange(0, nt)

vStream_x[:, 2:3, :] = 1

X, Y = my_meshgrid(xs, ys)
print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

# set up grid
# start and endpos are tuples of !!indices!!
g = timeOpt_grid(t_list, xs, ys, (1, 1), (1, 3))

policy={}
#f=open('pol_50x50x50x8.txt','r')
f=open('output.txt','r')
if f.mode == 'r':
    policy = eval(f.read())
f.close()


traj,(t,i,j),val=plot_trajectory(g, policy, xs, ys, X, Y, vStream_x,vStream_y)
#traje,(te,ie,je),vale=plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x,vStream_y)

print("Time-steps requirred for optimal path")
print("state-trajectory", t)
# print("position-trajectory", te)

#print("Value function", val)

