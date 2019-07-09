import numpy as np
import itertools
import math
from custom_functions import *


class Grid:
    def __init__(self, tlist, xs, ys, start, end):
        self.nj = len(xs)
        self.ni = len(ys)
        self.nt = len(tlist)
        print("shapes=", len(xs), len(ys), len(tlist))
        self.dj = np.abs(xs[1] - xs[0])
        self.di = np.abs(ys[1] - ys[0])
        self.dt = tlist[1] - tlist[0]
        print("diffs=", self.dj, self.di, self.dt)

        self.xs = xs
        self.ys = ys
        self.tlist = tlist

        self.x = xs[start[1]]
        self.y = ys[self.ni - 1 - start[0]]
        # i, j, t , start and end store indices!!
        self.t = int(0)
        self.i = int(start[0])
        self.j = int(start[1])

        self.endpos = end
        self.start_state = (0, start[0], start[1])

    # Rewards and Actions to be dicitonaries
    def set_AR(self, Actions):
        self.actions = Actions
        # self.rewards= Rewards

    # explicitly set state. state is a tuple of indices(m,n,p)
    def set_state(self, state):
        self.t = state[0]
        self.i = state[1]
        self.j = state[2]

    def current_state(self):
        return (int(self.t), int(self.i), int(self.j))

    def current_pos(self):
        return (int(self.i), int(self.j))

    # MAY NEED TO CHANGE DEFINITION
    def is_terminal(self):
        # return self.actions[state]==None
        return (self.current_pos() == self.endpos)

    def move(self, action, Vx, Vy):
        r = 0
        so = self.current_state()
        if self.is_terminal() == False:
            thrust, angle = action

            # x0.0091 for m to degrees
            vnetx = (thrust * math.cos(angle) + (Vx))
            vnety = (thrust * math.sin(angle) + (Vy))

            xnew = (self.xs[int(self.j)]) + (vnetx * self.dt)
            ynew = (self.ys[int(self.ni - 1 - self.i)]) + (vnety * self.dt)

            # xnew = (self.xs[int(self.j)]) + (vnetx * self.dt * 0.0091)
            # ynew = (self.ys[int(self.ni - 1 - self.i)]) + (vnety * self.dt * 0.0091)

            # if state happens to go out of of grid, bring it back inside
            if xnew > self.xs[-1]:
                xnew = self.xs[-1]
                r -= 10
            elif xnew < self.xs[0]:
                xnew = self.xs[0]
                r -= 10
            if ynew > self.ys[-1]:
                ynew = self.ys[-1]
                r -= 10
            elif ynew < self.ys[0]:
                ynew = self.ys[0]
                r -= 10

            # rounding to prevent invalid keys

            remx = (xnew - self.xs[0]) % self.dj
            remy = -(ynew - self.ys[-1]) % self.di
            xind = (xnew - self.xs[0]) // self.dj
            yind = -(ynew - self.ys[-1]) // self.di

            if remx >= 0.5 * self.dj and remy >= 0.5 * self.di:
                xind += 1
                yind += 1
            elif remx >= 0.5 * self.dj and remy < 0.5 * self.di:
                xind += 1
            elif remx < 0.5 * self.dj and remy >= 0.5 * self.di:
                yind += 1

            self.i = int(yind)
            self.j = int(xind)
            # putting before update because only i,j matter
            # and makes it easier to compare so,sn in compare fn
            sn = (self.t, self.i, self.j)

            self.t = self.t + 1

            r += calculate_reward_const_dt(self.dt, self.xs, self.ys, so, sn, vnetx, vnety)

            if self.is_terminal():
                r += 100

        return r

    def move_exact(self, action, Vx, Vy):
        r = 0
        if self.is_terminal() == False:
            thrust, angle = action
            # x0.01 for cm/s to m/s
            # x0.0091 for m to degrees
            vnetx = (thrust * math.cos(angle) + (Vx))
            vnety = (thrust * math.sin(angle) + (Vy))
            xnew = self.x + (vnetx * self.dt)
            ynew = self.y + (vnety * self.dt)
            # print("xnew, ynew",xnew,ynew)
            # if state happens to go out of of grid, bring it back inside
            if xnew > self.xs[-1]:
                xnew = self.xs[-1]
            elif xnew < self.xs[0]:
                xnew = self.xs[0]

            if ynew > self.ys[-1]:
                ynew = self.ys[-1]
            elif ynew < self.ys[0]:
                ynew = self.ys[0]
            # print("xnew, ynew after boundingbox", xnew, ynew)
            # rounding to prevent invalid keys

            self.x = xnew
            self.y = ynew

            remx = (xnew - self.xs[0]) % self.dj
            remy = -(ynew - self.ys[-1]) % self.di
            xind = (xnew - self.xs[0]) // self.dj
            yind = -(ynew - self.ys[-1]) // self.di

            # print("rex,remy,xind,yind", remx,remy,xind,yind)

            if remx >= 0.5 * self.dj and remy >= 0.5 * self.di:
                xind += 1
                yind += 1
            elif remx >= 0.5 * self.dj and remy < 0.5 * self.di:
                xind += 1
            elif remx < 0.5 * self.dj and remy >= 0.5 * self.di:
                yind += 1
            # print("rex,remy,xind,yind after upate", remx, remy, xind, yind)
            # print("(i,j)", (yind,xind))
            self.i = int(yind)
            self.j = int(xind)

            self.t = self.t + 1
            sn = (self.t, self.i, self.j)

            r = -self.dt

            if self.is_terminal():
                r += 10

        return r

    # !! time to mentioned by index !!
    def ac_state_space(self, time=None):
        a = set()
        if time == None:
            for t in range((self.nt) - 1):  # does not include the states in the last time stice.
                for i in range(self.ni):
                    for j in range(self.nj):
                        if ((i, j) != self.endpos):  # does not include states with pos as endpos
                            a.add((t, i, j))

        else:
            for i in range(self.ni):
                for j in range(self.nj):
                    if ((i, j) != self.endpos):
                        a.add((time, i, j))

        return sorted(a)

    def state_space(self, edge=None):
        a = set()
        if edge == None:
            for t in range(self.nt):
                for i in range(self.ni):
                    for j in range(self.nj):
                        a.add((t, i, j))

        elif edge == 'l':
            j = 0
            for t in range(self.nt):
                for i in range(self.ni):
                    a.add((t, i, j))

        elif edge == 'd':
            i = self.ni - 1
            for t in range(self.nt):
                for j in range(self.nj):
                    a.add((t, i, j))

        elif edge == 'r':
            j = (self.nj) - 1
            for t in range(self.nt):
                for i in range(self.ni):
                    a.add((t, i, j))

        elif edge == 'u':
            i = 0
            for t in range(self.nt):
                for j in range(self.nj):
                    a.add((t, i, j))

        elif edge == 'm':
            for t in range(self.nt):
                for i in range(1, (self.ni) - 1):
                    for j in range(1, (self.nj) - 1):
                        a.add((t, i, j))

        elif edge == 'llc':
            j = 0
            i = self.ni - 1
            for t in range(self.nt):
                a.add((t, i, j))

        elif edge == 'ulc':
            j = 0
            i = 0
            for t in range(self.nt):
                a.add((t, i, j))

        elif edge == 'lrc':
            j = (self.nj) - 1
            i = self.ni - 1
            for t in range(self.nt):
                a.add((t, i, j))

        elif edge == 'urc':
            j = (self.nj) - 1
            i = 0
            for t in range(self.nt):
                a.add((t, i, j))

        elif edge == 'end':
            j = self.endpos[1]
            i = self.endpos[0]
            for t in range(self.nt):
                a.add((t, i, j))

        return sorted(a)

    def if_within_time(self):
        return (self.t >= 0 and self.t < self.nt)

    def if_within_actionable_time(self):
        return (self.t >= 0 and self.t < self.nt - self.dt)

    def if_within_TD_actionable_time(self):
        return (self.t >= 0 and self.t < self.nt - 2*self.dt)

    def if_within_grid(self, s):
        t = s[0]
        i = s[1]
        j = s[2]
        return (j <= (self.nj) - 1 and j >= 0 and i <= (self.ni) - 1 and i >= 0 and t <= (self.nt) - 1 and t >= 0)

    # rrpos is relatively reachable positions: an i,j tuple
    def reachable_state(self, s, rrpos):
        t = s[0]
        i = s[1]
        j = s[2]
        irel = rrpos[0]
        jrel = rrpos[1]
        iabs = i + irel
        jabs = j + jrel

        return (t, iabs, jabs)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def timeOpt_grid(tlist, xs, ys, startpos, endpos):
    g = Grid(tlist, xs, ys, startpos, endpos)

    # define actions and rewards
    Pi = math.pi
    # speeds in m/s
    speed_list = [1]  # speeds except zero
    angle_list_l = [0, 0.25 * Pi, 0.5 * Pi, 1.5 * Pi, 1.75 * Pi]
    angle_list_d = [0, 0.25 * Pi, 0.5 * Pi, 0.75 * Pi, Pi]
    angle_list_u = [0, Pi, 1.25 * Pi, 1.5 * Pi, 1.75 * Pi]
    angle_list_r = [0.5 * Pi, 0.75 * Pi, Pi, 1.25 * Pi, 1.5 * Pi]
    angle_list_m = [0, 0.25 * Pi, 0.5 * Pi, 0.75 * Pi, Pi, 1.25 * Pi, 1.5 * Pi, 1.75 * Pi]

    # angle_list_l=[0]
    # angle_list_d=[0, Pi]
    # angle_list_u=[0, Pi]
    # angle_list_r=[Pi]
    # angle_list_m=[0, Pi]

    action_list_l = list(itertools.product(speed_list, angle_list_l))
    # action_list_l.append((0,0))
    action_list_d = list(itertools.product(speed_list, angle_list_d))
    # action_list_d.append((0,0))
    action_list_u = list(itertools.product(speed_list, angle_list_u))
    # action_list_u.append((0,0))
    action_list_r = list(itertools.product(speed_list, angle_list_r))
    # action_list_r.append((0,0))
    action_list_m = list(itertools.product(speed_list, angle_list_m))
    # action_list_m.append((0,0))

    actions = {}
    for s in g.state_space():
        actions[s] = action_list_m

    # update action set for grid edges
    for s in g.state_space('l'):
        actions[s] = action_list_l

    for s in g.state_space('u'):
        actions[s] = action_list_u

    for s in g.state_space('r'):
        actions[s] = action_list_r

    for s in g.state_space('d'):
        actions[s] = action_list_d

    # update action set for grid corners
    for s in g.state_space('llc'):
        actions[s] = intersection(action_list_d, action_list_l)

    for s in g.state_space('ulc'):
        actions[s] = intersection(action_list_u, action_list_l)

    for s in g.state_space('lrc'):
        actions[s] = intersection(action_list_d, action_list_r)

    for s in g.state_space('urc'):
        actions[s] = intersection(action_list_u, action_list_r)

    # update action for terminal state
    for s in g.state_space('end'):
        actions[s] = None

    # set action set for states in last time step to none
    for s in g.ac_state_space((g.nt) - 1):
        actions[s] = None

    # set ations for grid
    g.set_AR(actions)

    return g


def print_values(V, g):
    for t in range(g.nt):
        print('t=', t)
        print("")
        for j in reversed(range(g.ni)):
            print("---------------------------")
            for i in range(g.nj):
                v = V[t, i, j]
                print(v, end=" ")

            print("")
        print("")


def print_policy(P, g):
    for t in range(g.nt - 1):
        print('t=', t)
        print("")
        for j in reversed(range(g.ni)):
            print("---------------------------")
            for i in range(g.nj):
                if (i, j) != g.endpos:
                    a = P[(t, i, j)]
                    if a[0] != 0:
                        if a[1] == 0:
                            a = (a[0], ' R')
                        elif a[1] == math.pi * 0.5:
                            a = (a[0], ' U')
                        elif a[1] == math.pi:
                            a = (a[0], ' L')
                        elif a[1] == math.pi * 1.5:
                            a = (a[0], ' D')
                        elif a[1] == math.pi * 0.25:
                            a = (a[0], 'UR')
                        elif a[1] == math.pi * 0.75:
                            a = (a[0], 'UL')
                        elif a[1] == math.pi * 1.25:
                            a = (a[0], 'DL')
                        elif a[1] == math.pi * 1.75:
                            a = (a[0], 'DR')
                else:
                    a = 'None'
                print(a, end=" ")
            print("")
        print("")


def print_test(g):
    for t in range(g.nt):
        for j in reversed(range(g.ni)):
            for i in range(g.nj):
                print(t, i, j),

            print("")
