from netCDF4 import Dataset
from scipy import interpolate
import numpy as np
from custom_functions import my_meshgrid

# nx=100
# ny=100

def generate_states(xmin, xmax, nx):
    xs = np.linspace(xmin,xmax,nx)
    return xs

def pos_interpolate_vel(velx_data, X, Y, xs, ys, nx, ny, tsteps_d):
    posV = np.zeros((tsteps_d, nx, ny))
    for i in range(tsteps_d):
        velx_data_f = np.ndarray.flatten(velx_data[i, :, :])
        posV[i, :, :] = interpolate.griddata((X, Y), velx_data_f, (xs, ys), method='linear')
    return posV

def time_interpolate_vel(posVx, tlist_i, tlist_f, xs, ys, nx, ny, nt):
    Vx = np.zeros((nt, nx, ny))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            int_t = interpolate.interp1d(tlist_i, posVx[:, i, j])
            for k in range(nt):
                Vx[k, i, j] = int_t(tlist_f[k])
    return Vx


def velocity_field():
    nx = 10
    ny = 10
    nt = 30
    dt = 9

    to_id = 1
    tf_id = 3
    xo_id = 102
    xf_id = 202
    yo_id = 103
    yf_id = 203
    depth_id = 0

    dataset = Dataset('/home/rohit/Documents/Reinforcement_Learning/Research/Run01/pe_out.nc')

    velx_data = 0.01*dataset.variables['vtot'][to_id:tf_id, xo_id:xf_id, yo_id:yf_id, depth_id, 0]
    vely_data = 0.01*dataset.variables['vtot'][to_id:tf_id, xo_id:xf_id, yo_id:yf_id, depth_id, 1]

    X = dataset.variables['vgrid2'][xo_id:xf_id, yo_id:yf_id, 0]
    Y = dataset.variables['vgrid2'][xo_id:xf_id, yo_id:yf_id, 1]
    print("Xshape=", X.shape)

    # boundingbox
    xmin = np.max(X[:, 0])
    xmax = np.min(X[:, -1])
    ymin = np.max(Y[0, :])
    ymax = np.min(Y[-1, :])

    # to be eligible for scipy.int.griddata
    X = np.ndarray.flatten(X)
    Y = np.ndarray.flatten(Y)

    # grid for statespace
    xs1d = generate_states(xmin, xmax, nx)
    ys1d = generate_states(ymin, ymax, ny)
    xs, ys = my_meshgrid(xs1d, ys1d)
    print("xs,xs1d, ys, ys1d shape=", xs.shape, xs1d.shape, ys.shape, ys1d.shape)

    tsteps_d = velx_data.shape[0]

    posVx = pos_interpolate_vel(velx_data, X, Y, xs, ys, nx, ny, tsteps_d)
    posVy = pos_interpolate_vel(vely_data, X, Y, xs, ys, nx, ny, tsteps_d)

    # replace 13 with general variables
    # initial list of times as per ocean data in nc file
    tlist_i = np.zeros(tsteps_d)
    for i in range(tsteps_d):
        tlist_i[i] = 3600 * 3 * i

    # final list of time as per required in statespace

    tlist_f = np.zeros(nt)
    for i in range(nt):
        tlist_f[i] = (i * dt)
    print("tlistf",tlist_f)
    print("tlisti",tlist_i)

    Vx = time_interpolate_vel(posVx, tlist_i, tlist_f, xs, ys, nx, ny, nt)
    Vy = time_interpolate_vel(posVy, tlist_i, tlist_f, xs, ys, nx, ny, nt)

    return Vx, Vy, xs1d, ys1d, tlist_f
