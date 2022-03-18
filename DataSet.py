import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.io

path = 'F:/project/VScode/data/'

class DataSet:
    def __init__(self, x_range, y_range, Nx_train, Ny_train, N_bc):
        self.x_range = x_range
        self.y_range = y_range
        self.Nx_train = Nx_train
        self.Ny_train = Ny_train
        self.N_bc = N_bc
        self.e = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
        self.w = np.full((9, 1), 0.0)
        self.w[0] = 4 / 9
        self.w[1:5] = 1 / 9
        self.w[5:] = 1 / 36
        self.RT = 100
        self.xi = self.e * np.sqrt(3 * self.RT)
        self.uw = 0.1
        self.Re = 400.0
        self.L = 1.
        self.tau = 3 * self.L * self.uw / self.Re + 0.5

        self.sess = tf.Session()

    def feq_gradient(self, rou, u, v, x, y):

        rou_x = tf.gradients(rou, x)[0]
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        rou_y = tf.gradients(rou, y)[0]
        u_y = tf.gradients(u, y)[0]
        v_y = tf.gradients(v, y)[0]
        f_sum = self.feq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y)
        return f_sum

    # concat of dfeq / dX
    def feq_xy(self, rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y):
        f_sum = self.dfeq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y, 0)
        for i in range(1, 9):
            f_ = self.dfeq_xy(rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y, i)
            f_sum = tf.concat([f_sum, f_], 1)
        return f_sum

    # difference of f_eq for x, y
    def dfeq_xy(self, rou, u, v, rou_x, rou_y, u_x, v_x, u_y, v_y, i):
        feq_x = self.w[i, :] * rou_x * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_x + self.xi[i, 1] * v_x) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_x + self.xi[i, 1] * v_x) / self.RT ** 2 - (u * u_x + v * v_x) / self.RT)
        # here need to change the equations
        feq_y = self.w[i, :] * rou_y * (1 + (self.xi[i, 0] * u + self.xi[i, 1] * v) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) ** 2 / 2 / self.RT ** 2 - (u ** 2 + v ** 2) / 2 / self.RT) + \
                self.w[i, :] * rou * ((self.xi[i, 0] * u_y + self.xi[i, 1] * v_y) / self.RT + (self.xi[i, 0] * u + self.xi[i, 1] * v) * (self.xi[i, 0] * u_y + self.xi[i, 1] * v_y) / self.RT ** 2 - (u * u_y + v * v_y) / self.RT)
        dfeq_xy = self.xi[i, 0] * feq_x + self.xi[i, 1] * feq_y
        return dfeq_xy

    # concat of f_eq_i
    def f_eq(self, rou, u, v):
        f_eq_sum = self.f_eqk(rou, u, v, 0)
        for i in range(1, 9):
            f_eq = self.f_eqk(rou, u, v, i)
            f_eq_sum = tf.concat([f_eq_sum, f_eq], 1)
        return f_eq_sum

    # f_eq equation
    def f_eqk(self, rou, u, v, k):
        f_eqk = self.w[k, :] * rou * (1 + (self.xi[k, 0]*u + self.xi[k, 1]*v) / self.RT + (self.xi[k, 0]*u + self.xi[k, 1]*v) ** 2 / 2 / self.RT ** 2 - (u*u + v*v) / 2 / self.RT)
        return f_eqk

    # the mean pde
    def bgk(self, f_neq, rou, u, v, x, y):
        feq_pre = self.feq_gradient(rou, u, v, x, y)
        R_sum = 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq[:, k][:, None], x)[0]
            fneq_y = tf.gradients(f_neq[:, k][:, None], y)[0]
            R = (self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None])) ** 2
            R_sum = R_sum + R
        return R_sum

    # the equation residual
    def Eq_res(self, f_neq, rou, u, v, x, y):
        feq_pre = self.feq_gradient(rou, u, v, x, y)
        Eq_sum = x * 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq[:, k][:, None], x)[0]
            fneq_y = tf.gradients(f_neq[:, k][:, None], y)[0]
            Eq = tf.abs(self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (f_neq[:, k][:, None]))
            Eq_sum = tf.concat([Eq_sum, Eq], 1)
        return Eq_sum[:, 1]

    # boundary condition
    def inward_judge(self, x, y):
        x = tf.where(tf.equal(x, 2.0), x * 0 - 3.0, x)
        x = tf.where(tf.equal(x, -0.5), x * 0 + 3.0, x)
        x = tf.where(tf.equal(tf.abs(x), 3.0), x / 3.0, x * 0.0)
        y = tf.where(tf.equal(y, 1.5), y * 0 - 3.0, y)
        y = tf.where(tf.equal(y, -0.5), y * 0 + 3.0, y)
        y = tf.where(tf.equal(tf.abs(y), 3.0), y / 3.0, y * 0.0)
        return x, y

    def bgk_cond(self, f_neq, feq_pre, x, y):
        [xx, yy] = self.inward_judge(x, y)
        R_sum = 0
        for k in range(9):
            fneq_x = tf.gradients(f_neq, x)[0]
            fneq_y = tf.gradients(f_neq, y)[0]
            cond = self.xi[k, 0] * xx + self.xi[k, 1] * yy
            cond = tf.squeeze(cond)
            R_ = (self.xi[k, 0] * fneq_x + self.xi[k, 1] * fneq_y + feq_pre[:, k][:, None] + 1 / self.tau * (
            f_neq[:, k][:, None])) ** 2
            R = tf.where(tf.greater(cond, cond * 0), R_ * 0, R_)
            R_sum = R_sum + R

        return R_sum

    def fneq_train(self, u, v, rho):
        data = scipy.io.loadmat('Results/PINN_BGK.mat')
        rho_x = data['rho_x']
        rho_y = data['rho_y']
        u_x = data['u_x']
        u_y = data['u_y']
        v_x = data['v_x']
        v_y = data['v_y']

        fneq = -self.tau * self.feq_xy(rho, u, v, rho_x, rho_y, u_x, v_x, u_y, v_y)
        fneq = fneq.eval(session=self.sess)

        return fneq

    def LoadData(self):
        data_rho = pd.read_table(path + 'rho.dat', sep=' ', header=None, engine='python')
        rho = np.array(data_rho)
        rho = rho[:, :-1]

        data_u = pd.read_table(path + 'u.dat', sep=' ', header=None, engine='python')
        u = np.array(data_u)
        u = u[:, :-1]

        data_v = pd.read_table(path + 'v.dat', sep=' ', header=None, engine='python')
        v = np.array(data_v)
        v = v[:, :-1]

        data_x = pd.read_table(path + 'x.dat', sep='\t', header=None, engine='python')
        x = np.array(data_x)

        data_y = pd.read_table(path + 'y.dat', sep='\t', header=None, engine='python')
        y = np.array(data_y)

        return rho, u, v, x, y

    def Data_Generation(self):
        rho, u, v, x, y = self.LoadData()

        dx = 1/257
        dy = 1/257

        x_u = self.x_range.max()
        x_l = self.x_range.min()
        y_u = self.y_range.max()
        y_l = self.y_range.min()

        # domain data
        X_data = np.random.random((16000, 1)) * (x_u - x_l) + x_l
        Y_data = np.random.random((16000, 1)) * (y_u - y_l) + y_l

        # boundary data
        # up walll
        x_1 = x
        y_1 = np.full((1 / dy, 1), y.max())
        # down wall
        x_2 = x
        y_2 = np.full((1 / dy, 1), y.min())
        # left wall
        x_3 = np.full((1/dx, 1), x.min())
        y_3 = y
        # right wall
        x_4 = np.full((1/dx, 1), x.max())
        y_4 = y

        x_b = np.vstack((x_1, x_2, x_3, x_4))
        y_b = np.vstack((y_1, y_2, y_3, y_4))

        u_data = np.hstack((u[-1, :], u[0, :], u[:, 0], u[:, -1]))[:, None]
        v_data = np.hstack((v[-1, :], v[0, :], v[:, 0], v[:, -1]))[:, None]
        rho_data = np.hstack((rho[-1, :], rho[0, :], rho[:, 0], rho[:, -1]))[:, None]

        fneq_data = self.fneq_train(u_data, v_data, rho_data)

        return X_data, Y_data, x_b, y_b, u_data, v_data, rho_data, fneq_data