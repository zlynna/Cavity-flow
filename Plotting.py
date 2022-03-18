import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from DataSet import DataSet

SavePath = './Results/'
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

class Plotting:
    def __init__(self, x_range, NX, y_range, NY, sess):
        self.x_range = x_range
        self.y_range = y_range
        self.NX = NX
        self.NY = NY
        self.N_bc = 300
        self.sess = sess

    def Saveplot(self, u_pre, v_pre, Eq_res, x_train, y_train):
        # exact solution
        data = DataSet(self.x_range, self.y_range, self.NX, self.NY, self.N_bc)
        rho, u, v, x, y = data.LoadData()
        xx, yy = np.meshgrid(x, y)
        x_test = np.ravel(xx).T[:, None]
        y_test = np.ravel(yy).T[:, None]

        tf_dict = {x_train: x_test, y_train: y_test}
        u_pred = self.sess.run(u_pre, tf_dict)
        u_pred = u_pred.reshape(257, 257)
        v_pred = self.sess.run(v_pre, tf_dict)
        v_pred = v_pred.reshape(257, 257)

        u_error = np.abs(u_pred - u)
        v_error = np.abs(v_pred - v)

        """for i in range(1, 10):
            error_eq = tf.reduce_mean(Eq_res[:, i])
            error_eq = error_eq.eval(session=self.sess)
            print('Error Equation_res %o: %e' % (i, error_eq))"""

        error_u = self.relative_error_(u_pred, u)
        error_v = self.relative_error_(v_pred, v)
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))

        fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))
        self.plotter(fig1, ax1, u_pred, r'$PINN-BGK_u$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax2, u, r'$Exact_u$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax3, u_error, r'$Error_u$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax4, v_pred, r'$PINN-BGK_v$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax5, v, r'$Exact_v$', 'x', 'y', xx, yy)
        self.plotter(fig1, ax6, v_error, r'$Error_v$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(SavePath + 'speed.png')

        """fig5, ((ax34, ax35, ax36), (ax37, ax38, ax39), (ax40, ax41, ax42)) = plt.subplots(3, 3, figsize=(15, 8))
        self.plotter(fig5, ax34, Eq_res[:, 1], r'$Eq_{f_{0}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax35, Eq_res[:, 1], r'$Eq_{f_{1}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax36, Eq_res[:, 2], r'$Eq_{f_{2}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax37, Eq_res[:, 3], r'$Eq_{f_{3}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax38, Eq_res[:, 4], r'$Eq_{f_{4}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax39, Eq_res[:, 5], r'$Eq_{f_{5}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax40, Eq_res[:, 6], r'$Eq_{f_{6}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax41, Eq_res[:, 7], r'$Eq_{f_{7}}$', 'x', 'y', xx, yy)
        self.plotter(fig5, ax42, Eq_res[:, 8], r'$Eq_{f_{8}}$', 'x', 'y', xx, yy)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(SavePath + 'Eq_residual.png')"""

        plt.show()

    def plotter(self, fig, ax, dat, title, xlabel, ylabel, xx, yy):
        dat = dat.reshape((257, 257))
        levels = np.linspace(dat.min(), dat.max(), 100)
        zs = ax.contourf(xx, yy, dat, cmap='jet', levels=levels)
        fig.colorbar(zs, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        return zs

    def relative_error_(self, pred, exact):
        if type(pred) is np.ndarray:
            return np.sqrt(np.sum(np.square(pred - exact)) / np.sum(np.square(exact)))
        return tf.sqrt(tf.square(pred - exact) / tf.square(exact))
