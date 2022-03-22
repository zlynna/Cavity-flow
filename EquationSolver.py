import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
import time

from DataSet import DataSet
from net import Net
from ModelTrain import Train
from Plotting import Plotting

np.random.seed(1234)
tf.set_random_seed(1234)

UseLoadNet = False

def main():
    y_range = np.array((0.5/257, 256.5/257))
    x_range = np.array((0.5/257, 256.5/257))
    NX = 17000
    Ny = 17000
    N_bc = 300

    data = DataSet(x_range, y_range, NX, Ny, N_bc)
    # input data
    x_data, y_data, x_bc, y_bc, u_data, v_data, rho_data, fneq_data = data.Data_Generation()
    # size of the DNN
    layers_eq = [2] + 4 * [40] + [3]
    layers_neq = [2] + 4 * [40] + [9]
    # definition of placeholder
    [x_train, y_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(2)]
    [x_bc_train, y_bc_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(2)]
    [u_train, v_train, rho_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
    fneq_train = tf.placeholder(tf.float32, shape=[None, 9])
    # definition of nn
    net_eq = Net(x_data, y_data, layers=layers_eq)
    net_neq = Net(x_data, y_data, layers=layers_neq)

    [rou_pre, u_pre, v_pre] = net_eq(x_train, y_train)
    fneq_pre = net_neq(x_train, y_train) / 10
    [rou_bc_pre, u_bc_pre, v_bc_pre] = net_eq(x_bc_train, y_bc_train)
    fneq_bc_pre = net_neq(x_bc_train, y_bc_train) / 10

    bgk = data.bgk(fneq_pre, rou_pre, u_pre, v_pre, x_train, y_train)
    bgk_bc = data.bgk(fneq_bc_pre, rou_bc_pre, u_bc_pre, v_bc_pre, x_bc_train, y_bc_train)

    Eq_res = data.Eq_res(fneq_pre, rou_pre, u_pre, v_pre, x_train, y_train)

    # loss
    loss = tf.reduce_mean(tf.square(u_bc_pre - u_train)) * 1e3 + \
           tf.reduce_mean(tf.square(v_bc_pre - v_train)) * 1e3 + \
           tf.reduce_mean(tf.square(rou_bc_pre - rho_train)) + \
           tf.reduce_mean(bgk) + \
           tf.reduce_mean(bgk_bc) + \
           tf.reduce_mean(tf.square(fneq_bc_pre - fneq_train)) * 10

    train_adam = tf.train.AdamOptimizer().minimize(loss)
    train_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                         method="L-BFGS-B",
                                                         options={'maxiter': 50000,
                                                                  'ftol': 1.0 * np.finfo(float).eps
                                                                  }
                                                         )
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf_dict = {x_train: x_data, y_train: y_data,
               x_bc_train: x_bc, y_bc_train: y_bc,
               u_train: u_data, v_train: v_data, rho_train: rho_data, fneq_train: fneq_data}

    Model = Train(tf_dict)
    start_time = time.perf_counter()

    if UseLoadNet:
        Model.LoadModel(sess)
    else:
        Model.ModelTrain(sess, loss, train_adam, train_lbfgs)

    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds' % (stop_time - start_time))

    NX_test = 125
    NY_test = 100
    Plotter = Plotting(x_range, NX_test, y_range, NY_test, sess)
    Plotter.Saveplot(u_pre, v_pre, Eq_res, x_train, y_train)

if __name__ == '__main__':
    main()
