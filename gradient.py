import tensorflow as tf
import numpy as np
from DataSet import DataSet
from ModelTrain import Train
from net import Net
import scipy.io

def main():
    y_range = np.array((0.5/257, 256.5/257))
    x_range = np.array((0.5/257, 256.5/257))
    NX = 17000
    Ny = 17000
    N_bc = 300

    data = DataSet(x_range, y_range, NX, Ny, N_bc)
    # input data
    _, _, x_bc, y_bc, u_data, v_data, rho_data, _ = data.Data_Generation()
    # size of nn
    layers = [2] + 2*[30] +[3]
    # placeholder
    [x_tf, y_tf, u_tf, v_tf, rho_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
    # nn
    gra = Net(x_bc, y_bc, layers=layers)
    # pre
    [rho_pre, u_pre, v_pre] = gra(x_tf, y_tf)
    U = tf.concat([rho_pre, u_pre, v_pre], 1)
    U_x = tf.gradients(U, x_tf)[0]

    rho_x = tf.gradients(rho_pre, x_tf)[0]
    u_x = tf.gradients(u_pre, x_tf)[0]
    v_x = tf.gradients(v_pre, x_tf)[0]

    rho_y = tf.gradients(rho_pre, y_tf)[0]
    u_y = tf.gradients(u_pre, y_tf)[0]
    v_y = tf.gradients(v_pre, y_tf)[0]

    # loss
    loss = tf.reduce_mean(tf.square(u_pre - u_tf)) + \
           tf.reduce_mean(tf.square(v_pre - v_tf)) + \
           tf.reduce_mean(tf.square(rho_pre - rho_tf))

    train_adam = tf.train.AdamOptimizer().minimize(loss)
    train_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B", options={'maxiter': 50000})

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf_dict = {x_tf: x_bc, y_tf: y_bc, u_tf: u_data, v_tf: v_data, rho_tf: rho_data}
    Model = Train(tf_dict)
    Model.ModelTrain(sess, loss, train_adam, train_lbfgs)

    # predict
    tf_dict2 = {x_tf: x_bc, y_tf: y_bc}
    u = sess.run(u_pre, tf_dict2)
    v = sess.run(v_pre, tf_dict2)
    rho = sess.run(rho_pre, tf_dict2)
    rho_x = sess.run(rho_x, tf_dict2)
    rho_y = sess.run(rho_y, tf_dict2)
    u_x = sess.run(u_x, tf_dict2)
    u_y = sess.run(u_y, tf_dict2)
    v_x = sess.run(v_x, tf_dict2)
    v_y = sess.run(v_y, tf_dict2)
    U_x = sess.run(U_x, tf_dict2)

    #
    print(U_x)
    print(rho_x)
    with tf.Session() as sess:
        print(U_x.eval())
        print(rho_x.eval())

    # save
    # scipy.io.savemat('Results/PINN_BGK.mat',{'rho_x': rho_x, 'rho_y': rho_y, 'u_x': u_x, 'u_y': u_y, 'v_x': v_x, 'v_y': v_y})

if __name__=='__main__':
    main()