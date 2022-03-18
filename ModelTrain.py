import tensorflow as tf
import numpy as np
import os

SavePath = './Net/'
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

class Train:
    def __init__(self, tf_dict):
        self.tf_dict = tf_dict
        self.step = 0
        self.Saver = tf.train.Saver()
    def callback(self, loss_):
        self.step += 1
        if self.step%100 == 0:
            print('Loss: %.3e'%(loss_))
    def ModelTrain(self, sess, loss, train_adam, train_lbfgs):
        n = 0
        nmax = 50000
        loss_c = 1.0e-3
        loss_ = 1.0
        while n < nmax and loss_ > loss_c:
            n += 1
            loss_, _ = sess.run([loss, train_adam], feed_dict=self.tf_dict)
            if n % 10 == 0:
                print('Steps: %d, loss: %.5e' % (n, loss_))

        train_lbfgs.minimize(sess, feed_dict=self.tf_dict, fetches=[loss], loss_callback=self.callback)
        self.Saver.save(sess, SavePath + 'PINN.ckpt')

    # Load Model
    def LoadModel(self, sess):
        self.Saver.restore(sess, SavePath + 'PINN.ckpt')
