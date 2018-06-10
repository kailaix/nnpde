import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
#from drawnow import drawnow, figure

def assert_shape(x, shape):
    S = x.get_shape().as_list()
    if len(S)!=len(shape):
        raise Exception("Shape mismatch: {} -- {}".format(S, shape))
    for i in range(len(S)):
        if S[i]!=shape[i]:
            raise Exception("Shape mismatch: {} -- {}".format(S, shape))

def compute_delta(u, x):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:,0], x)[0]
    g2 = tf.gradients(grad[:,1], x)[0]
    delta = g1[:,0] + g2[:,1]
    assert_shape(delta, (None,))
    return delta

def compute_delta_nd(u, x, n):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:, 0], x)[0]
    delta = g1[:,0]
    for i in range(1,n):
        g = tf.gradients(grad[:,i], x)[0]
        delta += g[:,i]
    assert_shape(delta, (None,))
    return delta

def compute_dx(u,x):
    grad = tf.gradients(u, x)[0]
    dudx = grad[:,0]
    assert_shape(dudx, (None,))
    return dudx

def compute_dy(u,x):
    grad = tf.gradients(u, x)[0]
    dudy = grad[:,1]
    assert_shape(dudy, (None,))
    return dudy


def rectspace(a,b,c,d,n):
    x = np.linspace(a,b,n)
    y = np.linspace(c,d,n)
    [X,Y] = np.meshgrid(x,y)
    return np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)

##
class NNPDE_ND:
    def __init__(self,batch_size, N, d): # d- dimension, N-number of layers
        self.d = d
        self.batch_size = batch_size
        self.N = N

        self.x = tf.placeholder(tf.float64, (None, d))  # inner data
        self.x_b = tf.placeholder(tf.float64, (None, d))  # boundary data

        self.u_b = self.bsubnetwork(self.x_b, False)
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)

        self.bloss = tf.reduce_sum((self.tfexactsol(self.x_b) - self.u_b) ** 2)
        self.loss = self.loss_function()

        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss, var_list=var_list1)
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()

        self.X_test = np.array([0.5]*self.d)[np.newaxis,:]

    def tfexactsol(self, x):
        raise NotImplementedError

    def exactsol(self, x):
        raise NotImplementedError

    def B(self, x):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def subnetwork(self, x, reuse = False):
        with tf.variable_scope("inner"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="dense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def loss_function(self):
        deltah = compute_delta_nd(self.u, self.x, self.d)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    def compute_L2(self, sess, x):
        u0 = self.exactsol(x)
        u1 = sess.run(self.u, feed_dict={self.x: x})[0]
        return np.sqrt(np.mean((u0-u1)**2))



class NNPDE:
    def __init__(self, batch_size, N, refn):
        self.refn = refn  # reference points
        x = np.linspace(0, 1, refn)
        y = np.linspace(0, 1, refn)
        self.X, self.Y = np.meshgrid(x, y)
        self.refnX = np.concatenate([self.X.reshape((-1, 1)), self.Y.reshape((-1, 1))], axis=1)

        self.batch_size = batch_size  # batchsize
        self.N = N

        self.x = tf.placeholder(tf.float64, (None, 2))
        self.u = self.u_out(self.x)
        self.loss = self.loss_function()
        self.ploss = self.point_wise_loss()
        self.fig = plt.figure()

        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def exactsol(self, x, y):
        raise NotImplementedError

    # data to be modified
    def A(self, x):
        raise NotImplementedError

    def B(self, x):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    # end modification

    def subnetwork(self, x):
        for i in range(self.N):
            x = tf.layers.dense(x, 256, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 1, activation=None, name="last")
        x = tf.squeeze(x, axis=1)
        return x

    def u_out(self, x):
        res = self.A(x) + self.B(x) * self.subnetwork(x)
        assert_shape(res, (None,))
        return res

    def point_wise_loss(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.abs(deltah - delta)
        assert_shape(res, (None,))
        return res

    # def visualize(self, sess, showonlysol=False):

    #     x = np.linspace(0, 1, self.refn)
    #     y = np.linspace(0, 1, self.refn)
    #     [X, Y] = np.meshgrid(x, y)

    #     uh = sess.run(self.u, feed_dict={self.x: self.refnX})
    #     Z = uh.reshape((self.refn, self.refn))

    #     uhref = self.exactsol(X, Y)

    #     def draw():
    #         ax = self.fig.gca(projection='3d')
    #         if not showonlysol:
    #             ax.plot_surface(X, Y, uhref, rstride=1, cstride=1, cmap=cm.autumn,
    #                             linewidth=0, antialiased=False, alpha=0.3)

    #         ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.summer,
    #                         linewidth=0, antialiased=False, alpha=0.8)
    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)
    #         ax.set_zlim(0, 1.1)

    #         ax.set_xlabel('$x$')
    #         ax.set_ylabel('$y$')

    #     drawnow(draw)

    # def visualize_point_wise_loss(self, sess):
    #     ploss = sess.run(self.ploss, feed_dict={self.x: self.refnX})
    #     x = np.linspace(0, 1, self.refn)
    #     y = np.linspace(0, 1, self.refn)
    #     [X, Y] = np.meshgrid(x, y)
    #     Z = ploss.reshape((self.refn, self.refn))

    #     def draw():
    #         ax = self.fig.gca(projection='3d')
    #         Z0 = np.log(Z + 1e-16) / np.log(10.0)
    #         Z0[Z0 < 1e-10] = 0
    #         ax.plot_surface(X, Y, Z0, rstride=1, cstride=1, cmap=cm.winter,
    #                         linewidth=0, antialiased=False)

    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)

    #         ax.set_xlabel('$x$')
    #         ax.set_ylabel('$y$')

    #     drawnow(draw)

    # def visualize_error(self, sess):
    #     x = np.linspace(0, 1, self.refn)
    #     y = np.linspace(0, 1, self.refn)
    #     [X, Y] = np.meshgrid(x, y)
    #     uh = sess.run(self.u, feed_dict={self.x: np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)})
    #     Z = uh.reshape((self.refn, self.refn))

    #     def draw():
    #         ax = self.fig.gca(projection='3d')

    #         ax.plot_surface(X, Y, self.exactsol(X, Y) - Z, rstride=1, cstride=1, cmap=cm.winter,
    #                         linewidth=0, antialiased=False, alpha=0.85)

    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)
    #         ax.set_zlim(0, 1.1)

    #         ax.set_xlabel('$x$')
    #         ax.set_ylabel('$y$')

    #     drawnow(draw)

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        X = np.random.rand(self.batch_size, 2)
        _, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
        if i % 10 == 0:
            print("Iteration={}, loss= {}".format(i, loss))

# u(x;w_1, w_2) = A(x;w_1) + B(x) * N(x;w_2)
# L u(x;w_1, w_2) = L A(x;w_1) + L( B(x) * N(x;w_2) ) --> f

class NNPDE2:
    def __init__(self, batch_size, N, refn):
        self.rloss = []
        self.rbloss = []
        self.rl2 = []

        self.refn = refn  # reference points
        x = np.linspace(0, 1, refn)
        y = np.linspace(0, 1, refn)
        self.X, self.Y = np.meshgrid(x, y)
        self.refX = np.concatenate([self.X.reshape((-1, 1)), self.Y.reshape((-1, 1))], axis=1)

        self.batch_size = batch_size  # batchsize
        self.N = N # number of dense layers

        self.x = tf.placeholder(tf.float64, (None, 2)) # inner data
        self.x_b = tf.placeholder(tf.float64, (None, 2)) # boundary data

        self.u_b = self.bsubnetwork(self.x_b, False)
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)

        self.bloss = tf.reduce_sum((self.tfexactsol(self.x_b)-self.u_b)**2)
        self.loss = self.loss_function()

        self.ploss = self.point_wise_loss()



        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss,var_list=var_list1)
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()


    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def exactsol(self, x, y):
        raise NotImplementedError

    def tfexactsol(self, x):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    # end modification

    def subnetwork(self, x, reuse = False):
        with tf.variable_scope("inner"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="dense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def point_wise_loss(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.abs(deltah - delta)
        assert_shape(res, (None,))
        return res

    def plot_exactsol(self):
        Z = self.exactsol(self.X, self.Y)
        ax = self.fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                        linewidth=0, antialiased=False, alpha=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')




    # def visualize(self, sess, showonlysol=False, i=None, savefig=None):

    #     x = np.linspace(0, 1, self.refn)
    #     y = np.linspace(0, 1, self.refn)
    #     [X, Y] = np.meshgrid(x, y)

    #     uh = sess.run(self.u, feed_dict={self.x: self.refX})
    #     Z = uh.reshape((self.refn, self.refn))

    #     uhref = self.exactsol(X, Y)

    #     def draw():
    #         self.fig = plt.figure()
    #         ax = self.fig.gca(projection='3d')

    #         if not showonlysol:
    #             ax.plot_surface(X, Y, uhref, rstride=1, cstride=1, cmap=cm.autumn,
    #                             linewidth=0, antialiased=False, alpha=0.3)

    #         ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.summer,
    #                         linewidth=0, antialiased=False, alpha=0.5)
    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)
    #         ax.set_zlim(0, 1.1)

    #         ax.set_xlabel('$x$')
    #         ax.set_ylabel('$y$')
    #         if i:
    #             plt.title("Iteration {}".format(i))
    #         if savefig:
    #             plt.savefig("{}/fig{}".format(savefig,0 if i is None else i))

    #     drawnow(draw)

    # def visualize_point_wise_loss(self, sess):
    #     ploss = sess.run(self.ploss, feed_dict={self.x: self.refnX})
    #     x = np.linspace(0, 1, self.refn)
    #     y = np.linspace(0, 1, self.refn)
    #     [X, Y] = np.meshgrid(x, y)
    #     Z = ploss.reshape((self.refn, self.refn))

    #     def draw():
    #         ax = self.fig.gca(projection='3d')
    #         Z0 = np.log(Z + 1e-16) / np.log(10.0)
    #         Z0[Z0 < 1e-10] = 0
    #         ax.plot_surface(X, Y, Z0, rstride=1, cstride=1, cmap=cm.winter,
    #                         linewidth=0, antialiased=False)

    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)

    #         ax.set_xlabel('$x$')
    #         ax.set_ylabel('$y$')

    #     # drawnow(draw)
    #     draw()

    # def visualize_error(self, sess):
    #     x = np.linspace(0, 1, self.refn)
    #     y = np.linspace(0, 1, self.refn)
    #     [X, Y] = np.meshgrid(x, y)
    #     uh = sess.run(self.u, feed_dict={self.x: np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)})
    #     Z = uh.reshape((self.refn, self.refn))

    #     def draw():
    #         ax = self.fig.gca(projection='3d')

    #         ax.plot_surface(X, Y, self.exactsol(X, Y) - Z, rstride=1, cstride=1, cmap=cm.winter,
    #                         linewidth=0, antialiased=False, alpha=0.85)

    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)
    #         ax.set_zlim(0, 1.1)

    #         ax.set_xlabel('$x$')
    #         ax.set_ylabel('$y$')

    #     drawnow(draw)

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        bX = np.zeros((4*self.batch_size, 2))
        bX[:self.batch_size,0] = np.random.rand(self.batch_size)
        bX[:self.batch_size,1] = 0.0

        bX[self.batch_size:2*self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2*self.batch_size, 1] = 1.0

        bX[2*self.batch_size:3*self.batch_size, 0] = 0.0
        bX[2*self.batch_size:3*self.batch_size, 1] = np.random.rand(self.batch_size)

        bX[3*self.batch_size:4*self.batch_size, 0] = 1.0
        bX[3 * self.batch_size:4 * self.batch_size, 1] = np.random.rand(self.batch_size)

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

        X = np.random.rand(self.batch_size, 2)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})


        ########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX})
        Z = uh.reshape((self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y)
        self.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )
        ########## record loss ############


        if i % 10 == 0:
            print("Iteration={}, bloss = {}, loss= {}, L2={}".format(i, bloss, loss, self.rl2[-1]))