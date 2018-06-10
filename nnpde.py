from pdebase import *

##
class Problem1(NNPDE):
    # data to be modified
    def exactsol(self,x,y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def A(self, x):
        return 0

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res
        # end modification
##
class Problem1_BD(NNPDE2):

    def exactsol(self,x,y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def tfexactsol(self,x):
        return tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])


    def f(self, x):
        return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

class ProblemPeak_BD(NNPDE2):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE2.__init__(self,batch_size, N, refn)

    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    # def loss_function(self):
    #     deltah = compute_delta(self.u, self.x)
    #     delta = self.f(self.x)
    #     delta = tf.clip_by_value(delta, -1e2, 1e2)
    #     deltah = tf.clip_by_value(deltah, -1e2, 1e2)
    #     res = tf.reduce_sum((deltah - delta) ** 2)
    #     assert_shape(res, ())
    #     return res

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        # delta = tf.clip_by_value(delta, -1e2, 1e2)
        # deltah = tf.clip_by_value(deltah, -1e2, 1e2)
        # weight = tf.clip_by_norm(1/delta**2, 10)
        # weight = tf.reduce_sum(delta**2)/delta**2
        res = tf.reduce_sum( 1/deltah**2 * (deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    def exactsol(self, x, y):
        return np.exp(-self.alpha*((x-self.xc)**2+(y-self.yc)**2)) + np.sin(np.pi * x)

    def tfexactsol(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2))+ tf.sin(np.pi * x[:,0])

    def f(self, x):
        return -4*self.alpha*self.tfexactsol(self.x) + 4*self.alpha**2*self.tfexactsol(self.x)* \
                                                       ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2) - np.pi**2 * tf.sin(np.pi * x[:,0])

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

        for _ in range(5):
            _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

        X = np.random.rand(self.batch_size, 2)
        # if i>50:
        X = np.concatenate([X,rectspace(0.4,0.5,0.4,0.5,5)], axis=0)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})

        if i % 10 == 0:
            print("Iteration={}, bloss = {}, loss= {}".format(i, bloss, loss))


class ProblemBLSingularity_BD(NNPDE2):
    def __init__(self, batch_size, N, refn):
        self.alpha = 0.6
        NNPDE2.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y):
        return y**0.6

    def tfexactsol(self, x):
        return tf.pow(x[:,1],0.6)

    def f(self, x):
        return self.alpha*(self.alpha-1)*x[:,1]**(self.alpha-2)

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        delta = tf.clip_by_value(delta, -1e2, 1e2)
        deltah = tf.clip_by_value(deltah, -1e2, 1e2)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

class ProblemPeak(NNPDE):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE.__init__(self,batch_size, N, refn)

    # data to be modified
    def exactsol(self,x,y):
        return np.exp(-1000 * ((x - self.xc) ** 2 + (y - self.yc) ** 2))

    def A(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2)) +tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])



    def tfexactsol(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2))

    def f(self, x):
        return -4*self.alpha*self.tfexactsol(self.x) + 4*self.alpha**2*self.tfexactsol(self.x)* \
                                                       ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2)

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        X = np.random.rand(self.batch_size, 2)
        # X = np.concatenate([X, rectspace(0.4, 0.5, 0.4, 0.5, 5)], axis=0)
        _, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
        if i % 10 == 0:
            print("Iteration={}, loss= {}".format(i, loss))



class ProblemBLSingularity(NNPDE):
    # data to be modified
    def exactsol(self,x,y):
        return x**0.6

    def A(self, x):
        return x[:,0]**0.6+tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return 0.6*(0.6-1)*x[:,0]**(0.6-2)




# # dir = 'p1'
# # npde = Problem1_BD(64, 3, 50) # works very well
# # with tf.Session() as sess:
# #     sess.run(npde.init)
# #     for i in range(10000):
# #         if( i>1000 ):
# #             break
# #         npde.train(sess, i)
# #         if i%10==0:
# #             npde.visualize(sess, False, i=i, savefig=dir)
#
# dir = 'p2'
# npde = ProblemPeak_BD(64, 3, 50) # works very well
# # npde.plot_exactsol()
# # plt.show()
# # exit(0)
# with tf.Session() as sess:
#     sess.run(npde.init)
#     for i in range(10000):
#         if( i>1000 ):
#             break
#         npde.train(sess, i)
#         if i%10==0:
#             npde.visualize(sess, True, i=i, savefig=dir)
#
# # dir = 'p3'
# # npde = ProblemBLSingularity_BD(64, 3, 50) # works very well
# # with tf.Session() as sess:
# #     sess.run(npde.init)
# #     for i in range(10000):
# #         if( i>1000 ):
# #             break
# #         npde.train(sess, i)
# #         if i%50==0:
# #             npde.visualize(sess, False, i=i, savefig=dir)
#
#
# plt.close('all')
# plt.semilogy(npde.rbloss)
# plt.xlabel('Iteration')
# plt.ylabel('$L_b$')
# plt.savefig(dir + '/lb.png')
#
# plt.close('all')
# plt.semilogy(npde.rloss)
# plt.xlabel('Iteration')
# plt.ylabel('$L_i$')
# plt.savefig(dir + '/li.png')
#
# plt.close('all')
# plt.semilogy(npde.rl2)
# plt.xlabel('Iteration')
# plt.ylabel('$||u-u_h||_2$')
# plt.savefig(dir + '/l2.png')




## High dimension test
class HighDimension(NNPDE_ND):
    def tfexactsol(self,x):
        return tf.reduce_prod(tf.sin(np.pi * x), axis=1)

    def exactsol(self, x):
        return np.prod(np.sin(np.pi * x), axis=1)

    def f(self, x):
        return -np.pi**2*self.d* self.tfexactsol(x)

    def B(self, x):
        return tf.reduce_prod(x*(1-x),axis=1)

    def train(self, sess, i):
        self.rbloss = []
        self.rloss = []
        self.rl2 = []
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        bX = np.random.rand(2*self.d*self.batch_size, self.d)
        for j in range(self.d):
            bX[2*j*self.batch_size:(2*j+1)*self.batch_size, j] = 1.0
            bX[(2 * j+1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 0.0

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

        X = np.random.rand(self.batch_size, self.d)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})


        # ######### record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        self.rl2.append( self.compute_L2(sess, self.X_test) )
        # ######### record loss ############
        # loss = np.inf

        if i % 10 == 0:
        	pass
            #print("Iteration={}, bloss = {}, loss= {}, L2={}".format(i, bloss, loss, self.rl2[-1]))

##

# 1. tunable parameters: batch-size, number of layers
# 2. for different dimension, record bloss, loss, L2
# 3. generate meaningful plots

dir = './high/'


layers = [3]

from matplotlib import pyplot as plt
# import pickle
for dim in [3]:

    rblossFull = []
    rlossFull = []
    rl2Full = []
    for layer in layers:
        #print(layer)
        npde = HighDimension(64, layer, dim) # batch-size, number of layers, dimension
        rbloss = []
        rloss = []
        rl2 = []
        with tf.Session() as sess:

            sess.run(npde.init)
            for i in range(10000):
                if( i % 1000 == 0):
                    print('Now step'+ str(i))
                npde.train(sess, i)
                rbloss.append(npde.rbloss)
                rloss.append(npde.rloss)
                rl2.append(npde.rl2)
        #print(rbloss)
        #npde.visualize(sess, False, i=i, savefig=dir)
        tf.reset_default_graph()
        rblossFull.append(rbloss)
        rlossFull.append(rloss)
        rl2Full.append(rl2)

    with open(dir + str(dim) + '.txt', 'w') as file:
        for l in rblossFull:
            for item in l:
                file.write("%s," % item)
            file.write("\t")
        file.write("\n")
        for l in rlossFull:
            for item in l:
                file.write("%s," % item)
            file.write("\t")
        file.write("\n")
        for l in rl2Full:
            for item in l:
                file.write("%s," % item)
            file.write("\t")


    plt.close('all')
    plt.figure(1)
    for i in range(len(rblossFull)):
        plt.semilogy(rblossFull[i],label="nLayer=%d" %layers[i])
    plt.xlabel('Iteration')
    plt.ylabel('$L_b$')
    plt.legend()
    plt.savefig(dir + str(dim) + '_' 'lb.png')
    
    plt.figure(2)
    for i in range(len(rlossFull)):
        plt.semilogy(rlossFull[i],label="nLayer=%d"%layers[i])
    plt.xlabel('Iteration')
    plt.ylabel('$L_i$')
    plt.legend()
    plt.savefig(dir + str(dim) + '_' 'li.png')
    
    plt.figure(3)
    for i in range(len(rl2Full)):
        plt.semilogy(rl2Full[i],label="nLayer=%d"%layers[i])
    plt.xlabel('Iteration')
    plt.ylabel('$L_2 = ||u-u_h||_2$')
    plt.legend()
    plt.savefig(dir + str(dim) + '_' 'l2.png')
    plt.show()
    # /Users/shuyiyin/Dropbox/CS230_project/mcode
    # python nnpde.py