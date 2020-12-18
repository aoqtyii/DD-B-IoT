# _*_ coding utf-8 _*_
"""
@File : geographical_coordination.py
@Author: yxwang
@Date : 2020/4/29
@Desc :
"""

from scipy.optimize import minimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


# 1. 2D inhomogeneous PPP分布
# 2. 对于非齐次泊松点过程的模拟，首先模拟一个均匀的泊松点过程，然后根据确定性函数适当地变换这些点
# 3. 模拟联合分布的随机变量的标准方法是使用  马尔可夫链蒙特卡洛；应用MCMC方法就是简单地将随机点处理操作重复应用于所有点
#    将使用基于Thinning的通用但更简单的方法(Thinning是模拟非均匀泊松点过程的最简单，最通用的方法)

# plt.close('all')

class geographical_coordination:
    def __init__(self, xMin, xMax, yMin, yMax, n_of_nodes, num_Sim=1, s=0.7, ):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.xDelta = xMax - xMin
        self.yDelta = yMax - yMin
        self.xMin_norm = -1
        self.xMax_norm = 1
        self.yMin_norm = -1
        self.yMax_norm = 1
        self.xDelta_norm = self.xMax_norm - self.xMin_norm
        self.yDelta_norm = self.yMax_norm - self.yMin_norm
        self.areaTotal_norm = self.xDelta_norm * self.yDelta_norm
        self.areaTotal = self.xDelta * self.yDelta

        self.num_Sim = num_Sim
        self.s = s

        self.n_of_nodes = n_of_nodes

        self.resultsOpt = None
        self.lambdaNegMin = None
        self.lambdaMax = None
        self.numbPointsRetained = None
        self.numbPoints = None

        self.xxRetained = []
        self.yyRetained = []
        self.xxThinned = []
        self.yyThinned = []
        self.xxRetained_for_n = []
        self.yyRetained_for_n = []

    # point process params
    def fun_lambda(self, x, y):
        # intensity function
        return 100 * np.exp(-(x ** 2 + y ** 2) / self.s ** 2)

    # define thinning probability function
    def fun_p(self, x, y):
        return self.fun_lambda(x, y) / self.lambdaMax

    def fun_neg(self, x):
        # negative of lambda
        # fun_neg = lambda x: -fun_lambda(x[0], x[1])
        return -self.fun_lambda(x[0], x[1])

    def geographical_coordinates(self):
        # initial value(ie center)
        # xy0 = [(self.xMin + self.xMax) / 2, (self.yMin + self.yMax) / 2]
        xy0 = [(self.xMin_norm + self.xMax_norm) / 2, (self.yMin_norm + self.yMax_norm) / 2]

        # Find largest lambda value
        self.resultsOpt = minimize(self.fun_neg, xy0, bounds=((self.xMin_norm, self.xMax_norm), (self.yMin_norm, self.yMax_norm)))
        self.lambdaNegMin = self.resultsOpt.fun  # retrieve minimum value found by minimize
        self.lambdaMax = -self.lambdaNegMin

        # for collecting statistics -- set num_Sim=1 for one simulation
        self.numbPointsRetained = np.zeros(self.num_Sim)

        for ii in range(self.num_Sim):
            # Simulate a Poisson point process
            # Poisson number of points
            self.numbPoints = np.random.poisson(self.areaTotal_norm * self.lambdaMax)
            # get the number of Poisson points, numbPoints > n_of_IIot points

            # x coordinates of Poisson points
            # y coordinates of Poisson points
            xx = np.random.uniform(0, self.xDelta_norm, (self.numbPoints, 1)) + self.xMin_norm
            yy = np.random.uniform(0, self.yDelta_norm, (self.numbPoints, 1)) + self.yMin_norm

            # calculate spatially-dependent thinning probabilities
            p = self.fun_p(xx, yy)

            # Generate Bernoulli variables (ie coin flips) for thinning
            # points to be retained
            # Spatially independent thinning
            booleRetained = np.random.uniform(0, 1, (self.numbPoints, 1)) < p
            # index_of_Retained = np.argwhere(booleRetained == True)
            # booleThinned = ~booleRetained

            # assert sum(booleRetained == True).__int__() > self.n_of_nodes

            # x/y locations of retained points
            self.xxRetained = xx[booleRetained]
            self.yyRetained = yy[booleRetained]
            # self.xxThinned = xx[booleThinned]
            # self.yyThinned = yy[booleThinned]
            # for index in index_of_Retained:
            #     self.xxRetained.append(xx[index[0]][index[1]])
            #     self.yyRetained.append(yy[index[0]][index[1]])

            # self.numbPointsRetained[ii] = self.xxRetained.size

            for i in range(self.n_of_nodes):
                # self.xxRetained_for_n[i] = self.xxRetained[i]/self.xDelta_norm*self.xDelta
                # self.yyRetained_for_n[i] = self.yyRetained[i]/self.yDelta_norm*self.yDelta
                self.xxRetained_for_n.append(self.xxRetained[i]/self.xDelta_norm*self.xDelta)
                self.yyRetained_for_n.append(self.yyRetained[i]/self.yDelta_norm*self.yDelta)

        # return n coordinates of IIoT nodes.
        return np.array(self.xxRetained_for_n), np.array(self.yyRetained_for_n), self.s
