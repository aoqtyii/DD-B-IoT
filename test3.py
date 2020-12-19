# _*_ coding utf-8 _*_
"""
@File : test3.py
@Author: yxwang
@Date : 2020/6/14
@Desc :
"""
from collections import Counter

import numpy as np  # NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt  # For plotting
from scipy.optimize import minimize  # For optimizing
from scipy import integrate  # For integrating


def plot1(s):
    plt.close('all')  # 关闭所有绘图

    # 模拟的窗口为矩形
    xMin = -1
    xMax = 1
    yMin = -1
    yMax = 1
    xDelta = xMax - xMin
    yDelta = yMax - yMin
    areaTotal = xDelta * yDelta

    numbPointsRetained = 21

    # s = 0.8  # 比例参数

    # 点过程参数
    def fun_lambda(x, y):
        return 10 * np.exp(-(x ** 2 + y ** 2) / s ** 2)  # 强度函数

    # START -- 找到最大的强度 -- START
    # 对任意的强度函数lambda，找到矩形区域内最大的lambda
    def fun_Neg(x):
        m = -fun_lambda(x[0], x[1])
        return m  # 负的lambda

    xy0 = [(xMin + xMax) / 2, (yMin + yMax) / 2]  # 初始值(中心位置值)
    # 找到最大的lambda值
    resultsOpt = minimize(fun_Neg, xy0, bounds=((xMin, xMax), (yMin, yMax)))
    lambdaNegMin = resultsOpt.fun  # 通过minimize函数找到最小值
    lambdaMax = -lambdaNegMin

    # END -- 找到最大的lambda值 -- END

    # 定义 thinning prob 函数
    def fun_p(x, y):
        return fun_lambda(x, y) / lambdaMax

    # fun_p = lambda x, y: fun_lambda(x, y) / lambdaMax;
    # numbPointsRetained = np.zeros(numbSim)

    # 模拟泊松点过程
    while True:
        numbPoints = np.random.poisson(areaTotal * lambdaMax)  # 泊松分布的点个数
        if numbPoints > numbPointsRetained:
            xx = np.random.uniform(0, xDelta, (numbPoints, 1)) + xMin  # 泊松分布点坐标的x值
            yy = np.random.uniform(0, yDelta, (numbPoints, 1)) + yMin  # 泊松分布点坐标的y值

            # 计算空间独立的thinning prob.
            p = fun_p(xx, yy)

            # 为thinning生成伯努利变量
            booleRetained = np.random.uniform(0, 1, (numbPoints, 1)) < p  # 被保留的点

            c = Counter(booleRetained.T[0]).get(True)

            # 保留点的坐标
            xxRetained = xx[booleRetained]
            yyRetained = yy[booleRetained]
            break

    def plot():
        # 绘图
        plt.scatter(xxRetained, yyRetained, edgecolor='b', facecolor='none', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        # run empirical test on number of points generated
        # if numbSim >= 10:
        #     # total mean measure (average number of points)
        #     LambdaNumerical = integrate.dblquad(fun_lambda, xMin, xMax, lambda x: yMin, lambda y: yMax)[0]
        #     # Test: as numbSim increases, numbPointsMean converges to LambdaNumerical
        #     numbPointsMean = np.mean(numbPointsRetained)
        #     # Test: as numbSim increases, numbPointsVar converges to LambdaNumerical
        #     numbPointsVar = np.var(numbPointsRetained)

    plot()
