# _*_ coding utf-8 _*_
"""
@File : test1.py
@Author: yxwang
@Date : 2020/6/10
@Desc :
"""
import collections

import numpy as np

# def get_xy():
#     x = []
#     y = []
#
#     for a in range(10):
#         x.append(a)
#
#     for b in range(10):
#         y.append(10 - b)
#
#     c = [1] * 10
#
#     return x, y, c
#
#
# xy = ((get_xy()[0][i], get_xy()[1][i]) for i in range(len(get_xy()[0])))
#
# while True:
#     try:
#         print(xy.__next__())
#     except:
#         break
from gym import spaces

# action = spaces.Box(low=-1.0, high=2.0, shape=(3, 4))
#
# for i in range(3):
#     print(action.sample())

import numpy as np
# n_of_nodes = 3
# observation_space = spaces.Dict({
#         'throughout': spaces.Box(low=0, high=np.inf, shape=(1,)),
#         'G_gamma': spaces.Box(low=0, high=1, shape=(1,)),
#         'G_lambda': spaces.Box(low=0, high=1, shape=(1,)),
#         # 'geographical_of_nodes': spaces.Box(low=np.array([-1000, -1000]), high=np.array([-1000, -1000])),
#         'computing_capacity_of_IIoT_nodes': spaces.Box(low=10, high=30, shape=(n_of_nodes,)),
#         'transmission_rate': spaces.Box(low=10, high=100, shape=(n_of_nodes,n_of_nodes)),
#         'coef_of_security': spaces.Box(low=0, high=1, shape=(1,))
# })
#
# print(observation_space.sample())

# action_space = spaces.Dict({
#             # 对所有的节点进行判断，值为1的节点是block_producer
#             # 选择共识算法，将不同的共识算法记做0,1,2
#             'no_block_producer': spaces.MultiBinary(5),
#             'no_consensus_algorithm': spaces.Discrete(3),
#             'block_size': spaces.Box(low=0, high=10, shape=(1,)),
#             'block_interval': spaces.Box(low=0, high=10, shape=(1,))
# })
#
# sample = action_space.sample()['no_block_producer']
# print(sample)
# print(collections.Counter(sample).most_common())
# print(np.count_nonzero(sample))
# print(collections.Counter(sample).most_common()[1])
# print(collections.Counter(sample).most_common()[1][1])

# from geographical_coordination import geographical_coordination
#
# geo = geographical_coordination(-10, 10, -10, 10, 2)
# xx = geo.geographical_coordinates()[0]
# yy = geo.geographical_coordinates()[1]
# print(geo.geographical_coordinates())
# print(geo.numbPointsRetained)
# print(geo.numbPoints)

import numpy as np

# booleRetained = np.random.uniform(0, 1, (10, 1)) < 0.6
# print(booleRetained)
# print(sum(booleRetained == True).__int__())

# n_of_nodes = 5
# xxRetained = []
# yyRetained = []
# xxRetained_for_n = []
# yyRetained_for_n = []
#
# for i in range(10):
#     xxRetained.append(np.random.randint(10))
#     yyRetained.append(np.random.randint(10))
#
# print(xxRetained)
# print(yyRetained)
#
# for i in range(n_of_nodes):
#     xxRetained_for_n.append(xxRetained[i])
#     yyRetained_for_n.append(yyRetained[i])
#
# print(xxRetained_for_n)
# print(yyRetained_for_n)

# x = [1, 2, 2, 3, 4, 2, 4, 1, 0]
# y = []
# index = np.where(np.array(x) == 2)
# print(index)
#
# for i in index[0].tolist():
#     y.append(x[i])
#
# print(y)

# n_of_nodes = 5
# observation_space = spaces.Dict({
#             'throughout': spaces.Box(low=0, high=np.inf, shape=(1,)),
#             'G_gamma': spaces.Box(low=0, high=1, shape=(1,)),
#             'G_lambda': spaces.Box(low=0, high=1, shape=(1,)),
#             # 'geographical_of_nodes': spaces.Box(low=np.array([-1000, -1000]), high=np.array([-1000, -1000])),
#             'computing_capacity_of_IIoT_nodes': spaces.Box(low=10, high=30, shape=(n_of_nodes,)),
#             'transmission_rate': spaces.Box(low=10, high=100, shape=(n_of_nodes,n_of_nodes)),
#             'coef_of_security': spaces.Box(low=0, high=1, shape=(1,))
# })
#
# state = observation_space.sample()
# G_gamma, G_lambda, coef_of_security, computing_capacity_of_IIoT_nodes, \
#         throughout, transmission_rate = (state[k] for k in state.keys())
#
# print(state)
# print(G_gamma, G_lambda, coef_of_security, computing_capacity_of_IIoT_nodes, \
#         throughout, transmission_rate)


# b = np.random.uniform(0, 1, (10, 1)) < 0.3
# print(b)
# index = np.argwhere(b == True)
#
# print(index)


# a = [1, 2, 3, 4, 5]
# print(type(a))
#
# b = np.array(a)
# print(b)
# print(type(b))
# print(b.shape)

# a = [3, 5, 1, 7, 2]
#
# b = np.random.choice(a)
# print(b)

# a = np.ones(3, )*10
# print(a)


# print(np.floor(3 / 2))

# a = np.ones(10, )
# print(a)
#
# b = np.random.randint(low=1, high=10, size=10)
# print(b)
#
# c = min(a / b)
# print(c)


# 根据client, primary, replica节点需要产生和验证的MAC，计算节点所需的计算能力
# a = np.random.randint(low=1, high=10, size=(10, 10))
# print(a)
#
# b = 100
#
# c = max(b / a[3][j] for j in (0, 3))
# print(c)

# a = np.random.randint(low=1, high=100, size=(10, 10))
# print(a)
#
# b = (a[i][j] for i in range(10) for j in range(10) if i != j and i != 0)
#
# n = 0
# while True:
#     print(b.__next__(), end=',')

# a = np.ones((10, 1))
# b = np.random.randint(low=1, high=100, size=(10, 1))
# print(a)
# print(b)
#
# c = max(b[i] / a[i] for i in range(10) if i not in [1, 2, 3, 4, 5, 6, 7])
# print(c)


tr = np.random.randint(1, 20, (15, 10))
print(tr)
print(type(tr))
l = [1, 3, 5]
print([tr[i][j] for i in l for j in l if i != 0 and j != 0 and i != j])
