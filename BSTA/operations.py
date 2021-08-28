import numpy as np
from random import choice, sample,randint
from fitness import fitness

"""
函数描述： 从一堆解state中找到最好的解，本来数据和标签最好在评价函数哪儿，但是，如果，每次评价都读文件的话，浪费时间，所以将其作为参数在算法中传递
参数： 
    state  - 解矩阵，每一行是一个二值解向量
    SE     - 解个数
    data   - 训练用的数据  
    label  - 对应的标签
"""
def getBest(state, fbest, SE, data, label):
    max_index = 0
    for i in range(SE):
        acc = fitness(state[i, :], data, label)
        if acc > fbest:
            fbest = acc
            max_index = i
    return fbest, state[max_index, :]

'''
函数描述：通过交换两个维度（index0 和 index1）的取值,产生 SE 个新解,为了减少冗余的0 交换0，1交换1 ，所以分别随机从0 和1 中选一个来交换
param
    oldBest - 当前最优值，历史最优值
    SE      - 产生解的个数
    dim     - 维度
    data    - 训练用的数据  
    label   - 对应的标签
return:
    fbest   - 最优值
    newBest - 最优解
'''
def op_swap(oldBest, fbest, SE, dim, data, label):
    state = np.zeros((SE, dim), dtype=int)

    templist0 = np.where(oldBest == 0)[0]
    templist1 = np.where(oldBest == 1)[0]
    for i in range(SE):
        temp = oldBest.copy()
        index0 = choice(templist0)
        index1 = choice(templist1)
        temp[index0], temp[index1] = temp[index1], temp[index0]   # 交叉
        state[i, :] = temp
    fbest, newBest = getBest(state, fbest, SE, data, label)
    return fbest, newBest

'''
函数描述：随机让一个维度变化，0变1,1变0  best[index] = ~best[index] 
param
    oldBest - 当前最优值，历史最优值
    SE      - 产生解的个数
    dim     - 维度
    data    - 训练用的数据  
    label   - 对应的标签
return:
    fbest   - 最优值
    newBest - 最优解
'''
def op_substitute(oldBest, fbest, SE, dim, data, label):
    state = np.zeros((SE, dim), dtype=int)
    for i in range(SE):
        temp = oldBest.copy()
        index = randint(0,dim-1)   # 使用的事random 库中的，所以事包含后面的上界的，如果你使用numpy 里面的random.randint，则不用减一
        if temp[index] == 0:
            temp[index] = 1
        else:
            temp[index] = 0
        state[i, :] = temp
    fbest, newBest = getBest(state, fbest, SE, data, label)
    return fbest, newBest

'''
函数描述：通过从start开始循环左移 start-end 位来产生新解  在[start，end] 区间循环座左移 一位   best[start，end] -> best[start+1,start+2, ..., end,start]
param
    oldBest - 当前最优值，历史最优值
    SE      - 产生解的个数
    dim     - 维度
    data    - 训练用的数据  
    label   - 对应的标签
return:
    fbest   - 最优值
    newBest - 最优解
'''
def op_shift(oldBest, fbest, SE, dim, data, label):
    state = np.zeros((SE, dim), dtype=int)
    for i in range(SE):
        temp = oldBest.copy()
        """产生随机区间"""
        index = sample(list(range(dim)), 2)
        start = min(index)
        end = max(index)

        """循环左移一位"""
        temp_start = temp[start]
        temp[start:end] = temp[start+1:end+1]
        temp[end] = temp_start

        state[i, :] = temp
    fbest, newBest = getBest(state, fbest, SE, data, label)
    return fbest, newBest

'''
函数描述：在[start，end] 区间对称变换 即  best[start，end] -> best[end,start]
param
    oldBest - 当前最优值，历史最优值
    SE      - 产生解的个数
    dim     - 维度
    data    - 训练用的数据  
    label   - 对应的标签
return:
    fbest   - 最优值
    newBest - 最优解
'''
def op_symmetry(oldBest, fbest, SE, dim, data, label):
    state = np.zeros((SE, dim), dtype=int)
    for i in range(SE):
        temp = oldBest.copy()
        """产生随机区间"""
        index = sample(list(range(dim)), 2)
        start = min(index)
        end = max(index)

        """对称变换"""
        temp[start:end+1] = temp[list(range(end,start-1,-1))]

        state[i, :] = temp
    fbest, newBest = getBest(state, fbest, SE, data, label)
    return fbest, newBest

def initial(dim):
    index = sample(list(range(dim)),dim//2)
    init_state = np.zeros(dim,dtype=int)
    init_state[index] = 1
    return init_state




