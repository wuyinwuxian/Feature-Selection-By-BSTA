import pandas as pd
from operations import *

'''
函数描述：通过 BSTA 来做特征选择，评价标准是在 SVM 学习器上的分类准确率
param
    data     - 训练用的数据  
    label    - 对应的标签
    SE       - 产生解的个数
    dim      - 维度
    max_iter - 最大迭代次数
return:
    fbest    - 最优值
    newBest  - 最优解
'''
def BSTA(data,label,SE,dim,max_iter):
    newBest = initial(dim)
    fbest = fitness(newBest, data, label)
    for i in range(max_iter):
        fbest, newBest = op_swap(newBest, fbest,SE, dim, data, label)
        fbest, newBest = op_shift(newBest,fbest, SE, dim, data, label)
        fbest, newBest = op_substitute(newBest,fbest, SE, dim, data, label)
        fbest, newBest = op_symmetry(newBest,fbest, SE, dim, data, label)

    return fbest, newBest

if __name__ =="__main__":
    max_iter = 10
    SE = 50        # 解个数

    """导入数据"""
    df = pd.read_csv('parkinsons.csv', header=None)
    dim = df.values.shape[1] - 1
    data = df.values[:, 0:dim]
    label = df.values[:, -1]

    fbest, newBest = BSTA(data,label,SE,dim,max_iter)