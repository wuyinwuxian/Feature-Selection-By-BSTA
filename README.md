# Feature-Selection-By-BSTA
特征选择建模成单目标优化问题，目标函数是特征子集在学习器（SVM）上的分类准确率，并将解（特征子集）用0，1编码，即 0 代表该特征不被选中，1代表被选中。因此，特征选择问题变成了一个离散二值优化问题，然后使用离散状态转移算法来做特征选择
