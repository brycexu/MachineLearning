A1-1-KNN
   
如何加载CIFAR-10数据，请见 Eclipse 的 Tensorflow 文档

Training data shape: (50000,32,32,3)
   
KNN（K Nearest Neighbors）

1. 计算距离
输入：X(num_test,D) X_train(num_train,D)
输出：dists(num_test,num_train)
dists[i][j]：测试样本i到训练样本j的距离
>>>
num_test = X.shape[0]
num_train = X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    for j in range(num_train):
        dists[i][j]=np.sqrt(np.sum((X[i,:]-self.X_train[j,:])**2))
>>>

2. 预测标签
输入：dists(num_test,num_train)
输出：y_pred(num_test,1)
y_pred[i][1]：测试样本i的预测标签
>>>
num_test = dists.shape[0]
y_pred = np.zeros(num_test)
for i in range(num_test):
    # 对于测试样本i来说
    # np.argsort(dists[i])[0:k]:选取距离测试样本i最近的k个训练样本的index
    self.y_train[np.argsort(dists[i])[0:k]]:提出这k个训练样本对应的标签
    closest_y = self.y_train[np.argsort(dists[i])[0:k]]
    # 选取这k个标签中出现最多的作为测试样本i的预测标签
    y_pred[i] = np.argmax(np.bincount(closest_y))
>>>


Cross-validation 交叉验证

之前，我们是有(X_train,y_train)和(X_test,y_test)，假设数量为1000和200
现在，我们把train再拆，
拆成800个train和200个val，这200个val有5种选法

>>>
# 将X_train和y_train先拆分成5份
X_train_folds = np.array_split(X_train,5)
y_train_folds = np.array_split(y_train,5)
# 交叉验证
# X_val和y_val选取第i个包
# X_train和y_train将其余的4个包拼接起来
for i in range(5): 
    X_val = X_train_folds[i]
    y_val = y_train_folds[i]
    for j in range(5-1):
            X_train_new.append(X_train_folds[(i+1+j)%5])
            y_train_new.append(y_train_folds[(i+1+j)%5])
            X_train_new = np.concatenate(list(X_train_new))
            y_train_new = np.concatenate(list(y_train_new)
            ...
>>>



