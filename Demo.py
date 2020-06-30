from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier as XGBC
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error as MSE
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
from time import time
from sklearn.metrics  import roc_curve,auc
import datetime
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def processdata(data):
    sp = [] #记录哪些时是字符型的列
    for col in range(41):
         if data[col].dtypes == data[41].dtypes:
            sp.append(col)
            le = LabelEncoder()
            le.fit(data[col])
            data[col] = le.transform(data[col])
            # one-code编码  max(data[2])
            df_processed = pd.get_dummies(data[col],prefix_sep="_")
            # 转化后的属性
            df_processed.columns = [str(col)+'.' + str(int(i))  \
            for i in range(df_processed.shape[1])]
            data = pd.concat([data, df_processed], axis=1)
    #实现标签二分类处理,存储标签
    data[41] = np.where(data[41] == 'normal', 1, 0)
    label = data[41]
    # 去除替换后的字符型数据，同时把网络受攻击程度42列去掉,
    # 得到样本特征空间
    data = data.drop(data.columns[sp+[41,42]], axis=1)  # axis=1，试图指定列
    datas = pd.concat([data, label], axis=1)
    return datas




def plot_learning_curve(estimator, title, X, y,
                        ax=None,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                , shuffle=True
                                                , cv=cv
                                                , random_state=420
                                                , n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g", label="Test score")
    ax.legend(loc="best")
    return ax


def sample_influence(xt,yt):
    # 训练集样本的大小影响
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 交叉验证模式
    plot_learning_curve(XGBC(n_estimators=100, random_state=42), \
                        "XGBC", xt, yt, ax=None, cv=cv)
    plt.show()

def ntrees_estimators(Xtrain,Ytrain):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 交叉验证模式
    axisx = range(120, 200,20)
    rs = []
    var = []
    ge = []
    for i in axisx:
        print(i)
        reg = XGBC(n_estimators=i, random_state=420,
                   subsample=0.8,
                   learning_rate=0.2)
        cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
        rs.append(cvresult.mean())
        var.append(cvresult.var())
        ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
    print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
    print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
    print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
    rs = np.array(rs)
    var = np.array(var)
    plt.plot(axisx, rs, c="black", label="XGB")
    print(rs)
    # 添加方差线
    plt.plot(axisx, rs + 10*var, c="red", linestyle='-.')
    plt.plot(axisx, rs - 10*var, c="red", linestyle='-.')
    plt.legend()
    plt.show()

def gamasolve(Xtrain,Ytrain):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)  # 交叉验证模式
    axisx = range(0,6)
    rs = []
    var = []
    ge = []
    for i in axisx:
        print(i)
        reg = XGBC(n_estimators=140, random_state=420,
                   subsample=0.8,
                   max_depth=2,
                   gamma=0,
                   learning_rate=0.2,
                   reg_lambda=2,
                   seed=0,
                   booster='gbtree'
                   )
        cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
        rs.append(cvresult.mean())
        var.append(cvresult.var())
        ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
    print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
    print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
    print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
    rs = np.array(rs)
    var = np.array(var)
    plt.plot(axisx, rs, c="black", label="XGB")
    # 添加方差线
    plt.plot(axisx, rs + var, c="red", linestyle='-.')
    plt.plot(axisx, rs - var, c="red", linestyle='-.')
    plt.legend()
    plt.show()


def ytest(xt, yt, x_test, y_test):
    # xt, yt, x_test, y_test分别表示训练集、训练集标签、测试集、测试标签
    time0 = time()
    xgbt = XGBC(n_estimators=140,
                subsample=0.8,
                learning_rate=0.2,
                random_state=20,
                gamma=0,
                reg_lambda= 2,
                max_depth = 2,
                object = 'binary:logistic',
                # min_child_weight=2,
                seed= 0,
                booster='gbtree')
    y_predict = xgbt.fit(xt, yt).predict(x_test)
    x_predict = xgbt.fit(xt, yt).predict(xt)
    plot_importance(xgbt)
    print('训练集 :',accuracy_score(x_predict,yt))
    print('测试集 :',accuracy_score(y_predict, y_test))
    print('测试r2指标 :',r2_score(y_predict, y_test))
    print('测试MSE指标 :',MSE(y_predict, y_test))
    # print('运行时间 :',time() - time0)
    plt.show()
    y_score = xgbt.fit(xt, yt).predict_proba(x_test)
    fpr,tpr,threshold=roc_curve(y_test,y_score[:, 1])
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr, tpr, color='darkorange',
    lw=2, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
if __name__ == '__main__':
    '''
    # KDDtrain_data = pd.read_csv("xgboost\\DATA\\KDDTrain+.txt" ,
    #                             names = list(range(43)), header=0)
    # pickle.dump(KDDtrain_data, open("KDDtrain_data.dat","wb"))
    #
    # KDDTest = pd.read_csv("xgboost\\DATA\\KDDTest+.txt",
    #                       names=list(range(43)), header=0 )
    # pickle.dump(KDDTest, open("KDDTest.dat","wb"))
    # 加载文件
    '''
    KDDtrain_data = pickle.load(open("KDDtrain_data.dat", "rb"))
    KDDTest = pickle.load(open("KDDTest.dat", "rb"))
    dataxtrain = processdata(KDDtrain_data)
    dataxtext = processdata(KDDTest)
    # 训练集、测试集划分
    xt, yt, x_test, y_test = dataxtrain.iloc[:, 0:41], \
    dataxtrain.iloc[:, 41], dataxtext.iloc[:, 0:41], dataxtext.iloc[:, 41]
    # sample_influence(xt, yt)
    # ntrees_estimators(xt,yt)   #弱学习器的个数
    # gamasolve(xt,yt)
    ytest(xt, yt, x_test, y_test)




