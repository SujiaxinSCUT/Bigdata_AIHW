import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #####################################
    #
    # 本次实验获取kaggle房价预测数据集，通过线性回归对房价进行预测
    # 数据集地址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
    #
    #####################################


    # 获取训练集和测试集
    train_path = os.path.join('~/s3data/dataset', 'house_price', 'train.csv')
    test_path = os.path.join('~/s3data/dataset', 'house_price', 'test.csv')

    # 创建结果集路径
    result_path = os.path.join('~/s3data/dataset', 'house_price', 'result.csv')

    # 读取训练集数据
    # train_set = pd.read_csv(train_path)
    # test_set = pd.read_csv(test_path)
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')

    #  将无关索引项删去并保存测试集索引项
    train_set.drop('Id', axis=1, inplace=True)

    # # 显示所有列
    # pd.set_option('display.max_columns', None)
    #
    # # 前十条示例数据
    # print(train_set.head(10))

    #查看数据集大小
    print("=============The original dataset's shape================")
    print(f"train_set shape:",train_set.shape)
    print(f"test_set shape:",test_set.shape)

    #####################################
    #
    # # 去除重复行
    #
    #####################################

    train_set.drop_duplicates(keep="first",inplace=True)
    test_set.drop_duplicates(keep="first",inplace=True)
    # 查看处理结果
    print("=============The dataset's shape after deleting duplicate rows================")
    print(f"train_set shape:",train_set.shape)
    print(f"test_set shape:",test_set.shape)

    #####################################
    #
    # 处理缺失值
    # # 将缺失率大于15%的列直接删去
    # # 将剩下的含有缺失值的按行删去
    #
    #####################################

    train_nan_count = train_set.shape[0] - train_set.count().sort_values()
    # 获取缺失行比率
    train_nan_rate = train_nan_count / train_set.shape[0]
    # 将缺失行数量和缺失行比率生成dataframe处理
    train_nan_det = pd.concat([train_nan_count, train_nan_rate], axis=1, keys=['nan_count', 'nan_rate'])
    # 截取缺失值大于15%的行的索引，该索引即为对应的列名
    train_more_than_15per_index = train_nan_det[train_nan_det['nan_rate'] > 0.15].index
    # 从训练集和测试集中删去缺失严重的列
    for col in train_more_than_15per_index:
        train_set.drop(col, axis=1, inplace=True)
        test_set.drop(col, axis=1, inplace=True)

    # 删除含有缺失值的行，大概会损失7%左右的数据量，应当是可以接受的
    train_set.dropna(inplace=True)
    test_set.dropna(inplace=True)

    print("=============The dataset's shape after deleting the nan values================")
    print(f"train_set shape:",train_set.shape)
    print(f"test_set shape:",test_set.shape)

    #####################################
    #
    # # 根据实际业务场景，删除列强相关项
    #
    #####################################

    train_set.drop('GarageYrBlt', axis=1, inplace=True)
    train_set.drop('GarageFinish', axis=1, inplace=True)
    train_set.drop('GarageCond', axis=1, inplace=True)
    train_set.drop('GarageQual', axis=1, inplace=True)
    train_set.drop('GarageType', axis=1, inplace=True)
    test_set.drop('GarageYrBlt', axis=1, inplace=True)
    test_set.drop('GarageFinish', axis=1, inplace=True)
    test_set.drop('GarageCond', axis=1, inplace=True)
    test_set.drop('GarageQual', axis=1, inplace=True)
    test_set.drop('GarageType', axis=1, inplace=True)

    train_set.drop('BsmtCond', axis=1, inplace=True)
    train_set.drop('BsmtExposure', axis=1, inplace=True)
    train_set.drop('BsmtQual', axis=1, inplace=True)
    train_set.drop('BsmtFinType2', axis=1, inplace=True)
    train_set.drop('BsmtFinType1', axis=1, inplace=True)
    test_set.drop('BsmtCond', axis=1, inplace=True)
    test_set.drop('BsmtExposure', axis=1, inplace=True)
    test_set.drop('BsmtQual', axis=1, inplace=True)
    test_set.drop('BsmtFinType2', axis=1, inplace=True)
    test_set.drop('BsmtFinType1', axis=1, inplace=True)
    
    train_set.drop('MasVnrType', axis=1, inplace=True)
    train_set.drop('MasVnrArea', axis=1, inplace=True)
    test_set.drop('MasVnrType', axis=1, inplace=True)
    test_set.drop('MasVnrArea', axis=1, inplace=True)

    print("=============The dataset's shape after deleting the non-relative values================")
    print(f"train_set shape:",train_set.shape)
    print(f"test_set shape:",test_set.shape)


    #####################################
    #
    # 处理训练集和测试集
    # # 将训练集分割出特征集和类集
    # # 对特征集和测试集进行独热编码
    #
    #####################################

    #分割训练集
    X = train_set.drop('SalePrice', axis=1)
    y = train_set.get('SalePrice')


    # 对特征集和测试集进行独热编码
    object_list = []
    for i in X.columns.values:
        if X[i].dtypes == 'object':
            # 非数据类型才进行独热编码
            object_list.append(X[i].name)
    # 对搜集的非数据类型进行独热编码
    X_object_set = pd.get_dummies(X[object_list])
    # 剔除独热编码的列
    X.drop(object_list, axis=1, inplace=True)
    # 将独热编码的列和原数据列合并
    X = X.join(X_object_set)
    # 对搜集的非数据类型进行独热编码
    test_object_set = pd.get_dummies(test_set[object_list])
    # 剔除独热编码的列
    test_set.drop(object_list, axis=1, inplace=True)
    # 顺便将无关项Id分离出来
    Id_set = test_set.get('Id')
    test_set.drop('Id', axis=1, inplace=True)
    # 将独热编码的列和原数据列合并
    test_set = test_set.join(test_object_set)

    print("=============The dataset's shape after oneHot conding================")
    print(f"X shape:",X.shape)
    print(f"test_set shape:",test_set.shape)

    #由于训练集和测试集中存在有彼此没有的值，故而对独热编码之后的训练集和测试集进行扩充
    for i in X.columns.values:
        if X[i].name not in test_set.columns:
            test_set[X[i].name] = 0

    for i in test_set.columns.values:
        if test_set[i].name not in X.columns:
            X[test_set[i].name] = 0

    print("=============The dataset's shape after expanding================")
    print(f"X shape:",X.shape)
    print(f"test_set shape:",test_set.shape)

    #####################################
    #
    # # 分割训练集进行交叉验证
    # # 训练线性回归模型
    #
    #####################################

    print("=============Starting training model and predicting================")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"The prediction score:",model.score(X_test, y_test))

    y_pred = model.predict(X_test)

    plt.scatter(y_pred, y_test, color='y', marker='o')
    plt.scatter(y_test, y_test, color='g', marker='+')
    # plt.show()

    #####################################
    #
    # # 预测测试集
    # # 将预测结果写入csv文件
    #
    #####################################

    print('============saving prediction to result.csv...===================')
    y_pred_test = model.predict(test_set)
    df = pd.DataFrame({'Id':Id_set, 'SalePrice':y_pred_test})
    if (os.path.exists('result.csv')):
        os.remove('result.csv')
    df.to_csv('result.csv', index=False, sep=',')

    plt.show()