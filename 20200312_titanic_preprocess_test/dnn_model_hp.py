#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/3/12 14:59
# @Author    : Shawn Li
# @FileName  : dnn_model_hp.py.py
# @IDE       : PyCharm
# @Blog      : 暂无

import tensorflow as tf
import numpy as np
import pandas as pd
from kerastuner import HyperModel
from kerastuner.tuners.bayesian import BayesianOptimization
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import json
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
# from sklearn.lda import LDA


# 数据清洗函数
def clean_handler(df_train, df_test, *, columns):
    assert (isinstance(df_train, pd.DataFrame)) & (isinstance(df_test, pd.DataFrame)), 'Not a dataframe!'
    assert isinstance(columns, (list, tuple)), 'Not a columns list or tuple!'
    # 检查是提供的特征是否均为数据所持特征
    for column in columns:
        assert (isinstance(column, str)) & (column in df_train.columns), 'Column name "%s" is invalid!' % column

    df_train = df_train.copy()
    df_test = df_test.copy()

    # 数据清洗
    for column in columns:
        del df_train[column]
        del df_test[column]

    return df_train, df_test


# 缺失值计算函数
def missing_handler(df_train, df_test, method='mean', condition=None, *, columns):
    assert (isinstance(df_train, pd.DataFrame)) & (isinstance(df_test, pd.DataFrame)), 'Not a dataframe!'
    assert method in ('mode', 'mean', 'median', 'min', 'max'), 'Unsupported method "%s"' % method
    assert (not condition) | (isinstance(condition, dict)), 'Not a conditions dict!'
    assert isinstance(columns, (list, tuple)), 'Not a columns list or tuple!'
    # 检查是提供的特征是否均为数据所持特征
    for column in columns:
        assert (isinstance(column, str)) & (column in df_train.columns), 'Column name "%s" is invalid!' % column

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train_condition = df_train.copy().dropna()  # 用于进行条件筛选后进行统计值计算

    # 通过条件进行筛选，依据condition提供的条件进行缺失值计算（仅支持==判断，不支持大小比较运算）
    if condition:
        for column, value in condition.items():
            assert column in df_train.columns, 'Column name "%s" is invalid!' % column
            assert value in df_train[column].unique(), 'Does not have value %s!' % value

            df_train_condition = df_train_condition[df_train_condition[column] == value]

    # 如果符合条件的条目为0， 则直接返回
    if not df_train_condition.shape[0]:
        return df_train, df_test

    columns_stats = None  # 用于保存训练数据的均值/中间值/最小值/最大值/众数等统计值

    # 除求众数以外的其他统计值均采用DataFrame提供的统计值
    if method == 'mean':
        columns_stats = dict(df_train_condition.mean())
    elif method == 'median':
        columns_stats = dict(df_train_condition.median())
    elif method == 'min':
        columns_stats = dict(df_train_condition.min())
    elif method == 'max':
        columns_stats = dict(df_train_condition.max())
    elif method == 'mode':
        # 求众数的方法'mode'仅存在于Series中
        columns_stats = dict()
        for column in columns:
            columns_stats[column] = df_train_condition[df_train_condition[column].notnull()][column].mode()[0]

    # 缺失值补齐
    for column in columns:
        df_train.loc[df_train[column].isnull(), column] = columns_stats[column]
        df_test.loc[df_test[column].isnull(), column] = columns_stats[column]

    return df_train, df_test


# 数值编码函数
def encode_handler(df_train, df_test, method='OneHot', columns_thresholds=None, *, columns):
    assert (isinstance(df_train, pd.DataFrame)) & (isinstance(df_test, pd.DataFrame)), 'Not a dataframe!'
    assert method in ('OneHot', 'Label'), 'Unsupported method "%s"' % method
    assert isinstance(columns, (list, tuple)), 'Not a columns list or tuple!'
    # 检查是提供的特征是否均为数据所持特征
    for column in columns:
        assert (isinstance(column, str)) & (column in df_train.columns), 'Column name "%s" is invalid!' % column

    df_train = df_train.copy()
    df_test = df_test.copy()

    if method == 'OneHot':
        for column in columns:
            # 训练集编码
            df_train_column_encoded = pd.get_dummies(df_train[column], prefix=column)
            df_train = df_train.join(df_train_column_encoded)
            del df_train[column]
            # 测试集编码
            df_test_column_encoded = pd.get_dummies(df_test[column], prefix=column)
            df_test = df_test.join(df_test_column_encoded)
            del df_test[column]
    elif method == 'Label':
        # 检查阈值字典
        assert isinstance(columns_thresholds, dict), 'Not a columns_thresholds dict!'
        for column, thresholds in columns_thresholds.items():
            assert isinstance(thresholds, (list, tuple)), 'Not a thresholds list or tuple!'
            # 检查阈值是否为数值
            for threshold in thresholds:
                assert isinstance(threshold, (int, float)), 'Not a threshold int or float!'

        for column in columns:
            thresholds = columns_thresholds[column]
            thresholds.sort(reverse=True)

            # 设置离散化分箱上下界
            right = [float('inf')] + thresholds[:]
            left = thresholds[:]
            left.append(float('-inf'))

            train_column_bins = df_train[column].copy()
            test_column_bins = df_test[column].copy()

            # 连续变量离散化(
            for (left_side, right_side) in zip(left, right):
                train_column_bins[(train_column_bins >= left_side) & (train_column_bins < right_side)] = left_side
                test_column_bins[(test_column_bins >= left_side) & (test_column_bins < right_side)] = left_side

            # 获取离散化分箱类型
            bins = train_column_bins.unique()
            bins.sort()

            # 使用sklearn包里的LabelEncoder进行编码
            le = LabelEncoder()
            le.fit(bins)
            new_column = column + '_labelencoded'
            for column_bins, df in zip((train_column_bins, test_column_bins), (df_train, df_test)):
                column_le = le.transform(column_bins)
                # 将编码后的数据列加入数据集，并删除原数据列
                df[new_column] = column_le
                del df[column]

    return df_train, df_test


# 无量纲化函数
def scale_handler(df_train, df_test, scaler_type='min_max', *, columns):
    assert (isinstance(df_train, pd.DataFrame)) & (isinstance(df_test, pd.DataFrame)), 'Not a dataframe!'
    assert scaler_type in ('standard', 'min_max', 'normalize'), 'Unsupported scaler "%s"' % scaler_type
    assert isinstance(columns, (list, tuple)), 'Not a columns list or tuple!'
    # 检查是提供的特征是否均为数据所持特征
    for column in columns:
        assert (isinstance(column, str)) & (column in df_train.columns), 'Column name "%s" is invalid!' % column

    df_train = df_train.copy()
    df_test = df_test.copy()

    # 实例化无量纲化器
    scaler = None
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'min_max':
        scaler = MinMaxScaler()
    elif scaler_type == 'normalize':
        scaler = Normalizer()

    for column in columns:
        scaler.fit(np.expand_dims(df_train[column], axis=-1))
        new_column = column + '_scaled'
        for df in (df_train, df_test):
            column_2d = np.expand_dims(df[column], axis=-1)
            column_scaled_2d = scaler.transform(column_2d)
            column_scaled = np.squeeze(column_scaled_2d)
            # 将无量纲化后的数据列加入数据集，并删除原数据列
            df[new_column] = column_scaled
            del df[column]

    return df_train, df_test


# 特征评分函数
def _score_fun(X, Y):
    results_features_list = list(map(lambda x: pearsonr(x, Y), X.T))  # 使用皮尔森相关系数进行评分
    results_corr_pval_list = np.asarray(results_features_list).T
    corr_list = results_corr_pval_list[0]
    pval_list = results_corr_pval_list[1]
    return corr_list, pval_list


# 特征选择函数
def select_handler(df_train_x, df_test_x, y_train, score_fun=_score_fun, *, topk):
    assert (isinstance(df_train_x, pd.DataFrame)) & (isinstance(df_test_x, pd.DataFrame)), 'Not a dataframe!'
    assert isinstance(y_train, (pd.Series, np.ndarray, list)), 'Not a label pd.Series、 np.ndarray or list!'
    assert hasattr(score_fun, '__call__'), 'Not a function!'
    assert isinstance(topk, (int, float)) & (topk > 0) & (topk < df_train_x.shape[1]), 'Not a int or float or invalid!'

    df_train_x = df_train_x.copy()
    df_test_x = df_test_x.copy()

    skb = SelectKBest(score_fun, k=int(topk))
    skb.fit(df_train_x, y_train)
    skb.transform(df_train_x)

    feature_ifchosed = skb.get_support(indices=False)  # Boolean 数组，表明每个特征是否被选中
    origin_features = list(df_train_x.columns)  # 原始特征名称
    feature_chosed = list(
        filter(lambda feature: feature_ifchosed[origin_features.index(feature)], origin_features))  # 被选中的特征名称

    return df_train_x[feature_chosed], df_test_x[feature_chosed]


# 特征降维
def dimension_reduct_handler(df_train_x, df_test_x, y_train=None, method='pca', *, n_components):
    assert (isinstance(df_train_x, pd.DataFrame)) & (isinstance(df_test_x, pd.DataFrame)), 'Not a dataframe!'
    assert (not y_train) | isinstance(y_train,
                                      (pd.Series, np.ndarray, list)), 'Not a label pd.Series、 np.ndarray or list!'
    assert method in ('pca', 'lda'), 'Unsupported method "%s"' % method
    assert isinstance(n_components, (int, float)) & (n_components > 0) & (
                n_components < df_train_x.shape[1]), 'Not a int or float or invalid!'

    df_train_x = df_train_x.copy()
    df_test_x = df_test_x.copy()

    x_train_dimension_reducted = df_train_x
    x_test_dimension_reducted = df_test_x

    if method == 'pca':
        pca = PCA(int(n_components))
        x_train_dimension_reducted = pca.fit_transform(df_train_x)
        x_test_dimension_reducted = pca.transform(df_test_x)
    elif method == 'lda':
        print('Not supported!')

    return x_train_dimension_reducted, x_test_dimension_reducted


class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        for i in range(hp.Int('num_layers', 2, 10)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                         min_value=16,
                                                         max_value=256,
                                                         step=16),
                                            activation=tf.nn.relu))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(rate=hp.Float('rate_' + str(i),
                                                            min_value=0,
                                                            max_value=0.5,
                                                            step=0.05)))
        # 输出层
        model.add(tf.keras.layers.Dense(units=1,
                                        activation=tf.nn.sigmoid))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model


def my_bayesian_search(x_train, y_train, x_test, input_shape, valid_size=0.1, prefix=''):
    # 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size)

    # 贝叶斯优化器参数
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    OBJECTIVE = 'val_accuracy'
    MAX_TRIALS = 25
    TUNER_EPOCHS = 1000
    TUNER_BATCH_SIZE = 64
    CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    KERAS_TUNER_DIR = os.path.join(CUR_PATH, 'keras_tuner_dir')
    PROJECT_NAME = '%s_titanic_dnn_keras_tuner_%s' % (prefix, DATETIME)

    # 实例化贝叶斯优化器
    hypermodel = MyHyperModel(input_shape)
    tuner = BayesianOptimization(hypermodel, objective=OBJECTIVE, max_trials=MAX_TRIALS, directory=KERAS_TUNER_DIR, project_name=PROJECT_NAME)
    # 开始计时超参数搜索
    tuner_start_time = datetime.now()
    tuner_start = time.time()
    # 开始超参数搜索
    tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, callbacks=CALLBACKS, validation_data=(x_valid, y_valid))
    # 结束计时超参数搜索
    tuner_end_time = datetime.now()
    tuner_end = time.time()
    # 统计超参数搜索用时
    tuner_duration = tuner_end - tuner_start

    # 获取前BEST_NUM个最优超参数--------------------------------------------------------------
    BEST_NUM = 3  # 获取的最优超参数组数
    best_hps = tuner.get_best_hyperparameters(BEST_NUM)
    best_models = tuner.get_best_models(BEST_NUM)

    # 训练前先保存前BEST_NUM个最优模型
    for i in range(len(best_models)):
        origin_model_path = os.path.join(os.getcwd(), 'keras_tuner_models', '%s_%d_origin_model_%s.h5' % (prefix, i, DATETIME))
        best_models[i].save(origin_model_path)

    # 对前BEST_NUM个最优模型进行训练并评估排序------------------------------------------------
    FIT_BATCH_SIZE = 64
    FIT_EPOCHS = 1000
    VERBOSE = 2

    # 依次进行拟合测试
    model_test_results = list()
    for i in range(len(best_models)):
        hp = best_hps[i].values
        model = best_models[i]
        history = model.fit(x_train, y_train, batch_size=FIT_BATCH_SIZE, epochs=FIT_EPOCHS, callbacks=CALLBACKS, validation_data=(x_valid, y_valid), verbose=VERBOSE)
        y_pred = model.predict(x_test)
        y_pred = [1 if y >= 0.5 else 0 for y in np.squeeze(y_pred)]  # 转换为标签
        model_performance = dict(search_rank=i, hp=hp, history=history.history.__str__(), y_pred=y_pred)
        model_test_results.append(model_performance)

    # 记录本次实验结果
    experiment = dict(
        tuner_start_time=tuner_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        tuner_end_time=tuner_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        tuner_duration=tuner_duration,
        valid_size=VALID_SIZE,
        max_trials=MAX_TRIALS,
        tuner_epochs=TUNER_EPOCHS,
        tuner_batch_size=TUNER_BATCH_SIZE,
        best_num=BEST_NUM,
        fit_batch_size=FIT_BATCH_SIZE,
        fit_epochs=FIT_EPOCHS,
        model_test_results=model_test_results
    )

    save_experiment_path = os.path.join(os.getcwd(), '%s_titanic_dnn_keras_tuner_%s.json' % (prefix, DATETIME))
    with open(save_experiment_path, 'w') as f:
        json.dump(experiment, f)


# 主函数--------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')

    # 实验路径等信息--------------------------------------------------------------------------
    CUR_PATH = os.getcwd()
    VALID_SIZE = 0.1

    # 获取数据集------------------------------------------------------------------------------
    TRAIN_FILE_PATH = os.path.join(CUR_PATH, 'titanic', 'train.csv')  # 原始训练集（含标签）文件路径
    TEST_FILE_PATH = os.path.join(CUR_PATH, 'titanic', 'test.csv')  # 原始测试集文件路径
    df_train = pd.read_csv(TRAIN_FILE_PATH, index_col='PassengerId')  # 原始训练集（含标签）
    df_test = pd.read_csv(TEST_FILE_PATH, index_col='PassengerId')  # 原始测试集

    # 数据再加工------------------------------------------------------------------------------
    # 拷贝原始训练集和测试集用于数据再加工
    df_train_munging, df_test_munging = df_train.copy(), df_test.copy()
    y_train = df_train_munging['Survived'].copy()
    df_train_munging = df_train_munging.iloc[:, 1:].copy()

    # 数据清洗，删除无用数据（'Name', 'Ticket', 'Cabin'）
    df_train_munging, df_test_munging = clean_handler(df_train_munging, df_test_munging,
                                                      columns=['Name', 'Ticket', 'Cabin'])
    df_train_clean, df_test_clean = df_train_munging.copy(), df_test_munging.copy()

    # 缺失值补齐（'Age', 'Embarked', 'Fare'）
    df_train_munging, df_test_munging = missing_handler(df_train_munging, df_test_munging,
                                                        method='median',
                                                        columns=['Age'])
    df_train_munging, df_test_munging = missing_handler(df_train_munging, df_test_munging,
                                                        method='mode',
                                                        condition=dict(Pclass=1, Sex='female', SibSp=0, Parch=0),
                                                        columns=['Embarked'])
    df_train_munging, df_test_munging = missing_handler(df_train_munging, df_test_munging,
                                                        method='median',
                                                        condition=dict(Pclass=3, Sex='male', SibSp=0, Parch=0, Embarked='S'),
                                                        columns=['Fare'])
    df_train_patched, df_test_patched = df_train_munging.copy(), df_test_munging.copy()

    # 数值编码（OneHotEncoding('Sex', 'Embarked')、LabelEncoding('SibSp', 'Parch', 'Fare')）
    df_train_munging, df_test_munging = encode_handler(df_train_munging, df_test_munging,
                                                       method='OneHot',
                                                       columns=['Sex', 'Embarked'])
    df_train_munging, df_test_munging = encode_handler(df_train_munging, df_test_munging,
                                                       method='Label',
                                                       columns_thresholds=dict(SibSp=[1, 2],
                                                                               Parch=[1, 2, 3],
                                                                               Fare=[df_train_munging['Fare'].quantile(0.2),
                                                                                     df_train_munging['Fare'].quantile(0.4),
                                                                                     df_train_munging['Fare'].quantile(0.6),
                                                                                     df_train_munging['Fare'].quantile(0.8)]),
                                                       columns=['SibSp', 'Parch', 'Fare'])
    df_train_encoded, df_test_encoded = df_train_munging.copy(), df_test_munging.copy()

    # 无量纲化（min_max('Age', 'Pclass', 'SibSp_labelencoded', 'Parch_labelencoded', 'Fare_labelencoded')）
    df_train_munging, df_test_munging = scale_handler(df_train_munging, df_test_munging,
                                                      scaler_type='min_max',
                                                      columns=['Age', 'Pclass', 'SibSp_labelencoded',
                                                               'Parch_labelencoded', 'Fare_labelencoded'])
    df_train_scaled, df_test_scaled = df_train_munging.copy(), df_test_munging.copy()

    # 贝叶斯优化------------------------------------------------------------------------------
    chars_num = df_train_encoded.shape[1]  # 原始特征数量
    input_shape = (chars_num,)
    # 数据清洗——缺失值计算——数值编码
    my_bayesian_search(df_train_encoded, y_train, df_test_encoded, input_shape, valid_size=VALID_SIZE, prefix='encoded')
    # 数据清洗——缺失值计算——数值编码——无量纲化
    my_bayesian_search(df_train_scaled, y_train, df_test_scaled, input_shape, valid_size=VALID_SIZE, prefix='scaled')

    # 数据清洗——缺失值计算——数值编码——无量纲化——降维处理（至少保留2个特征）
    for drop_dimension_num in range(1, chars_num - 1):
        n_components = chars_num - drop_dimension_num
        df_train_reducted, df_test_reducted = dimension_reduct_handler(df_train_scaled.copy(), df_test_scaled.copy(),
                                                                       n_components=n_components)
        input_shape = (n_components,)
        my_bayesian_search(df_train_reducted, y_train, df_test_reducted, input_shape, valid_size=VALID_SIZE,
                           prefix='reducted_%d' % n_components)

    # 特征选择(至少保留3个特征)
    for drop_char_num in range(1, chars_num - 2):
        # 数据清洗——缺失值计算——数值编码——无量纲化——特征选择
        topk = chars_num - drop_char_num
        df_train_selected, df_test_selected = select_handler(df_train_scaled.copy(), df_test_scaled.copy(), y_train, topk=topk)
        input_shape = (topk,)
        my_bayesian_search(df_train_selected, y_train, df_test_selected, input_shape, valid_size=VALID_SIZE, prefix='selected_%d' % topk)
        # 数据清洗——缺失值计算——数值编码——无量纲化——特征选择——降维处理
        # 降维处理（至少保留2个特征）
        for drop_dimension_num in range(1, topk - 1):
            n_components = topk - drop_dimension_num
            df_train_reducted, df_test_reducted = dimension_reduct_handler(df_train_selected.copy(), df_test_selected.copy(), n_components=n_components)
            input_shape = (n_components,)
            my_bayesian_search(df_train_reducted, y_train, df_test_reducted, input_shape, valid_size=VALID_SIZE, prefix='selected_%d_reducted_%d' % (topk, n_components))

    print('Finish!')


