#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/3/11 15:18
# @Author    : Shawn Li
# @FileName  : handlers.py
# @IDE       : PyCharm
# @Blog      : 暂无

import numpy as np
import pandas as pd
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

            train_column_bins = df_train[column]
            test_column_bins = df_test[column]

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
