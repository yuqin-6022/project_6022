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

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
sys.path.append(rootPath)

from MyPreprocessing.handlers import clean_handler, missing_handler, encode_handler, scale_handler, select_handler, dimension_reduct_handler


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


def my_bayesian_search(x_train, y_train, x_test, input_shape, batch_size=64, epochs=100, valid_size=0.1, prefix=''):
    # 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size)

    # 贝叶斯优化器参数
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    OBJECTIVE = 'val_accuracy'
    MAX_TRIALS = 25
    TUNER_EPOCHS = epochs
    TUNER_BATCH_SIZE = batch_size
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
    # tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, callbacks=CALLBACKS, validation_data=(x_valid, y_valid))
    tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, validation_data=(x_valid, y_valid))
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
    FIT_BATCH_SIZE = batch_size
    FIT_EPOCHS = epochs
    VERBOSE = 2

    # 依次进行拟合测试
    model_test_results = list()
    for i in range(len(best_models)):
        hp = best_hps[i].values
        model = best_models[i]
        # history = model.fit(x_train, y_train, batch_size=FIT_BATCH_SIZE, epochs=FIT_EPOCHS, callbacks=CALLBACKS, validation_data=(x_valid, y_valid), verbose=VERBOSE)
        history = model.fit(x_train, y_train, batch_size=FIT_BATCH_SIZE, epochs=FIT_EPOCHS, validation_data=(x_valid, y_valid), verbose=VERBOSE)
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

    save_experiment_path = os.path.join(os.getcwd(), '%s_titanic_dnn_keras_tuner.json' % prefix)
    with open(save_experiment_path, 'w') as f:
        json.dump(experiment, f)

    return model_test_results[0]['history']


# 主函数--------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')

    # 设置gpu---------------------------------------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )

    # 实验路径等信息--------------------------------------------------------------------------
    CUR_PATH = os.getcwd()
    VALID_SIZE = 0.1
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    # 实验超参数设置--------------------------------------------------------------------------
    BATCH_SIZE = 256
    EPOCHS = 1000

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
                                                       columns_thresholds=dict(Age=[5, 15, 28, 45, 60],
                                                                               SibSp=[1, 2],
                                                                               Parch=[1, 2, 3],
                                                                               Fare=[df_train_munging['Fare'].quantile(0.25),
                                                                                     df_train_munging['Fare'].quantile(0.5),
                                                                                     df_train_munging['Fare'].quantile(0.75),
                                                                                     df_train_munging['Fare'].quantile(0.9)]),
                                                       columns=['Age', 'SibSp', 'Parch', 'Fare'])
    # df_train_munging, df_test_munging = encode_handler(df_train_munging, df_test_munging,
    #                                                    method='Label',
    #                                                    columns_thresholds=dict(SibSp=[1, 2],
    #                                                                            Parch=[1, 2, 3]),
    #                                                    columns=['SibSp', 'Parch'])
    df_train_encoded, df_test_encoded = df_train_munging.copy(), df_test_munging.copy()

    # 无量纲化（min_max('Age', 'Pclass', 'SibSp_labelencoded', 'Parch_labelencoded', 'Fare_labelencoded')）
    df_train_munging, df_test_munging = scale_handler(df_train_munging, df_test_munging,
                                                      scaler_type='min_max',
                                                      columns=['Age_labelencoded', 'Pclass', 'SibSp_labelencoded',
                                                               'Parch_labelencoded', 'Fare_labelencoded'])
    # df_train_munging, df_test_munging = scale_handler(df_train_munging, df_test_munging,
    #                                                   scaler_type='min_max',
    #                                                   columns=['Age', 'Pclass', 'SibSp_labelencoded',
    #                                                            'Parch_labelencoded', 'Fare'])
    df_train_scaled, df_test_scaled = df_train_munging.copy(), df_test_munging.copy()

    # 贝叶斯优化------------------------------------------------------------------------------
    histories = dict()

    chars_num = df_train_encoded.shape[1]  # 原始特征数量
    input_shape = (chars_num,)
    # 数据清洗——缺失值计算——数值编码
    encoded_prefix = 'encoded'
    histories['%s_history' % encoded_prefix] = my_bayesian_search(df_train_encoded, y_train, df_test_encoded, input_shape, batch_size=BATCH_SIZE, epochs=EPOCHS, valid_size=VALID_SIZE, prefix=encoded_prefix)
    # 数据清洗——缺失值计算——数值编码——无量纲化
    scaled_prefix = 'scaled'
    histories['%s_history' % scaled_prefix] = my_bayesian_search(df_train_scaled, y_train, df_test_scaled, input_shape, batch_size=BATCH_SIZE, epochs=EPOCHS,  valid_size=VALID_SIZE, prefix=scaled_prefix)

    # 数据清洗——缺失值计算——数值编码——无量纲化——降维处理（至少保留2个特征）
    for drop_dimension_num in range(1, chars_num - 1):
        n_components = chars_num - drop_dimension_num
        df_train_reducted, df_test_reducted = dimension_reduct_handler(df_train_scaled.copy(), df_test_scaled.copy(),
                                                                       n_components=n_components)
        input_shape = (n_components,)
        reducted_prefix = 'reducted_%d' % n_components
        histories['%s_history' % reducted_prefix] = my_bayesian_search(df_train_reducted, y_train, df_test_reducted, input_shape, batch_size=BATCH_SIZE, epochs=EPOCHS,  valid_size=VALID_SIZE, prefix=reducted_prefix)

    # 特征选择(至少保留3个特征)
    for drop_char_num in range(1, chars_num - 2):
        # 数据清洗——缺失值计算——数值编码——无量纲化——特征选择
        topk = chars_num - drop_char_num
        df_train_selected, df_test_selected = select_handler(df_train_scaled.copy(), df_test_scaled.copy(), y_train, topk=topk)
        input_shape = (topk,)
        selected_prefix = 'selected_%d' % topk
        histories['%s_history' % selected_prefix] = my_bayesian_search(df_train_selected, y_train, df_test_selected, input_shape, batch_size=BATCH_SIZE, epochs=EPOCHS,  valid_size=VALID_SIZE, prefix=selected_prefix)
        # 数据清洗——缺失值计算——数值编码——无量纲化——特征选择——降维处理
        # 降维处理（至少保留2个特征）
        for drop_dimension_num in range(1, topk - 1):
            n_components = topk - drop_dimension_num
            df_train_reducted, df_test_reducted = dimension_reduct_handler(df_train_selected.copy(), df_test_selected.copy(), n_components=n_components)
            input_shape = (n_components,)
            selected_reducted_prefix = 'selected_%d_reducted_%d' % (topk, n_components)
            histories['%s_history' % selected_reducted_prefix] = my_bayesian_search(df_train_reducted, y_train, df_test_reducted, input_shape, batch_size=BATCH_SIZE, epochs=EPOCHS,  valid_size=VALID_SIZE, prefix='selected_%d_reducted_%d' % (topk, n_components))

    save_histories_path = os.path.join(os.getcwd(), 'histories_%s.json' % DATETIME)
    with open(save_histories_path, 'w') as f:
        json.dump(histories, f)

    print('Finish!')


