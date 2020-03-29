#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/3/30 0:34
# @Author    : Shawn Li
# @FileName  : test_1.py
# @IDE       : PyCharm
# @Blog      : 暂无

import tensorflow as tf
import numpy as np
import pandas as pd
from kerastuner import HyperModel
from kerastuner.tuners.bayesian import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import os
import json


class MyHyperModel(HyperModel):
    def __init__(self, input_shape, output_num):
        super().__init__()
        self.input_shape = input_shape
        self.output_num = output_num

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=16, max_value=1024, step=16), activation=tf.nn.relu))
        # 输出层
        model.add(tf.keras.layers.Dense(units=self.output_num, activation=tf.nn.softmax))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        return model


if __name__ == '__main__':
    print('Starting...')

    TEST_SIZE = 3000
    OBJECTIVE = 'val_accuracy'
    MAX_TRIALS = 25
    EPOCHS = 10000
    BATCH_SIZE = 1024
    CUR_PATH = os.getcwd()
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')



    df = pd.read_csv('./dataset.csv')

    x = df.iloc[:, list(range(11))].copy().values
    y_dloc = df['dloc'].copy().values - 1
    y_ed = df['ED'].copy().values

    random_state = check_random_state(0)
    permutation = random_state.permutation(x.shape[0])
    x = x[permutation]
    y_dloc = y_dloc[permutation]
    y_ed = y_ed[permutation]

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    y = y_dloc  # 选择y的类型， 损伤位置还是程度
    x_train_origin, x_test, y_train_origin, y_test = train_test_split(x, y, test_size=TEST_SIZE)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_origin, y_train_origin, test_size=TEST_SIZE)



    # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    CALLBACKS = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.1, mode='auto')]
    KERAS_TUNER_DIR = os.path.join(CUR_PATH, 'keras_tuner_dir')
    PROJECT_NAME = 'homework_single_dnn_keras_tuner_%s' % DATETIME



    # 实例化贝叶斯优化器
    hypermodel = MyHyperModel((11, ), y.shape[0])
    tuner = BayesianOptimization(hypermodel, objective=OBJECTIVE, max_trials=MAX_TRIALS,
                                 directory=KERAS_TUNER_DIR, project_name=PROJECT_NAME)
    # 开始计时超参数搜索
    tuner_start_time = datetime.now()
    tuner_start = time.time()
    # 开始超参数搜索
    tuner.search(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=CALLBACKS, validation_data=(x_valid, y_valid))
    # tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, validation_data=(x_valid, y_valid))
    # 结束计时超参数搜索
    tuner_end_time = datetime.now()
    tuner_end = time.time()
    # 统计超参数搜索用时
    tuner_duration = tuner_end - tuner_start

    # 获取前BEST_NUM个最优超参数--------------------------------------------------------------
    best_models = tuner.get_best_models()
    best_model = best_models[0]
    history = best_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=CALLBACKS, verbose=2)
    evaluate = best_model.evaluate(x_test, y_test)

    result = dict(
        history=history.history,
        evaluate=evaluate
    )

    with open('test_%s.json' % DATETIME, 'w') as f:
        json.dump(result, f)

    print('Finish!')
