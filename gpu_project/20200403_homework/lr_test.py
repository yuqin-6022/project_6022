#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/4/4 21:54
# @Author    : Shawn Li
# @FileName  : lr_test.py
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
from sklearn.metrics import f1_score, recall_score, precision_score
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
        # 输出层
        model.add(tf.keras.layers.Dense(units=self.output_num, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        return model


# Metrics--------------------------------------------------------------------------------------------------------------
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


# 终端运行-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')

    # 设置gpu---------------------------------------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )

    TEST_SIZE = 3000
    OBJECTIVE = 'val_f1'
    MAX_TRIALS = 5
    EPOCHS = 25000
    BATCH_SIZE = 256
    CUR_PATH = os.getcwd()
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    KERAS_TUNER_DIR = os.path.join(CUR_PATH, 'keras_tuner_dir')
    if not os.path.exists(KERAS_TUNER_DIR):
        os.makedirs(KERAS_TUNER_DIR)

    KERAS_TUNER_MODEL_DIR = os.path.join(CUR_PATH, 'keras_tuner_models_%s' % DATETIME)
    if not os.path.exists(KERAS_TUNER_MODEL_DIR):
        os.makedirs(KERAS_TUNER_MODEL_DIR)

    # 数据集-----------------------------------------------------------------------------------------------------------
    df = pd.read_csv('./dataset.csv')

    random_state = check_random_state(0)
    permutation = random_state.permutation(df.shape[0])
    df = df.iloc[permutation, :]

    x = df.iloc[:, list(range(11))].copy().values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 超参搜索开始-----------------------------------------------------------------------------------------------------
    for y_type in ['dloc', 'ED', 'overload_loc']:

        y = df[y_type].copy().values

        x_train_origin, x_test, y_train_origin, y_test = train_test_split(x, y, test_size=TEST_SIZE)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_origin, y_train_origin, test_size=TEST_SIZE)

        # keras-tuner部分设置----------------------------------------------------------------------------------------------
        # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
        CALLBACKS = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.1, mode='auto'),
            Metrics(valid_data=(x_valid, y_valid))
        ]
        PROJECT_NAME = '%s_lr_keras_tuner_%s' % (y_type, DATETIME)
        MODEL_PATH = os.path.join(KERAS_TUNER_MODEL_DIR, '%s_lr.h5' % y_type)

        # 实例化贝叶斯优化器
        hypermodel = MyHyperModel((x.shape[1],), y.shape[0])
        tuner = BayesianOptimization(hypermodel, objective=OBJECTIVE, max_trials=MAX_TRIALS,
                                     directory=KERAS_TUNER_DIR, project_name=PROJECT_NAME)
        # 开始计时超参数搜索
        tuner_start_time = datetime.now()
        tuner_start = time.time()
        # 开始超参数搜索
        tuner.search(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                     validation_data=(x_valid, y_valid), callbacks=CALLBACKS)
        # tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, validation_data=(x_valid, y_valid))
        # 结束计时超参数搜索
        tuner_end_time = datetime.now()
        tuner_end = time.time()
        # 统计超参数搜索用时
        tuner_duration = tuner_end - tuner_start

        # 获取前BEST_NUM个最优超参数--------------------------------------------------------------
        best_models = tuner.get_best_models()
        best_model = best_models[0]
        best_model.save(KERAS_TUNER_MODEL_DIR)

        history = best_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=CALLBACKS, verbose=2)
        evaluate = best_model.evaluate(x_test, y_test)
        test_loss = float(evaluate[0])
        test_accuracy = float(evaluate[1])

        print('------------------------------------------------------------------------------------------------------')
        print('%s_test_result: ' % y_type)
        print('test_loss: %.4f, test_accuracy: %.4f' % (test_loss, test_accuracy))
        print('------------------------------------------------------------------------------------------------------')

        result = dict(
            history=history.history,
            test_loss=test_loss,
            test_accuracy=test_accuracy
        )

        with open('%s_lr_%s.json' % (y_type, DATETIME), 'w') as f:
            json.dump(result, f)

    print('Finish!')

