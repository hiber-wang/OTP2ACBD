import io
from sklearn import preprocessing
import pandas as pd
import preprocessing
import os
from tqdm import tqdm
import matlab.engine
import dataset
import math
import numpy as np


def build_word_vector(model_path, word_dic_path, metric_path, block_flag=True, line_flag=True, metric_flag=True):
    word_dic = pd.read_csv(word_dic_path)
    word_vector = {}

    # 获取到特征的复杂度
    if metric_flag:
        # get_matlab_metrics(eng_path=None, model_path=model_path, metric_path=metric_path)
        # regular_metrics_value(metric_path)
        print('import metric...')
        metrics = pd.read_csv(metric_path, index_col=0)
        metric_filename = metrics._stat_axis.values.tolist()
        metric_value = metrics.values.tolist()
        metric_column = metrics.columns.tolist()
    if os.path.exists(model_path):
        filenames = os.listdir(model_path)
        for filename in filenames:
            if 'mdl' not in filename:
                continue
            filename_i = filename
            filename = model_path + '/' + filename
            block = preprocessing.divide_block(filename)
            line = preprocessing.divide_line(filename)
            model = {}
            for i in word_dic.iloc[:, 0]:
                model[i] = 0
            if block_flag:
                for i in block.keys():
                    if i in model.keys():
                        model[i] = 1
            if line_flag:
                for j in line.keys():
                    str_j = "(" + "'" + j[0] + "'" + ", " + "'" + j[1] + "'" + ")"
                    if str_j in model.keys():
                        model[str_j] = 1
            # 新增复杂度判断标准
            if metric_flag:
                metric_filename_index = metric_filename.index(filename_i)
                metric_filename_index_metrics = metric_value[metric_filename_index]
                for metric_column_name in metric_column:
                    metric_column_index = metric_column.index(metric_column_name)
                    model[metric_column_name] = metric_filename_index_metrics[metric_column_index]

            word_vector[filename_i] = model
    return word_vector


def build_rv(word_vector_path, metric_path=None):
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    metric = None
    if metric_path is not None:
        metric = pd.read_csv(metric_path, index_col=0)
    rv = build_rv_2(word_vector, metric)
    return rv


def build_rv_2(word_vector, metric=None):
    rv = {}
    row_name = word_vector._stat_axis.values.tolist()
    for i in range(len(word_vector)):
        cnt = 0
        for j in word_vector.loc[row_name[i], :]:
            cnt += j
        rv[row_name[i]] = cnt

    if metric is not None:
        for i in range(len(word_vector)):
            cnt = 0
            for j in metric.loc[row_name[i], :]:
                cnt += j
            rv[row_name[i]] += cnt
    return rv


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


def build_dis(word_vector_path, dis_type='m'):
    word_vector = pd.read_csv(word_vector_path)
    tr, block = build_dis_2(word_vector, dis_type)
    return tr, block


def build_dis_2(word_vector, dis_type='m'):
    word_vector_len = len(word_vector)
    tr = [[0 for _ in range(word_vector_len)] for _ in range(word_vector_len)]
    block = []
    for i in tqdm(range(len(word_vector))):
        block.append(word_vector.iloc[i, 0])
        for j in range(i + 1, len(word_vector)):
            cnt = 0
            if dis_type == 'm':
                cnt = np.sum(np.abs(word_vector.iloc[i, 1:] - word_vector.iloc[j, 1:]))
            elif dis_type == 'e':
                cnt = np.sqrt(np.sum(np.square(word_vector.iloc[i, 1:] - word_vector.iloc[j, i:])))
            elif dis_type == 'c':
                cnt = 1 - cos_sim(list(word_vector.iloc[i, 1:]), list(word_vector.iloc[j, 1:]))
            # if dis_type == 'm' or dis_type == 'e':
            #     for k in range(1, 1 + len(word_vector.iloc[i, 1:])):
            #         if dis_type == 'm':
            #             diff = word_vector.iloc[i, k] - word_vector.iloc[j, k]
            #             if diff >= 0:
            #                 cnt += diff
            #             else:
            #                 cnt += (-diff)
            #         elif dis_type == 'e':
            #             diff = math.pow((word_vector.iloc[i, k] - word_vector.iloc[j, k]), 2)
            #             cnt += diff
            #     if dis_type == 'e':
            #         cnt = math.sqrt(cnt)
            # elif dis_type == 'c':
            #     cnt = cos_sim(list(word_vector.iloc[i, 1:]), list(word_vector.iloc[j, 1:]))
            tr[i][j] = tr[j][i] = cnt
    return tr, block


def get_matlab_metrics(eng_path, model_path, metric_path):
    filename_dic = {}
    path = model_path
    if os.path.exists(path):
        filenames = os.listdir(path)
        for filename in filenames:
            max_value = -1
            metrics_dic = {}
            if 'mdl' in filename:
                eng = matlab.engine.start_matlab()
                out = io.StringIO()
                err = io.StringIO()
                eng.cd(eng_path)
                test_case = filename[:-4]
                for metric in dataset.metrics:
                    result = -1
                    try:
                        result = eng.get_matlab_metrics(test_case, metric, nargout=1, stdout=out, stderr=err)
                        max_value = max(max_value, result)
                    except:
                        eng = matlab.engine.start_matlab()
                        eng.cd(eng_path)
                    metrics_dic[metric] = result
                if max_value == -1:
                    max_value = 1
                for metrics_dic_key in metrics_dic.keys():
                    if metrics_dic[metrics_dic_key] == -1:
                        metrics_dic[metrics_dic_key] = max_value
                filename_dic[filename] = metrics_dic
                print(filename, metrics_dic)

    filename_df = pd.DataFrame.from_dict(filename_dic)
    word_vector = filename_df.T
    word_vector.to_csv(metric_path)


def regular_metrics_value(metric_path):
    metrics_value = pd.read_csv(metric_path, index_col=0)
    row_name = metrics_value._stat_axis.values.tolist()
    scaler = preprocessing.MinMaxScaler().fit(metrics_value.T)
    metrics_value_T_scale = scaler.transform(metrics_value.T)
    metrics_value_scale = metrics_value_T_scale.transpose()
    for i in range(len(metrics_value)):
        for j in range(len(metrics_value.loc[row_name[i], :])):
            metrics_value.iloc[i, j] = metrics_value_scale[i][j]
    metrics_value.to_csv(metric_path)


if __name__ == '__main__':
    word_vector = pd.read_csv('../data/50/word_vector.csv')
    print(np.sum(np.abs(word_vector.iloc[1, 1:] - word_vector.iloc[2, 1:])))
    # metric_path = '../data/metrics.csv'
    # metrics = pd.read_csv(metric_path, index_col=0)
    # print(metrics.head())
    # filename = metrics._stat_axis.values.tolist()
    # values = metrics.values.tolist()
    # columns = metrics.columns.tolist()
    # for signal_filename in filename:
    #     filename_index = filename.index(signal_filename)
    #     for signal_metric in columns:
    #         metrics_index = columns.index(signal_metric)
    #         print(values[filename_index][metrics_index], end=" ")
    #     print()
    #     break
