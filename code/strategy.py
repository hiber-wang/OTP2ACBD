import pandas as pd
import testing
import modeling
import library
import dataset
import random
import time


def updateFV(word_vector_path, sigmoid, k):
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    word_vector = updateFV_2(word_vector, sigmoid, k)
    return word_vector


def updateFV_2(word_vector, sigmoid, k):
    word_vector_k = word_vector.loc[k, :]
    for i in range(len(word_vector_k)):
        if word_vector_k[i] > 0:
            for j in range(len(word_vector.iloc[:, i])):
                if word_vector.iloc[j, i] > 0:
                    word_vector.iloc[j, i] += sigmoid
    return word_vector


def updateFV_3(word_vector, k):
    word_vector_k = word_vector.loc[k, :]
    for i in range(len(word_vector_k)):
        if word_vector_k[i] > 0:
            for j in range(len(word_vector.iloc[:, i])):
                if word_vector.iloc[j, i] > 0:
                    word_vector.iloc[j, i] = 0.1
    return word_vector


def var_fast_sort(word_vector_path, dis_path, window_size):
    QTR = []
    dis = pd.read_csv(dis_path, index_col=0)
    dis = dis.values.tolist()
    rv = modeling.build_rv(word_vector_path)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    row_name = word_vector._stat_axis.values.tolist()

    dis_dict = {}
    for i in range(len(row_name)):
        dis_dict2 = {}
        for j in range(len(row_name)):
            dis_dict2[row_name[j]] = dis[i][j]
        dis_dict[row_name[i]] = dis_dict2

    rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
    maxRV = rv_sort[0][0]
    QTR.append(maxRV)
    word_vector = word_vector.drop(maxRV, axis=0)
    cnt = 0
    while len(word_vector) > 0:
        cnt += 1
        CTR = []
        rv = modeling.build_rv_2(word_vector)
        rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
        min_window_size = min(window_size, len(rv_sort))
        for i in range(min_window_size):
            CTR.append(rv_sort[i][0])
        maxDistance = -1
        for i in range(len(QTR)):
            for j in range(len(CTR)):
                if dis_dict[QTR[i]][CTR[j]] > maxDistance:
                    maxDistance = dis_dict[QTR[i]][CTR[j]]
                    maxRV = CTR[j]
        QTR.append(maxRV)
        word_vector = word_vector.drop(maxRV, axis=0)
    return QTR


def var(word_vector_path, dis_path, window_size, model_path=None):
    start = time.time()
    QTR = []
    bug_cat_copy = dataset.bug_cat.copy()
    total_cnt = len(bug_cat_copy)
    find_cnt = 0

    dis = pd.read_csv(dis_path, index_col=0)
    dis = dis.values.tolist()
    rv = modeling.build_rv(word_vector_path)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    row_name = word_vector._stat_axis.values.tolist()

    dis_dict = {}
    for i in range(len(row_name)):
        dis_dict2 = {}
        for j in range(len(row_name)):
            dis_dict2[row_name[j]] = dis[i][j]
        dis_dict[row_name[i]] = dis_dict2

    rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
    maxRV = rv_sort[0][0]
    QTR.append(maxRV)

    try:
        crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + maxRV)
    except:
        crash_flag = 1
    if crash_flag != 1:
        try:
            testing.detect_emi(filename=maxRV, file_path=model_path)
        except:
            print(maxRV, "can't use the detect_emi function")

    word_vector = word_vector.drop(maxRV, axis=0)
    cnt = 0
    if library.find_bug(maxRV, bug_cat_copy) == 1:
        find_cnt += 1
        print('find the bug', find_cnt, '/', total_cnt)
    while len(word_vector) > 0 and len(bug_cat_copy) > 0:
        cnt += 1
        CTR = []
        rv = modeling.build_rv_2(word_vector)
        rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
        min_window_size = min(window_size, len(rv_sort))
        for i in range(min_window_size):
            CTR.append(rv_sort[i][0])
        maxDistance = -1
        for i in range(len(QTR)):
            for j in range(len(CTR)):
                if dis_dict[QTR[i]][CTR[j]] > maxDistance:
                    maxDistance = dis_dict[QTR[i]][CTR[j]]
                    maxRV = CTR[j]
        QTR.append(maxRV)
        if library.find_bug(maxRV, bug_cat_copy) == 1:
            find_cnt += 1
            print('find the bug', find_cnt, '/', total_cnt)
        crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + maxRV)
        if crash_flag != 1:
            try:
                testing.detect_emi(filename=maxRV, file_path=model_path)
            except:
                print(maxRV, "can't use the detect_emi function")
        word_vector = word_vector.drop(maxRV, axis=0)
    end = time.time()
    return end - start


def danger_fast_test(word_vector_path, sigmoid):
    QTR = []
    rv = modeling.build_rv(word_vector_path)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
    maxRV = rv_sort[0][0]
    QTR.append(maxRV)
    if maxRV not in dataset.bug_total:
        word_vector = updateFV_3(word_vector, maxRV)
    else:
        word_vector = updateFV_2(word_vector, sigmoid, maxRV)
    word_vector = word_vector.drop(maxRV, axis=0)
    cnt = 0
    while len(word_vector) > 0:
        cnt += 1
        rv = modeling.build_rv_2(word_vector)
        rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
        maxRV = rv_sort[0][0]
        QTR.append(maxRV)
        if maxRV not in dataset.bug_total:
            word_vector = updateFV_3(word_vector, maxRV)
        else:
            word_vector = updateFV_2(word_vector, sigmoid, maxRV)
        word_vector = word_vector.drop(maxRV, axis=0)
    return QTR


def danger(word_vector_path, sigmoid, model_path):
    start = time.time()
    QTR = []
    bug_cat_copy = dataset.bug_cat.copy()
    find_cnt = 0
    total_cnt = len(bug_cat_copy)
    rv = modeling.build_rv(word_vector_path)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
    maxRV = rv_sort[0][0]
    QTR.append(maxRV)
    if maxRV not in dataset.bug_total:
        word_vector = updateFV_3(word_vector, maxRV)
    else:
        word_vector = updateFV_2(word_vector, sigmoid, maxRV)
    word_vector = word_vector.drop(maxRV, axis=0)
    cnt = 0
    try:
        crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + maxRV)
    except:
        crash_flag = 1
    if crash_flag != 1:
        try:
            testing.detect_emi(filename=maxRV, file_path=model_path)
        except:
            print(maxRV, "can't use the detect_emi function")
    if library.find_bug(maxRV, bug_cat_copy) == 1:
        find_cnt += 1
        print('find the bug', find_cnt, '/', total_cnt)
    while len(word_vector) > 0 and len(bug_cat_copy) > 0:
        cnt += 1
        rv = modeling.build_rv_2(word_vector)
        rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
        maxRV = rv_sort[0][0]
        QTR.append(maxRV)
        if maxRV not in dataset.bug_total:
            word_vector = updateFV_3(word_vector, maxRV)
        else:
            word_vector = updateFV_2(word_vector, sigmoid, maxRV)
        word_vector = word_vector.drop(maxRV, axis=0)
        try:
            crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + maxRV)
        except:
            crash_flag = 1
        if crash_flag != 1:
            try:
                testing.detect_emi(filename=maxRV, file_path=model_path)
            except:
                print(maxRV, "can't use the detect_emi function")
        if library.find_bug(maxRV, bug_cat_copy) == 1:
            find_cnt += 1
            print('find the bug', find_cnt, '/', total_cnt)
    end = time.time()
    return end - start


def var_danger_fast_test(word_vector_path, dis_path, window_size, sigmoid):
    QTR = []
    increment = 2
    dis = pd.read_csv(dis_path, index_col=0)
    dis = dis.values.tolist()
    rv = modeling.build_rv(word_vector_path)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    row_name = word_vector._stat_axis.values.tolist()
    dis_dict = {}
    for i in range(len(row_name)):
        dis_dict2 = {}
        for j in range(len(row_name)):
            dis_dict2[row_name[j]] = dis[i][j]
        dis_dict[row_name[i]] = dis_dict2
    rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
    maxRV = rv_sort[0][0]
    QTR.append(maxRV)
    if maxRV not in dataset.bug_total:
        word_vector = updateFV_3(word_vector, maxRV)
    else:
        word_vector = updateFV_2(word_vector, sigmoid, maxRV)
    word_vector = word_vector.drop(maxRV, axis=0)
    cnt = 0
    while len(word_vector) > 0:
        cnt += 1
        CTR = []
        rv = modeling.build_rv_2(word_vector)
        rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
        min_window_size = min(window_size, len(rv_sort))
        for i in range(min_window_size):
            CTR.append(rv_sort[i][0])
        maxDistance = -1
        for i in range(len(QTR)):
            for j in range(len(CTR)):
                if dis_dict[QTR[i]][CTR[j]] > maxDistance:
                    maxDistance = dis_dict[QTR[i]][CTR[j]]
                    maxRV = CTR[j]
        QTR.append(maxRV)
        if maxRV not in dataset.bug_total:
            word_vector = updateFV_3(word_vector, maxRV)
            if increment == 2:
                window_size += increment * window_size
            else:
                window_size += increment
            if window_size > len(rv_sort):
                window_size = len(rv_sort) // 2
                increment = 1
        else:
            word_vector = updateFV_2(word_vector, sigmoid, maxRV)
        word_vector = word_vector.drop(maxRV, axis=0)
    return QTR


def var_danger(word_vector_path, dis_path, window_size, sigmoid, model_path):
    start = time.time()
    QTR = []
    bug_cat_copy = dataset.bug_cat.copy()
    find_cnt = 0
    total_cnt = len(bug_cat_copy)
    increment = 2
    dis = pd.read_csv(dis_path, index_col=0)
    dis = dis.values.tolist()
    rv = modeling.build_rv(word_vector_path)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    row_name = word_vector._stat_axis.values.tolist()
    dis_dict = {}
    for i in range(len(row_name)):
        dis_dict2 = {}
        for j in range(len(row_name)):
            dis_dict2[row_name[j]] = dis[i][j]
        dis_dict[row_name[i]] = dis_dict2
    rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
    maxRV = rv_sort[0][0]
    QTR.append(maxRV)
    if maxRV not in dataset.bug_total:
        word_vector = updateFV_3(word_vector, maxRV)
    else:
        word_vector = updateFV_2(word_vector, sigmoid, maxRV)
    word_vector = word_vector.drop(maxRV, axis=0)
    cnt = 0
    try:
        crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + maxRV)
    except:
        crash_flag = 1
    if crash_flag != 1:
        try:
            testing.detect_emi(filename=maxRV, file_path=model_path)
        except:
            print(maxRV, "can't use the detect_emi function")
    if library.find_bug(maxRV, bug_cat_copy) == 1:
        find_cnt += 1
        print('find the bug', find_cnt, '/', total_cnt)
    while len(word_vector) > 0 and len(bug_cat_copy) > 0:
        cnt += 1
        CTR = []
        rv = modeling.build_rv_2(word_vector)
        rv_sort = sorted(rv.items(), key=lambda x: x[1], reverse=True)
        min_window_size = min(window_size, len(rv_sort))
        for i in range(min_window_size):
            CTR.append(rv_sort[i][0])
        maxDistance = -1
        for i in range(len(QTR)):
            for j in range(len(CTR)):
                if dis_dict[QTR[i]][CTR[j]] > maxDistance:
                    maxDistance = dis_dict[QTR[i]][CTR[j]]
                    maxRV = CTR[j]
        QTR.append(maxRV)
        if maxRV not in dataset.bug_total:
            word_vector = updateFV_3(word_vector, maxRV)
            if increment == 2:
                window_size += increment * window_size
            else:
                window_size += increment
            if window_size > len(rv_sort):
                window_size = len(rv_sort) // 2
                increment = 1
        else:
            word_vector = updateFV_2(word_vector, sigmoid, maxRV)
        word_vector = word_vector.drop(maxRV, axis=0)
        try:
            crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + maxRV)
        except:
            crash_flag = 1
        if crash_flag != 1:
            try:
                testing.detect_emi(filename=maxRV, file_path=model_path)
            except:
                print(maxRV, "can't use the detect_emi function")
        if library.find_bug(maxRV, bug_cat_copy) == 1:
            find_cnt += 1
            print('find the bug', find_cnt, '/', total_cnt)
    end = time.time()
    return end - start


def random_fast_test(word_vector_path):
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    row_name = word_vector._stat_axis.values.tolist()
    random_list = []
    while len(row_name) > 0:
        block = random.choice(row_name)
        row_name.remove(block)
        random_list.append(block)
        random_list.reverse()
    random_list.reverse()
    return random_list


def random_sort(word_vector_path, model_path):
    start = time.time()
    find_cnt = 0
    bug_cat_copy = dataset.bug_cat.copy()
    total_cnt = len(bug_cat_copy)
    word_vector = pd.read_csv(word_vector_path, index_col=0)
    row_name = word_vector._stat_axis.values.tolist()
    random_list = []
    while len(row_name) > 0 and len(bug_cat_copy) > 0:
        random.shuffle(row_name)
        block = random.choice(row_name)
        try:
            crash_flag = testing.detect_crash_with_1_testcase(model_path + '/' + block)
        except:
            crash_flag = 1
        if crash_flag != 1:
            try:
                testing.detect_emi(filename=block, file_path=model_path)
            except:
                print(block, "can't use the detect_emi function")
        find_flag = library.find_bug(block, bug_cat_copy)
        if find_flag == 1:
            find_cnt += 1
            print('find the bug', find_cnt, '/', total_cnt)
        row_name.remove(block)
        random_list.append(block)
        random_list.reverse()
    random_list.reverse()
    end = start.time()
    return end - start
