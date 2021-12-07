import random

import numpy as np
def adaptive(exs):
    book = [True for x in range(len(exs))] #判断当前索引是否已经使用过
    sort_process = {}
    sort_result = []
    k = 0
    while k != -1:
        dis_max = 0
        next_index = -1
        exs_num = exs.shape[0]
        for j in range(exs_num):
            # print(j)
            if not book[j]:
                continue
            else:
                dis_j = np.sum(abs(exs[k]-exs[j]))
                if dis_j > dis_max:
                    dis_max = dis_j
                    next_index = j
                    #print(next_index)
        book[k] = False
        sort_process['current'] = k
        sort_process['next_index'] = next_index
        #print(sort_process['current'])
        sort_result.append(k)
        k = next_index
    return sort_result

def adaptive_beam(exs, split, beam_num):
    book = [True for x in range(len(exs))]  # 判断当前索引是否已经使用过
    sort_process = {}
    sort_result = []
    k = 0
    while k != -1:
        dis_max = 0
        next_index = -1
        exs_num = exs.shape[0]
        for j in range(exs_num):
            # print(j)
            if not book[j]:
                continue
            else:
                dis_j = np.sum(abs(exs[k] - exs[j]))
                if dis_j > dis_max:
                    dis_max = dis_j
                    next_index = j
                    # print(next_index)
        book[k] = False
        sort_process['current'] = k
        sort_process['next_index'] = next_index
        # print(sort_process['current'])
        sort_result.append(k+split*beam_num)
        k = next_index
    return sort_result

def beam(exs):
    beam_num = 5
    sort_result = []
    exs_smalls = np.split(exs, exs.shape[0]/beam_num, axis=0)
    split = 0
    for i in exs_smalls:
        sort_exs_small = adaptive_beam(i, split, beam_num)
        split += 1
        # print(sort_exs_small)
        for j in sort_exs_small:
            sort_result.append(j)
    return sort_result