import random
import numpy as np
import extra_token
import greedy
import adaptive_random as ar
import x2
import standard
import Xmeans

def sortEval(sort_result):
    # bug用例在原始输入中的索引
    bug = list(range(0, 49))
    bug0 = list(range(0, 43))
    bug1 = list(range(44, 47))
    bug2 = list(range(47, 57))
    bug3 = list([58])
    bug5 = list([59])
    bug6 = list([60])
    bug7 = list([61])
    bug8 = list([62])
    bug9 = list([63])
    bug10 = list([64])
    bug11 = list([65])
    bug12 = list([66])


    nobug = list(range(49, 876))
    f0, f1, f2, f3, f5, f6, f7, f8, f9, f10, f11, f12 = True, True, True, True, True, True, True, True, True, True, True, True
    n = 1065
    m = 12
    sort_index = {}  # 排序后bug的索引字典
    sort_cnt = 0  # 排序后当前的用例
    TF = 0
    for i in sort_result:
        if i in bug0 and f0:
            sort_index[i] = sort_cnt
            f0 = False
        if i in bug1 and f1:
            sort_index[i] = sort_cnt
            f1 = False
        if i in bug2 and f2:
            sort_index[i] = sort_cnt
            f2 = False
        if i in bug3 and f3:
            sort_index[i] = sort_cnt
            f3 = False
        if i in bug5 and f5:
            sort_index[i] = sort_cnt
            f5 = False
        if i in bug6 and f6:
            sort_index[i] = sort_cnt
            f6 = False
        if i in bug7 and f7:
            sort_index[i] = sort_cnt
            f7 = False
        if i in bug8 and f8:
            sort_index[i] = sort_cnt
            f8 = False
        if i in bug9 and f9:
            sort_index[i] = sort_cnt
            f9 = False
        if i in bug10 and f10:
            sort_index[i] = sort_cnt
            f10 = False
        if i in bug11 and f11:
            sort_index[i] = sort_cnt
            f11 = False
        if i in bug12 and f12:
            sort_index[i] = sort_cnt
            f12 = False
        sort_cnt += 1
    sort_list = list(sort_index.values()) #bug用例在排序后的索引
    for i in range(len(sort_list)):
        TF += sort_list[i]
    APFD = 1 - (TF / (n * m)) + 1 / (2 * n)
    #print("bug索引值为：")
    #print(sort_list)
    #print(APFD)
    return APFD

def run():
    exs = extra_token.read_vec_list()
    target = list(range(len(exs)))
    #exs = float(exs)
    #print(exs)
    xmeans_result = x2.myxmeans(exs, target, 1, 20)
    dis_sorted = []
    max_len = 0
    # dis_sorted = greedy.compute(exs)
    # print(dis_sorted)
    for key in xmeans_result['clusters'].keys():
        print('----------------', key, '------------------')
        print(xmeans_result['clusters'][key]['target'], '\n')
        #一个类别包含所有用例的索引
        class_list_index = xmeans_result['clusters'][key]['target']
        class_dis_sorted_tuple = greedy.compute(exs, class_list_index) #返回的类型为元组数组 贪心
        print(class_dis_sorted_tuple)
        # 从每个元组中提出第一个元素
        class_dis_sorted = []
        for key, value in class_dis_sorted_tuple:
            class_dis_sorted.append(key)

        #单个子列表最大包含了多少用例
        if len(class_dis_sorted) > max_len:
            max_len = len(class_dis_sorted)
        #生成二维列表
        dis_sorted.append(class_dis_sorted)

    #子列表长度不规整，将其填充值，生成矩阵
    for i in range(len(dis_sorted)):
        fill_cnt = len(dis_sorted[i])
        while fill_cnt < max_len:
            dis_sorted[i].append(-1)
            fill_cnt += 1

    shunxu = []
    for i in range(max_len):
        for j in range(len(dis_sorted)):
            if dis_sorted[j][i] == -1:
                continue
            else:
                shunxu.append(dis_sorted[j][i])

    return shunxu

def run_adaptive():
    exs = extra_token.read_vec_numpy()
    sort_result = ar.adaptive(exs)
    return sort_result

def run_beam():
    exs = extra_token.read_vec_numpy()
    sort_result = ar.beam(exs)
    return sort_result

sort_cnt = 1000 # 排序次数
rand_cnt = 1000
model_sum = 0


"""search结果"""
shunxu = run_beam()
print("shunxu")
print(len(shunxu))
print("模型的结果（beam）：")
print(sortEval(shunxu))

"""自适应结果"""
shunxu = run_adaptive()
print('shunxu:')
print(len(shunxu))
print("模型的结果（自适应）：")
print(sortEval(shunxu))

"""贪心结果"""
shunxu = run()
nn = sortEval(shunxu)
print("模型的结果（贪心）：")
print(sortEval(shunxu))

# for i in range(sort_cnt):
#     shunxu = run()
#     print('shunxu:')
#     print(len(shunxu))
#     nn = sortEval(shunxu)
#     model_sum += nn
# print("模型的结果：")
# print(model_sum/sort_cnt)

shuffle_sum = 0
for i in range(rand_cnt):
    shuffle_list = list(range(0, 1065))
    random.shuffle(shuffle_list)
    rr = sortEval(shuffle_list)
    shuffle_sum += rr
print("随机的结果：")
print(shuffle_sum/rand_cnt)

