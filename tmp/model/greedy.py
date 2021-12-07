import numpy as np

def compute(exs_list, class_list_index):
    exs = np.array(exs_list)
    #ex_num = exs.shape[0]
    zero_vec = np.zeros(exs.shape[1])
    dis = {}
    for i in class_list_index:
        dis[i] = np.sum(np.abs(exs[i] - zero_vec))
    print(dis)
    dis_sorted = sorted(dis.items(), key=lambda dis: dis[1], reverse=True)
    return dis_sorted

def compute2(exs_list):
    exs = np.array(exs_list)
    print(exs)
    # ex_num = exs.shape[0]
    zero_vec = np.zeros(exs.shape[1])
    print(zero_vec.shape)
    print(exs.shape[1])
    dis = {}
    for i in exs_list:
        dis[i] = np.sum(np.abs(exs[i] - zero_vec))
    dis_sorted = sorted(dis.items(), key=lambda dis: dis[1], reverse=True)
    return dis_sorted

