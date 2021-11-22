import os

import preprocessing
import modeling
import time
import pandas as pd
import strategy
import library


def generate_dictionary_table(folder_path=None, word_dic_path=None, sigmoid=1):
    if folder_path is None or word_dic_path is None:
        print("Error: path isn't exist")
        return -1
    print("building dictionary...")
    start = time.time()
    block = preprocessing.make_feature_dictionary(folder_path, sigmoid)
    word_dic = pd.DataFrame.from_dict(block, orient='index', columns=['times'])
    word_dic.to_csv(word_dic_path)
    end = time.time()
    print("building dictionary:done")
    return end - start


def generate_word_vector_table(folder_path=None, word_dic_path=None, word_vector_path=None):
    if folder_path is None or word_dic_path is None or word_vector_path is None:
        print("Error: path isn't exist")
        return -1
    print('building word vector...')
    start = time.time()
    word_vector = modeling.build_word_vector(folder_path, word_dic_path)
    word_vector = pd.DataFrame.from_dict(word_vector)
    word_vector = word_vector.T
    word_vector.to_csv(word_vector_path)
    end = time.time()
    print('building word vector:done')
    return end - start


def generate_distance_matrix(word_vector_path=None, dis_path=None):
    if word_vector_path is None or dis_path is None:
        print("Error: path isn't exist")
        return -1
    print('building distance matrix...')
    start = time.time()
    div, block_list = modeling.build_dis(word_vector_path)
    div = pd.DataFrame(div, columns=block_list, index=block_list)
    div.to_csv(dis_path)
    print("building distance matrix:done")
    end = time.time()
    return end - start


def run_for_effectiveness_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None, dis_path=None,
                                          data_flag=True, random_epochs=100):
    if folder_path is None or word_dic_path is None or word_vector_path is None or dis_path is None:
        print("path isn't exist")
        return -1
    if data_flag:
        generate_dictionary_table(folder_path, word_dic_path)
        generate_word_vector_table(folder_path, word_dic_path, word_vector_path)
        generate_distance_matrix(word_vector_path, dis_path)
    _, total_number, _ = library.get_window_size(word_vector_path)

    print('using the var strategy ...')
    QTR = strategy.var_fast_sort(word_vector_path, dis_path, total_number)
    var_apfd = library.evaluate(QTR)
    print('the apfd of var strategy is', var_apfd)
    library.record(words='the apfd of var strategy is ' + str(var_apfd), filename='apfd_value')

    print('using the danger strategy ...')
    QTR = strategy.danger_fast_test(word_vector_path, 1)
    danger_apfd = library.evaluate(QTR)
    print('the apfd of danger strategy is', danger_apfd)
    library.record(words='the apfd of danger strategy is ' + str(danger_apfd), filename='apfd_value')

    print('using the var_danger strategy ...')
    QTR = strategy.var_danger_fast_test(word_vector_path, dis_path, 1, 1)
    var_danger_apfd = library.evaluate(QTR)
    print('the apfd of var_danger strategy is', var_danger_apfd)
    library.record(words='the apfd of var_danger strategy is ' + str(var_danger_apfd), filename='apfd_value')

    print('using the random_strategy ...')
    average_random_apfd = 0
    for epoch in range(random_epochs):
        QTR = strategy.random_fast_test(word_vector_path)
        random_apfd = library.evaluate(QTR)
        average_random_apfd += random_apfd
    average_random_apfd /= random_epochs
    print('the apfd of random strategy is', average_random_apfd)
    library.record(words='the apfd of random strategy is ' + str(average_random_apfd), filename='apfd_value')
    return var_apfd, danger_apfd, var_danger_apfd, average_random_apfd


def run_for_efficiency_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None, dis_path=None,
                                       model_path=None, random_epochs=10):
    if folder_path is None or word_dic_path is None or word_vector_path is None or dis_path is None or \
            model_path is None:
        print("path isn't exist")
        return -1

    dictionary_time = generate_dictionary_table(folder_path, word_dic_path)
    word_vector_time = generate_word_vector_table(folder_path, word_dic_path, word_vector_path)
    distance_matrix_time = generate_distance_matrix(word_vector_path, dis_path)
    _, total_number, _ = library.get_window_size(word_vector_path)

    print('using the var strategy ...')
    var_time = strategy.var(word_vector_path, dis_path, total_number, model_path)
    print('the time of var strategy is', var_time)
    library.record(words='the time of var strategy is ' + str(var_time), filename='run_time')

    print('using the danger strategy ...')
    danger_time = strategy.danger(word_vector_path, 1, model_path)
    print('the time of danger strategy is', danger_time)
    library.record(words='the time of danger strategy is ' + str(danger_time), filename='run_time')

    print('using the var_danger strategy ...')
    var_danger_time = strategy.var_danger(word_vector_path, dis_path, 1, 1, model_path)
    print('the time of var_danger strategy is', var_danger_time)
    library.record(words='the time of var_danger strategy is ' + str(var_danger_time), filename='run_time')

    print('using the random strategy ...')
    average_random_time = 0
    for epoch in range(random_epochs):
        random_time = strategy.random_sort(word_vector_path, model_path)
        average_random_time += random_time
    average_random_time /= random_epochs
    print('the time of random strategy is', average_random_time)
    library.record(words='the time of random strategy is ' + str(average_random_time), filename='run_time')

    total_var_time = dictionary_time + word_vector_time + distance_matrix_time + var_time
    total_danger_time = dictionary_time + word_vector_time + danger_time
    total_var_danger_time = dictionary_time + word_vector_time + distance_matrix_time + var_danger_time
    total_random_time = average_random_time
    return total_var_time, total_danger_time, total_var_danger_time, total_random_time


def run_for_effectiveness(model_path=None, data_path=None):
    var_apfd_list = []
    danger_apfd_list = []
    var_danger_apfd_list = []
    random_apfd_list = []

    if model_path is None or data_path is None:
        print("path isn't exist")
        return -1
    if os.path.exists(model_path):
        library.record(words='start calculate apfd ...', filename='apfd_value', mode='w')
        folders = os.listdir(model_path)
        for folder in folders:
            print('analysing the folder', folder, '...')
            library.record(words='the apfd in folder ' + folder, filename='apfd_value')
            folder_path = model_path + '/' + folder
            data_folder_path = data_path + '/' + folder
            word_dic_path = data_folder_path + '/' + 'word_dic.csv'
            word_vector_path = data_folder_path + '/' + 'word_vector.csv'
            dis_path = data_folder_path + '/' + 'div.csv'
            var_apfd, danger_apfd, var_danger_apfd, random_apfd = \
                run_for_effectiveness_with_one_folder(folder_path=folder_path, word_dic_path=word_dic_path,
                                                      word_vector_path=word_vector_path, dis_path=dis_path)
            var_apfd_list.append(var_apfd)
            danger_apfd_list.append(danger_apfd)
            var_danger_apfd_list.append(var_danger_apfd)
            random_apfd_list.append(random_apfd)
    library.record(words='the function has completed!', filename='apfd_value')
    library.record(words='var_apfd=' + str(var_apfd_list), filename='apfd_value')
    library.record(words='danger_apfd=' + str(danger_apfd_list), filename='apfd_value')
    library.record(words='var_danger_apfd=' + str(var_danger_apfd_list), filename='apfd_value')
    library.record(words='random_apfd=' + str(random_apfd_list), filename='apfd_value')
    return var_apfd_list, danger_apfd_list, var_danger_apfd_list, random_apfd_list


def run_for_efficiency(model_path=None, data_path=None):
    var_time_list = []
    danger_time_list = []
    var_danger_time_list = []
    random_time_list = []
    if model_path is None or data_path is None:
        print("path isn't exist")
        return -1
    if os.path.exists(model_path):
        library.record(words='start calculate time ...', filename='run_time', mode='w')
        folders = os.listdir(model_path)
        for folder in folders:
            print('analysing the folder', folder, '...')
            library.record(words='the time in folder ' + folder, filename='run_time')
            folder_path = model_path + '/' + folder
            data_folder_path = data_path + '/' + folder
            word_dic_path = data_folder_path + '/' + 'word_dic.csv'
            word_vector_path = data_folder_path + '/' + 'word_vector.csv'
            dis_path = data_folder_path + '/' + 'div.csv'
            var_time, danger_time, var_danger_time, random_time = \
                run_for_efficiency_with_one_folder(folder_path=folder_path, word_dic_path=word_dic_path,
                                                   word_vector_path=word_vector_path, dis_path=dis_path,
                                                   model_path=folder_path)
            var_time_list.append(var_time)
            danger_time_list.append(danger_time)
            var_danger_time_list.append(var_danger_time)
            random_time_list.append(random_time)
    library.record(words='the function has completed!', filename='run_time')
    library.record(words='var_time=' + str(var_time_list), filename='run_time')
    library.record(words='danger_time=' + str(danger_time_list), filename='run_time')
    library.record(words='var_danger_time=' + str(var_danger_time_list), filename='run_time')
    library.record(words='random_time=' + str(random_time_list), filename='run_time')
    return var_time_list, danger_time_list, var_danger_time_list, random_time_list


if __name__ == '__main__':
    run_for_efficiency('../model', '../data')
