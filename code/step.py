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


def generate_word_vector_table(folder_path=None, word_dic_path=None, word_vector_path=None,
                               metric_path=None, block_flag=True, line_flag=True, metric_flag=True):
    if folder_path is None or word_dic_path is None or word_vector_path is None or metric_path is None:
        print("Error: path isn't exist")
        return -1
    print('building word vector...')
    start = time.time()
    word_vector = modeling.build_word_vector(folder_path, word_dic_path, metric_path, block_flag=block_flag,
                                             line_flag=line_flag, metric_flag=metric_flag)
    word_vector = pd.DataFrame.from_dict(word_vector)
    word_vector = word_vector.T
    word_vector.to_csv(word_vector_path)
    end = time.time()
    print('building word vector:done')
    return end - start


def generate_distance_matrix(word_vector_path=None, dis_path=None, dis_type='m'):
    if word_vector_path is None or dis_path is None:
        print("Error: path isn't exist")
        return -1
    print('building distance matrix...')
    start = time.time()
    div, block_list = modeling.build_dis(word_vector_path, dis_type)
    div = pd.DataFrame(div, columns=block_list, index=block_list)
    div.to_csv(dis_path)
    print("building distance matrix:done")
    end = time.time()
    return end - start


def run_for_distance_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None, dis_path=None,
                                     metric_path=None):
    if folder_path is None or word_dic_path is None or word_vector_path is None or dis_path is None or metric_path is None:
        print("path isn't exist")
        return -1
    generate_dictionary_table(folder_path, word_dic_path)
    generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path)
    print('using Manhattan distance...')
    generate_distance_matrix(word_vector_path, dis_path, dis_type='m')
    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    manhattan_apfd = library.evaluate(QTR)
    print('the apfd of Manhattan is', manhattan_apfd)
    library.record(words='the apfd of Manhattan is ' + str(manhattan_apfd), filename='distance_value')

    print('using Euclidean distance...')
    generate_distance_matrix(word_vector_path, dis_path, dis_type='e')
    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    euclidean_apfd = library.evaluate(QTR)
    print('the apfd of Euclidean is', euclidean_apfd)
    library.record(words='the apfd of Euclidean is ' + str(euclidean_apfd), filename='distance_value')

    print('using Cosine distance...')
    generate_distance_matrix(word_vector_path, dis_path, dis_type='c')
    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    cosine_apfd = library.evaluate(QTR)
    print('the apfd of Cosine is', cosine_apfd)
    library.record(words='the apfd of Cosine is ' + str(cosine_apfd), filename='distance_value')
    return manhattan_apfd, euclidean_apfd, cosine_apfd


def run_for_distance(model_path=None, data_path=None, metric_path=None):
    manhattan_apfd_list = []
    euclidean_apfd_list = []
    cosine_apfd_list = []
    if model_path is None or data_path is None:
        print("path isn't exist")
        return -1
    if os.path.exists(model_path):
        library.record(words='start calculate distance ...', filename='distance_value', mode='w')
        folders = os.listdir(model_path)
        for folder in folders:
            print('analysing the folder', folder, '...')
            library.record(words='the apfd in folder ' + folder, filename='distance_value')
            folder_path = model_path + '/' + folder
            data_folder_path = data_path + '/' + folder
            word_dic_path = data_folder_path + '/' + 'word_dic.csv'
            word_vector_path = data_folder_path + '/' + 'word_vector.csv'
            dis_path = data_folder_path + '/' + 'div.csv'
            manhattan_apfd, euclidean_apfd, cosine_apfd = run_for_distance_with_one_folder(folder_path, word_dic_path,
                                                                                           word_vector_path, dis_path,
                                                                                           metric_path)
            manhattan_apfd_list.append(manhattan_apfd)
            euclidean_apfd_list.append(euclidean_apfd)
            cosine_apfd_list.append(cosine_apfd)
    library.record(words='the function has completed!', filename='distance_value')
    library.record(words='manhattan_apfd=' + str(manhattan_apfd_list), filename='distance_value')
    library.record(words='euclidean_apfd=' + str(euclidean_apfd_list), filename='distance_value')
    library.record(words='cosine_apfd=' + str(cosine_apfd_list), filename='distance_value')
    return manhattan_apfd_list, euclidean_apfd_list, cosine_apfd_list


def distance_strategy_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None,
                                      dis_path=None, data_flag=True, epochs=100, metric_path=None):
    if folder_path is None or word_dic_path is None or word_vector_path is None or dis_path is None:
        print("path isn't exist")
        return -1
    generate_dictionary_table(folder_path, word_dic_path)
    generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path)
    _, total_number, _ = library.get_window_size(word_vector_path)

    print('using the manhattan strategy ...')
    generate_distance_matrix(word_vector_path, dis_path, dis_type='m')
    QTR = strategy.var_fast_sort(word_vector_path, dis_path, total_number)
    manhattan_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of manhattan strategy is', manhattan_percent)
    library.record(words='the percent of manhattan strategy is ' + str(manhattan_percent), filename='dis_percent_value')

    print('using the euclidean strategy ...')
    generate_distance_matrix(word_vector_path, dis_path, dis_type='e')
    QTR = strategy.var_fast_sort(word_vector_path, dis_path, total_number)
    euclidean_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of euclidean strategy is', euclidean_percent)
    library.record(words='the percent of euclidean strategy is ' + str(euclidean_percent), filename='dis_percent_value')

    print('using the cosine strategy ...')
    generate_distance_matrix(word_vector_path, dis_path, dis_type='c')
    QTR = strategy.var_fast_sort(word_vector_path, dis_path, total_number)
    cosine_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of cosine strategy is', cosine_percent)
    library.record(words='the percent of cosine strategy is ' + str(cosine_percent), filename='dis_percent_value')

    return manhattan_percent, euclidean_percent, cosine_percent


def distance_strategy(model_path=None, data_path=None, metric_path=None):
    manhattan_percent_list = []
    euclidean_percent_list = []
    cosine_percent_list = []

    if model_path is None or data_path is None:
        print("path isn't exist")
        return -1
    if os.path.exists(model_path):
        library.record(words='start calculate percent ...', filename='dis_percent_value', mode='w')
        folders = os.listdir(model_path)
        for folder in folders:
            print('analysing the folder', folder, '...')
            library.record(words='the percent in folder ' + folder, filename='dis_percent_value')
            folder_path = model_path + '/' + folder
            data_folder_path = data_path + '/' + folder
            word_dic_path = data_folder_path + '/' + 'word_dic.csv'
            word_vector_path = data_folder_path + '/' + 'word_vector.csv'
            dis_path = data_folder_path + '/' + 'div.csv'
            manhattan_percent, euclidean_percent, cosine_percent = distance_strategy_with_one_folder(
                folder_path=folder_path, word_dic_path=word_dic_path,
                word_vector_path=word_vector_path, dis_path=dis_path,
                metric_path=metric_path, data_flag=False)
            manhattan_percent_list.append(manhattan_percent)
            euclidean_percent_list.append(euclidean_percent)
            cosine_percent_list.append(cosine_percent)
    library.record(words='the function has completed!', filename='dis_percent_value')
    library.record(words='manhattan_percent=' + str(manhattan_percent), filename='dis_percent_value')
    library.record(words='euclidean_percent=' + str(euclidean_percent), filename='dis_percent_value')
    library.record(words='cosine_percent=' + str(cosine_percent), filename='dis_percent_value')
    return manhattan_percent_list, euclidean_percent_list, cosine_percent_list


def run_for_percentage_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None, dis_path=None,
                                       data_flag=False, epochs=100, metric_path=None):
    if folder_path is None or word_dic_path is None or word_vector_path is None or dis_path is None:
        print("path isn't exist")
        return -1
    if data_flag:
        generate_dictionary_table(folder_path, word_dic_path)
        generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path)
        generate_distance_matrix(word_vector_path, dis_path)
    _, total_number, _ = library.get_window_size(word_vector_path)

    print('using the var strategy ...')
    QTR = strategy.var_fast_sort(word_vector_path, dis_path, total_number)
    var_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of var strategy is', var_percent)
    library.record(words='the percent of var strategy is ' + str(var_percent), filename='percent_value')

    print('using the danger strategy ...')
    QTR = strategy.danger_fast_test(word_dic_path, word_vector_path, 1)
    danger_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of danger strategy is', danger_percent)
    library.record(words='the percent of danger strategy is ' + str(danger_percent), filename='percent_value')

    print('using the var_danger strategy ...')
    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    var_danger_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of var_danger strategy is', var_danger_percent)
    library.record(words='the percent of var_danger strategy is ' + str(var_danger_percent), filename='percent_value')

    print('using the greedy strategy ...')
    QTR = strategy.greedy_fast_test(word_vector_path)
    greedy_percent = library.evaluate_percent(QTR, total_number)
    print('the percent of danger strategy is', greedy_percent)
    library.record(words='the percent of greedy strategy is ' + str(greedy_percent), filename='percent_value')

    print('using the local_beam_search_strategy ...')
    bug_number = len(greedy_percent)
    average_local_beam_search_percent = [0] * bug_number
    for epoch in range(epochs):
        QTR = strategy.local_beam_search_fast_test(word_vector_path, dis_path, 5)
        local_beam_search_percent = library.evaluate_percent(QTR, total_number)
        for i in range(bug_number):
            average_local_beam_search_percent[i] += local_beam_search_percent[i]
    for i in range(bug_number):
        average_local_beam_search_percent[i] /= epochs
    print('the percent of local_beam_search strategy is', average_local_beam_search_percent)
    library.record(words='the percent of local_beam_search strategy is ' + str(average_local_beam_search_percent),
                   filename='percent_value')

    print('using the random_strategy ...')
    bug_number = len(greedy_percent)
    average_random_percent = [0] * bug_number
    for epoch in range(epochs):
        QTR = strategy.random_fast_test(word_vector_path)
        random_percent = library.evaluate_percent(QTR, total_number)
        for i in range(bug_number):
            average_random_percent[i] += random_percent[i]
    for i in range(bug_number):
        average_random_percent[i] /= epochs
    print('the percent of random strategy is', average_random_percent)
    library.record(words='the percent of random strategy is ' + str(average_random_percent), filename='percent_value')
    return var_percent, danger_percent, var_danger_percent, greedy_percent, average_local_beam_search_percent, average_random_percent


def run_for_percentage(model_path=None, data_path=None, metric_path=None):
    var_percent_list = []
    danger_percent_list = []
    var_danger_percent_list = []
    greedy_percent_list = []
    random_percent_list = []
    local_beam_search_percent_list = []

    if model_path is None or data_path is None:
        print("path isn't exist")
        return -1
    if os.path.exists(model_path):
        library.record(words='start calculate percent ...', filename='percent_value', mode='w')
        folders = os.listdir(model_path)
        for folder in folders:
            print('analysing the folder', folder, '...')
            library.record(words='the percent in folder ' + folder, filename='percent_value')
            folder_path = model_path + '/' + folder
            data_folder_path = data_path + '/' + folder
            word_dic_path = data_folder_path + '/' + 'word_dic.csv'
            word_vector_path = data_folder_path + '/' + 'word_vector.csv'
            dis_path = data_folder_path + '/' + 'div.csv'
            var_percent, danger_percent, var_danger_percent, greedy_percent, local_beam_search_percent, random_percent = \
                run_for_percentage_with_one_folder(folder_path=folder_path, word_dic_path=word_dic_path,
                                                   word_vector_path=word_vector_path, dis_path=dis_path,
                                                   metric_path=metric_path, data_flag=False)
            var_percent_list.append(var_percent)
            danger_percent_list.append(danger_percent)
            var_danger_percent_list.append(var_danger_percent)
            greedy_percent_list.append(greedy_percent)
            local_beam_search_percent_list.append(local_beam_search_percent)
            random_percent_list.append(random_percent)
    library.record(words='the function has completed!', filename='percent_value')
    library.record(words='var_percent=' + str(var_percent_list), filename='percent_value')
    library.record(words='danger_percent=' + str(danger_percent_list), filename='percent_value')
    library.record(words='var_danger_percent=' + str(var_danger_percent_list), filename='percent_value')
    library.record(words='greedy_percent=' + str(greedy_percent_list), filename='percent_value')
    library.record(words='local_beam_search_percent=' + str(local_beam_search_percent_list), filename='percent_value')
    library.record(words='random_percent=' + str(random_percent_list), filename='percent_value')
    return var_percent_list, danger_percent_list, var_danger_percent_list, greedy_percent_list, local_beam_search_percent_list, random_percent_list


def run_for_effectiveness_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None, dis_path=None,
                                          data_flag=True, random_epochs=100, metric_path=None):
    if folder_path is None or word_dic_path is None or word_vector_path is None or dis_path is None:
        print("path isn't exist")
        return -1
    if data_flag:
        generate_dictionary_table(folder_path, word_dic_path)
        generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path)
        generate_distance_matrix(word_vector_path, dis_path)
    _, total_number, _ = library.get_window_size(word_vector_path)

    print('using the var strategy ...')
    QTR = strategy.var_fast_sort(word_vector_path, dis_path, total_number)
    var_apfd = library.evaluate(QTR)
    print('the apfd of var strategy is', var_apfd)
    library.record(words='the apfd of var strategy is ' + str(var_apfd), filename='apfd_value')

    print('using the danger strategy ...')
    QTR = strategy.danger_fast_test(word_dic_path, word_vector_path, 1)
    danger_apfd = library.evaluate(QTR)
    print('the apfd of danger strategy is', danger_apfd)
    library.record(words='the apfd of danger strategy is ' + str(danger_apfd), filename='apfd_value')

    print('using the var_danger strategy ...')
    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    var_danger_apfd = library.evaluate(QTR)
    print('the apfd of var_danger strategy is', var_danger_apfd)
    library.record(words='the apfd of var_danger strategy is ' + str(var_danger_apfd), filename='apfd_value')

    print('using the greedy strategy ...')
    QTR = strategy.greedy_fast_test(word_vector_path)
    greedy_apfd = library.evaluate(QTR)
    print('the apfd of danger strategy is', greedy_apfd)
    library.record(words='the apfd of greedy strategy is ' + str(greedy_apfd), filename='apfd_value')

    print('using the random_strategy ...')
    average_random_apfd = 0
    for epoch in range(random_epochs):
        QTR = strategy.random_fast_test(word_vector_path)
        random_apfd = library.evaluate(QTR)
        average_random_apfd += random_apfd
    average_random_apfd /= random_epochs
    print('the apfd of random strategy is', average_random_apfd)
    library.record(words='the apfd of random strategy is ' + str(average_random_apfd), filename='apfd_value')
    return var_apfd, danger_apfd, var_danger_apfd, greedy_apfd, average_random_apfd


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


def run_for_effectiveness(model_path=None, data_path=None, metric_path=None):
    var_apfd_list = []
    danger_apfd_list = []
    var_danger_apfd_list = []
    greedy_apfd_list = []
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
            var_apfd, danger_apfd, var_danger_apfd, greedy_apfd, random_apfd = \
                run_for_effectiveness_with_one_folder(folder_path=folder_path, word_dic_path=word_dic_path,
                                                      word_vector_path=word_vector_path, dis_path=dis_path,
                                                      metric_path=metric_path, data_flag=True)
            var_apfd_list.append(var_apfd)
            danger_apfd_list.append(danger_apfd)
            var_danger_apfd_list.append(var_danger_apfd)
            greedy_apfd_list.append(greedy_apfd)
            random_apfd_list.append(random_apfd)
    library.record(words='the function has completed!', filename='apfd_value')
    library.record(words='var_apfd=' + str(var_apfd_list), filename='apfd_value')
    library.record(words='danger_apfd=' + str(danger_apfd_list), filename='apfd_value')
    library.record(words='var_danger_apfd=' + str(var_danger_apfd_list), filename='apfd_value')
    library.record(words='greedy_apfd=' + str(greedy_apfd_list), filename='apfd_value')
    library.record(words='random_apfd=' + str(random_apfd_list), filename='apfd_value')
    return var_apfd_list, danger_apfd_list, var_danger_apfd_list, greedy_apfd_list, random_apfd_list


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


def compare_different_features_with_one_folder(folder_path=None, word_dic_path=None, word_vector_path=None,
                                               dis_path=None, metric_path=None):
    print('use the feature of block...')
    library.record(words='use the feature of block', filename='compare_features')
    generate_dictionary_table(folder_path, word_dic_path)
    generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path, line_flag=False,
                               metric_flag=False)
    generate_distance_matrix(word_vector_path, dis_path)

    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    block_apfd = library.evaluate(QTR)
    print('the apfd of block features is', block_apfd)
    library.record(words='the apfd of block features is ' + str(block_apfd), filename='compare_features')

    print('use the feature of line...')
    library.record(words='use the feature of line', filename='compare_features')
    generate_dictionary_table(folder_path, word_dic_path)
    generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path, block_flag=False,
                               metric_flag=False)
    generate_distance_matrix(word_vector_path, dis_path)

    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    line_apfd = library.evaluate(QTR)
    print('the apfd of line features is', line_apfd)
    library.record(words='the apfd of line features is ' + str(line_apfd), filename='compare_features')

    print('use the feature of metric...')
    library.record(words='use the feature of metric', filename='compare_features')
    generate_dictionary_table(folder_path, word_dic_path)
    generate_word_vector_table(folder_path, word_dic_path, word_vector_path, metric_path, block_flag=False,
                               line_flag=False)
    generate_distance_matrix(word_vector_path, dis_path)

    QTR = strategy.var_danger_fast_test(word_dic_path, word_vector_path, dis_path, 1, 1)
    metric_apfd = library.evaluate(QTR)
    print('the apfd of metric features is', metric_apfd)
    library.record(words='the apfd of metric features is ' + str(metric_apfd), filename='compare_features')

    return block_apfd, line_apfd, metric_apfd


def compare_different_features(model_path=None, data_path=None, metric_path=None):
    block_apfd_list = []
    line_apfd_list = []
    metric_apfd_list = []
    if model_path is None or data_path is None:
        print("path isn't exist")
        return -1
    if os.path.exists(model_path):
        library.record(words='start compare features ...', filename='compare_features', mode='w')
        folders = os.listdir(model_path)
        for folder in folders:
            if folder != '95':
                continue
            print('analysing the folder', folder, '...')
            library.record(words='the apfd in folder ' + folder, filename='compare_features')
            folder_path = model_path + '/' + folder
            data_folder_path = data_path + '/' + folder
            word_dic_path = data_folder_path + '/' + 'word_dic.csv'
            word_vector_path = data_folder_path + '/' + 'word_vector.csv'
            dis_path = data_folder_path + '/' + 'div.csv'
            block_apfd, line_apfd, metric_apfd = compare_different_features_with_one_folder(folder_path, word_dic_path,
                                                                                            word_vector_path, dis_path,
                                                                                            metric_path)
            block_apfd_list.append(block_apfd)
            line_apfd_list.append(line_apfd)
            metric_apfd_list.append(metric_apfd)
    library.record(words='the function has completed!', filename='compare_features')
    library.record(words='block_apfd=' + str(block_apfd_list), filename='compare_features')
    library.record(words='line_apfd=' + str(line_apfd_list), filename='compare_features')
    library.record(words='metric_apfd=' + str(metric_apfd_list), filename='compare_features')
    return block_apfd_list, line_apfd_list, metric_apfd_list


if __name__ == '__main__':
    run_for_effectiveness('../model', '../data', '../data/metrics.csv')
    # word_vector_path = '../data_test/word_vector.csv'
    # dis_path = '../data_test/div.csv'
    # QTR = strategy.var_fast_sort('../data_test/word_vector.csv', '../data_test/div.csv', 8)
    # print(QTR)
    # distance_strategy('../model', '../data', '../data/metrics.csv')
    # compare_different_features('../model', '../data', '../data/metrics.csv')
    # print('using the local_beam_search_strategy ...')
    # folders = ['50', '60', '70', '80', '90', '95']
    # local_beam_search_percent_list = []
    # for folder in folders:
    #     _, total_number, _ = library.get_window_size('../data/' + folder + '/word_vector.csv')
    #     bug_number = 22
    #     average_local_beam_search_percent = [0] * bug_number
    #     for epoch in range(10):
    #         QTR = strategy.local_beam_search_fast_test('../data/' + folder + '/word_vector.csv', '../data/' + folder + '/div.csv', 5)
    #         local_beam_search_percent = library.evaluate_percent(QTR, total_number)
    #         for i in range(bug_number):
    #             average_local_beam_search_percent[i] += local_beam_search_percent[i]
    #     for i in range(bug_number):
    #         average_local_beam_search_percent[i] /= 10
    #     local_beam_search_percent_list.append(average_local_beam_search_percent)
    #     print(folder, 'the percent of local_beam_search strategy is', average_local_beam_search_percent)
    # print(local_beam_search_percent_list)

    # run_for_percentage('../model', '../data', '../data/metrics.csv')
    # run_for_effectiveness('../model', '../data', '../data/metrics.csv')
    # compare_different_features('../model', '../data_for_compare_feature', '../data_for_compare_feature/metrics.csv')
