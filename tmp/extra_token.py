import pandas
import numpy as np
#有sorcetype sourceblock
#有 If
#有SubSystem
#line里面 如果srcport 中有和 disport中 有重复的
# class block:
#     def __init__(self):
#         self.num = 0
#         self.type = ""
# def start_extra():
#
# def readFile():
#     lines = open('.txt').readlines()
#     for i, line in enumerate(lines):
#         if line in "      BlockType":
#
#             break
def read_vec_list():
    df = pandas.read_csv('word_vector4.csv')
    del df['name']
    return list(df.values)

def read_vec_numpy():
    df = pandas.read_csv('word_vector4.csv')
    del df['name']
    exs = np.array(df.values)
    return exs
# lines = open('word_vector.csv', 'rb', encoding='utf-8').readlines()
# print(lines)