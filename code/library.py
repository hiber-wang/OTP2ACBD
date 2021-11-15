import dataset
import time
import pandas as pd


def find_bug(filename, bug_cat):
    if filename in dataset.bug0:
        if '0' in bug_cat:
            bug_cat.remove('0')
            return 1
    elif filename in dataset.bug1:
        if '1' in bug_cat:
            bug_cat.remove('1')
            return 1
    elif filename in dataset.bug2:
        if '2' in bug_cat:
            bug_cat.remove('2')
            return 1
    elif filename in dataset.bug3:
        if '3' in bug_cat:
            bug_cat.remove('3')
            return 1
    elif filename in dataset.bug4:
        if '4' in bug_cat:
            bug_cat.remove('4')
            return 1
    elif filename in dataset.bug5:
        if '5' in bug_cat:
            bug_cat.remove('5')
            return 1
    elif filename in dataset.bug6:
        if '6' in bug_cat:
            bug_cat.remove('6')
            return 1
    elif filename in dataset.bug7:
        if '7' in bug_cat:
            bug_cat.remove('7')
            return 1
    elif filename in dataset.bug8:
        if '8' in bug_cat:
            bug_cat.remove('8')
            return 1
    elif filename in dataset.bug9:
        if '9' in bug_cat:
            bug_cat.remove('9')
            return 1
    elif filename in dataset.buga:
        if 'a' in bug_cat:
            bug_cat.remove('a')
            return 1
    elif filename in dataset.bugb:
        if 'b' in bug_cat:
            bug_cat.remove('b')
            return 1
    elif filename in dataset.bugc:
        if 'c' in bug_cat:
            bug_cat.remove('c')
            return 1
    elif filename in dataset.bugd:
        if 'd' in bug_cat:
            bug_cat.remove('d')
            return 1
    elif filename in dataset.buge:
        if 'e' in bug_cat:
            bug_cat.remove('e')
            return 1
    elif filename in dataset.bugf:
        if 'f' in bug_cat:
            bug_cat.remove('f')
            return 1
    elif filename in dataset.bugg:
        if 'g' in bug_cat:
            bug_cat.remove('g')
            return 1
    elif filename in dataset.bugh:
        if 'h' in bug_cat:
            bug_cat.remove('h')
            return 1
    elif filename in dataset.bugi:
        if 'i' in bug_cat:
            bug_cat.remove('i')
            return 1
    elif filename in dataset.bugj:
        if 'j' in bug_cat:
            bug_cat.remove('j')
            return 1
    elif filename in dataset.bugk:
        if 'k' in bug_cat:
            bug_cat.remove('k')
            return 1
    elif filename in dataset.bugl:
        if 'l' in bug_cat:
            bug_cat.remove('l')
            return 1
    return 0


def evaluate(ordered_sequence):
    tf = 0
    M = 0
    bug_category = []
    for i in range(len(ordered_sequence)):
        if ordered_sequence[i] in dataset.bug0:
            if '0' not in bug_category:
                bug_category.append('0')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug1:
            if '1' not in bug_category:
                bug_category.append('1')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug2:
            if '2' not in bug_category:
                bug_category.append('2')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug3:
            if '3' not in bug_category:
                bug_category.append('3')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug4:
            if '4' not in bug_category:
                bug_category.append('4')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug5:
            if '5' not in bug_category:
                bug_category.append('5')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug6:
            if '6' not in bug_category:
                bug_category.append('6')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug7:
            if '7' not in bug_category:
                bug_category.append('7')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug8:
            if '8' not in bug_category:
                bug_category.append('8')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bug9:
            if '9' not in bug_category:
                bug_category.append('9')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.buga:
            if 'a' not in bug_category:
                bug_category.append('a')
                tf += i
                M += 1

        elif ordered_sequence[i] in dataset.bugb:
            if 'b' not in bug_category:
                bug_category.append('b')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugc:
            if 'c' not in bug_category:
                bug_category.append('c')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugd:
            if 'd' not in bug_category:
                bug_category.append('d')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.buge:
            if 'e' not in bug_category:
                bug_category.append('e')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugf:
            if 'f' not in bug_category:
                bug_category.append('f')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugg:
            if 'g' not in bug_category:
                bug_category.append('g')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugh:
            if 'h' not in bug_category:
                bug_category.append('h')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugi:
            if 'i' not in bug_category:
                bug_category.append('i')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugj:
            if 'j' not in bug_category:
                bug_category.append('j')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugk:
            if 'k' not in bug_category:
                bug_category.append('k')
                tf += i
                M += 1
        elif ordered_sequence[i] in dataset.bugl:
            if 'l' not in bug_category:
                bug_category.append('l')
                tf += i
                M += 1
    APFD = 1 - tf / (len(ordered_sequence) * M) + 1 / (2 * len(ordered_sequence))
    return APFD


def get_window_size(word_vector_path):
    window_size = []
    word_vector = pd.read_csv(word_vector_path)
    bug_number = 0
    for i in range(len(word_vector)):
        if word_vector.iloc[i, 0][0] == '0':
            bug_number += 1
    for size_rate in range(1, 21):
        window_size.append(int(0.05 * size_rate * len(word_vector)))
    return window_size, len(word_vector), bug_number


def record(words='', filename=None, record_path='../record', mode='a'):
    if filename is None:
        filename = str(time.strftime("%Y-%m-%d-%H-%M-%S"))
    filename = record_path + '/' + filename + '.txt'
    file = open(filename, mode)
    file.write(words + '\n')
    file.close()
