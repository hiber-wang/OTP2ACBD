import dataset


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