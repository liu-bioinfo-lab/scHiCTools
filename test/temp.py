def change(f1, f2):
    f = open(f2, 'w')
    for line in open(f1, 'r'):
        lst = line.strip().split()
        if lst[0] == '20':
            lst[0] = 'X'
        if lst[0] == '21':
            lst[0] = 'Y'
        if lst[2] == '20':
            lst[2] = 'X'
        if lst[2] == '21':
            lst[2] = 'Y'
        f.write(' '.join(lst) + '\n')
    f.close()

change('cell_3', 'cell_03')

