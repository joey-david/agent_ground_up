def count_pairs(xs, target):
    n = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] + xs[j] == target:
                n += 1
    return n
