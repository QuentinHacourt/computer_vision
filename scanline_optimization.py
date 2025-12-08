def delta(x, y):
    return abs(x - y)

def regularisation_cost(dx, prev):
    x = delta(dx, prev)
    if x == 1: return 0
    elif dx == prev: return 1
    else: return x - 1
