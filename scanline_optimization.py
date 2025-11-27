#!/usr/bin/env python3
import math

def dissimilarity(x, d):
    # TODO: similarity functie van noemi oproepen
    return -similarity(x, d)

def regularisation_cost(dx, prev):
    # TODO: dit moet met matrices rekenen en niet gewoon getallen
    if dx == prev:
        return 1
    elif abs(dx - prev) == 1:
        return 0
    else:
        return abs(dx - prev) -1



def score(x, d, weight):
    args = []
    for dx in d:
        tussen = math.inf
        for xi in x:
            dissimilarity(xi, dx) + weight * regularisation_cost(dx, dx-1)
