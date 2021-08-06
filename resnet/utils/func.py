import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item==0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d:d[1], reverse=True)
    return count_dict[0][0]
