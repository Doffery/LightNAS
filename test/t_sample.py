import random
import numpy as np

opt_num = 4
def _sample_path(length, num, num_cells):
    if num_cells*length*2 <= num:
        num = num_cells*length*2
        return _sample_path(length, num)
    sample_dict = {}
    while num:
        sample_ind = random.sample(range(num_cells), length)
        path = np.zeros(num_cells, dtype=np.int32)
        for ind in sample_ind:
            path[ind] = random.randint(0, opt_num)+1
        path_str = np.array2string(path)
        if path_str not in sample_dict:
            sample_dict[path_str] = path
            num -= 1
    return [t[1] for t in sample_dict.items()]

print(_sample_path(1, 6, 5))
print(_sample_path(2, 10, 5))
print(_sample_path(3, 15, 5))
