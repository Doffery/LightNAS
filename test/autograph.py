import tensorflow as tf
from tensorflow.contrib import autograph
import numpy as np

tf.enable_eager_execution()
opt_ind = 3#self.cd_opt_ind
end_ind = 2#self.cd_end_ind
cd_length = 4
num_cells =7 
dag = [0 ]
# dag = np.zeros((self.num_cells*self.cd_length), dtype=np.int32)
pre_op_idx = -1
i = 0

def _path2dag(path, opt_ind, end_ind, cd_length, num_cells, dag, pre_op_idx, i):
    dag = []
    autograph.set_element_type(dag, tf.int32)
    for x in range(5):
        dag.append(x)
    # logger.info(path)
    # logger.info(np.reshape(dag, (self.num_cells, self.cd_length)))
    return dag

print(autograph.to_code(_path2dag))
'''
    for op in path:
        start_idx = i*cd_length
        if op == 0:
            dag[start_idx] = 2
        else:
            dag[start_idx+opt_ind] = op
            dag[start_idx+opt_ind-num_cells+pre_op_idx+1] = 1
            if pre_op_idx != -1:
                dag[(pre_op_idx+1)*cd_length-1] = 0
            dag[start_idx+end_ind] = 1
            pre_op_idx = i
        i += 1
'''
