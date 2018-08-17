import numpy as np
import tensorflow as tf
num_cells = 5

def _path2dag(path):
    dag = np.zeros((num_cells*(num_cells+2)), dtype=np.int32)
    pre_op_idx = 0
    for i, op in enumerate(path):
        start_idx = i*(num_cells+2)
        if op == 0:
            dag[start_idx] = 2
        else:
            dag[start_idx+num_cells] = op
            dag[start_idx+pre_op_idx] = 1
            if pre_op_idx != 0:
                dag[pre_op_idx*(num_cells+2)-1] = 0
            dag[start_idx+num_cells+1] = 1
            pre_op_idx = i+1
    # return dag.reshape((num_cells, num_cells+2))
    return dag

print(_path2dag([2,0,3,1,0]))

def _merge_dag(da, db):
    d2a = np.reshape(da, (num_cells, (num_cells+2)))
    d2b = np.reshape(db, (num_cells, (num_cells+2)))
    d2c = np.zeros((num_cells, num_cells+2), dtype=np.int32)
    for ind in range(num_cells):
        if d2a[ind][num_cells] == 0 and \
                d2b[ind][num_cells] == 0:
            d2c[ind][num_cells] = 0
            d2c[ind][0] = 2
            continue
        if d2a[ind][num_cells] == 0:
            d2a[ind][0] = 0
        if d2b[ind][num_cells] == 0:
            d2b[ind][0] = 0
        for jnd in range(num_cells):
            if d2a[ind][jnd] != 0 or d2b[ind][jnd] != 0:
                d2c[ind][jnd] = 1
        # How do we decide the operator?
        # Choose the first, the better?
        if d2a[ind][num_cells] != 0:
            d2c[ind][num_cells] = d2a[ind][num_cells]
        else:
            d2c[ind][num_cells] = d2b[ind][num_cells]

        # is End or not
        if d2a[ind][num_cells+1] == 1 or \
                d2b[ind][num_cells+1] == 1:
            d2c[ind][num_cells+1] = 1
    return d2c .flatten()

print(_merge_dag(_path2dag([2,0,3,1,0]), _path2dag([2,0,3,1,0])))

def _to_dag( ops, path):
    ops = tf.multiply(ops, path)
    layers = []
    for i in range(num_cells):
        pre_layer = tf.one_hot(0, num_cells, dtype=tf.int32)
        for j in range(i):
            pre_layer = tf.cond(tf.equal(path[j], 0),
                   lambda: pre_layer,
                   lambda: tf.one_hot(j+1, num_cells, dtype=tf.int32))
        tmp = tf.add(tf.one_hot(0, num_cells, dtype=tf.int32),
                    tf.one_hot(0, num_cells, dtype=tf.int32))
        pre_layer = tf.cond(tf.equal(path[i], 0),
               lambda: tmp, lambda: pre_layer)
        end = tf.cond(tf.equal(path[i], 0),
                lambda: tf.constant([0], dtype=tf.int32),
                lambda: tf.constant([1], dtype=tf.int32))
        for k in range(i+1, num_cells):
            end = tf.cond(tf.equal(path[k], 0),
                    lambda: end, lambda: tf.constant([0], dtype=tf.int32))
        layers.append(tf.concat([pre_layer, ops[i:i+1], end], 0))

    return tf.stack(layers)

#with tf.Session() as ses:
#    print(ses.run(_to_dag([2,1,3,1,3], [0,0,0,1,0])))

