import lightnas.src.utils as utils
import random
import numpy as np
import t_path2dag

max_generation = 10
opt_num = 5
k_init_selection = 3
k_best_selection = 3
num_cells = 5

def test_evolving():
    path_pool = np.array([[1, 2, 0, 2, 0], [2, 2, 0, 2, 0], [3, 2, 0, 2, 0], [4, 2, 0, 2, 0]])
    path_pool_acc = np.array([0.1, 0.2, 0.3, 0.4])

    def _is_new_path(path):
        print("Checking new {}".format(path))
        for p in path_pool:
            print(p == path)
            if (p == path).all():
                print(p)
                return False
        return True

    print(_is_new_path(np.array([1, 2, 0, 2, 0])))

    # start evolving
    evolve_iter = 0
    while evolve_iter < max_generation:
        evolve_iter += 1
        # select top-k
        top_k_ind_acc = utils.find_top_k_ind(path_pool_acc,
                             k_init_selection)
        seeds = [x for x, _ in top_k_ind_acc]
        candidate_paths = []

        # apply mutation
        for ind in seeds:
            for i in range(num_cells):
                tmp_path = np.copy(path_pool[ind])
                tmp_path[i] = np.random.randint(0, opt_num+1)
                # if _check_path(tmp_path):
                print(_is_new_path(tmp_path))
                candidate_paths.append(tmp_path)
        print("Candi:{}".format(candidate_paths))

        # apply crossover
        # for ind1, acc1 in top_k_ind_acc:
        #     tmp_path1 = path_pool[ind1]
        #     for ind2, acc2 in top_k_ind_acc:
        #         tmp_path2 = path_pool[ind2]

        # predict and select best k
        candidate_accs = []
        for i, cpath in enumerate(candidate_paths):
            # cdag = _path2dag(cpath)
            # feed_dict = {child_ops["dag_arc"]: cdag}
            valid_acc = random.randint(0, 10) / 10
            candidate_accs.append(valid_acc)

        top_k_candidates = utils.find_top_k_ind(candidate_accs,
                            k_best_selection)

        # replace the worse with candidates
        for tk_ind, tk_acc in top_k_candidates:
            path_pool = np.append(path_pool, [candidate_paths[tk_ind]], axis=0)
            path_pool_acc = np.append(path_pool_acc, [tk_acc])
        bad_k_paths = utils.find_rtop_k_ind(path_pool_acc,
                            len(top_k_candidates))
        del_inx = [ind for ind, _ in bad_k_paths]
        # del_inx.sort(reverse=True)
        # for d_ind in del_inx:
        #     del path_pool[d_ind]
        #     del path_pool_acc[d_ind]
        tmp_path_pool = []
        tmp_path_pool_acc = []
        print(len(path_pool))
        print(len(top_k_candidates))
        for ind in range(len(path_pool)):
            if ind not in del_inx:
                print(ind, path_pool[ind])
                tmp_path_pool.append(path_pool[ind])
                tmp_path_pool_acc.append(path_pool_acc[ind])
        path_pool = np.array(tmp_path_pool)
        path_pool_acc = np.array(tmp_path_pool_acc)
        print(path_pool)
        print(path_pool_acc)


    # build dags from paths

# test_evolving()


path_pool = [  [2, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 1, 0,
                0, 1, 0, 0, 0, 3, 0,
                2, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 2, 1, ],
               [2, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 1, 0,
                2, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 2, 1,
                ]]

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
    return d2c# .flatten()

print(_merge_dag(path_pool[0], path_pool[1]))


def test_greedy_merge():
    path_pool = np.array([[1, 2, 0, 2, 0], [2, 2, 0, 2, 0], [3, 2, 0, 2, 0], [4, 2, 0, 2, 0]])
    path_pool_acc = np.array([0.1, 0.2, 0.3, 0.4])

    being_better = True
    sort_acc = sorted(enumerate(path_pool_acc), 
                      key=lambda p: p[1], reverse=True)
    check_list = np.zeros((len(path_pool)), dtype=np.bool)
    check_list[0] = True
    final_dag = t_path2dag._path2dag(path_pool[sort_acc[0][0]])
    final_acc = path_pool_acc[sort_acc[0][0]]
    while being_better:
        choose_dag = final_dag
        choose_acc = final_acc
        choose_ind = 0
        being_better = False
        for ind in range(1, len(sort_acc)):
            if check_list[ind]:
                continue
            tmp_dag = _merge_dag(final_dag, t_path2dag._path2dag(path_pool[sort_acc[ind][0]]))
            valid_acc = random.uniform(0, 1)
            print("Test {0}, acc: {1}".format(sort_acc[ind], valid_acc))
            if valid_acc > choose_acc:
                being_better = True
                choose_ind = ind
                choose_dag = tmp_dag
                choose_acc = valid_acc
        print("Choose {0}, acc: {1}".format(sort_acc[choose_ind], choose_acc))
        print("Dag: {}".format(choose_dag))
        print("*"*100)
        check_list[choose_ind] = True
        final_dag = choose_dag
        final_acc = choose_acc
    print("Final set {}".format(list(enumerate(check_list))))
    print("Final acc{}".format(final_acc))
    print("Final dag{}".format(final_dag))

# test_greedy_merge()


def _crossover(path1, path2, point):
    assert(point != 0 and point < num_cells)
    return np.concatenate((path1[:point], path2[point:])), \
            np.concatenate((path2[:point], path1[point:]))

def test_crossover():
    path_pool = np.array([[1, 2, 0, 3, 0], [2, 2, 0, 2, 0], [3, 2, 0, 2, 0], [4, 2, 0, 2, 0]])
    path_pool_acc = np.array([0.1, 0.2, 0.3, 0.4])
    for ind1 in range(len(path_pool)-1):
        for ind2 in range(ind1+1, len(path_pool)):
            for point in range(1, num_cells):
                cpath1, cpath2 = _crossover(path_pool[ind1], path_pool[ind2], point)
                print(cpath1, cpath2)

# test_crossover()
