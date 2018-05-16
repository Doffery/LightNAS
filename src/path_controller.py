from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import random

import numpy as np
import tensorflow as tf
import utils


flags = tf.app.flags
FLAGS = flags.FLAGS
logger = utils.logger


class PathController:
    def __init__(self,
                 num_cells,
                 cd_length,
                 opt_num,
                 path_pool_size,
                 k_init_selection_num,
                 k_best_selection_num,
                 max_generation):
        logger.info("*" * 80)
        logger.info('Start building controller')
        self.num_cells = num_cells
        self.num_cells_double = num_cells * 2
        self.cd_length = cd_length
        self.cd_opt_ind = self.cd_length-2
        self.cd_end_ind = self.cd_length-1
        self.opt_num = opt_num
        self.path_pool_size = path_pool_size
        self.max_generation = max_generation
        self.k_init_selection = k_init_selection_num
        self.k_best_selection = k_best_selection_num
        self.sample_path_arc = self._build_sampler()
        logger.info('Finished init PathController')

    def _build_sampler(self):
        logger.info('Sample a path arc')
        return tf.random_normal([1, 3], mean=-1, stddev=4)

    def _check_path(self, path):
        # return path.tolist().count(0) < self.num_cells-1
        return path[:self.num_cells].tolist().count(0) < self.num_cells-1 and \
                path[self.num_cells:].tolist().count(0) < self.num_cells-1

    def _path2dag(self, path):
        opt_ind = self.cd_opt_ind
        end_ind = self.cd_end_ind
        dag = np.zeros((self.num_cells*self.cd_length), dtype=np.int32)
        pre_op_idx = -1
        for i, op in enumerate(path):
            start_idx = i*self.cd_length
            if op == 0:
                dag[start_idx] = 2
            else:
                dag[start_idx+opt_ind] = op
                dag[start_idx+opt_ind-self.num_cells+pre_op_idx+1] = 1
                if pre_op_idx != -1:
                    dag[(pre_op_idx+1)*self.cd_length-1] = 0
                    dag[start_idx+end_ind] = 1
                pre_op_idx = i
        # logger.info(path)
        # logger.info(np.reshape(dag, (self.num_cells, self.cd_length)))
        return dag


    def _init_pool_as_paths(self):
        # path_pool = np.where(True, path_pool, -1)
        def _sample_path(length, num):
            if self.num_cells*length*2 <= num:
                num = self.num_cells*length*2
                return _sample_path(length, num)
            sample_dict = {}
            while num:
                sample_ind = random.sample(range(self.num_cells), length)
                path = np.zeros(self.num_cells, dtype=np.int32)
                for ind in sample_ind:
                    path[ind] = random.randint(1, self.opt_num)
                path_str = np.array2string(path)
                if path_str not in sample_dict and self._check_path(path):
                    sample_dict[path_str] = path
                    num -= 1
            return [t[1] for t in sample_dict.items()]

        # dag_pool = []
        path_pool = []
        remain = self.path_pool_size * 2  # for both dag and reduce dag
        for i in range(1, self.num_cells):
            length = i+1
            remain_each_length = remain // (self.num_cells-i)
            paths = _sample_path(length, remain_each_length)
            logger.info('Sampled {0} paths, length: {1}'.format(len(paths), length))
            for path in paths:
                logger.info(path)
                path_pool.append(path)
                # dag_pool.append(self._path2dag(path))
                remain -= 1
        np.random.shuffle(path_pool)  # short path, long path mixed together
        assert(len(path_pool) == self.path_pool_size*2)
        path_pool_concat = np.reshape(path_pool, 
                                      (self.path_pool_size, self.num_cells_double))
        # path_pool = [[1, 0, 3, 0, 2], [0, 0, 1, 0, 2]]
        # dag_pool = [  [1, 0, 0, 0, 0, 1, 0,
        #                 2, 0, 0, 0, 0, 0, 0,
        #                 0, 1, 0, 0, 0, 3, 0,
        #                 2, 0, 0, 0, 0, 0, 0,
        #                 1, 0, 0, 1, 0, 2, 1, ],
        #                [1, 0, 0, 0, 0, 0, 0,
        #                 2, 0, 0, 0, 0, 0, 0,
        #                 0, 1, 0, 0, 0, 1, 1,
        #                 2, 0, 0, 0, 0, 0, 0,
        #                 0, 0, 0, 1, 0, 2, 1,
        #                 ]]
        # we better return both dag and path
        return path_pool_concat  # , dag_pool

    def _init_pool(self, path_pool):
        path_pool = np.where(True, path_pool, -1)
        path_pool[0][0] = 1
        path_pool[0][2] = 3
        path_pool[0][4] = 0
        # path_pool[0][5] = 2
        path_pool[1][0] = 0
        path_pool[1][2] = 3
        path_pool[1][4] = 2
       #  path_pool[1][5] = 1
        # def _random_init_path(path, select_length):
        #     return path
        # for i in range(self.num_cells):


    def build_path_pool_full_arc(self, child_ops):
        # child_model.build_valid_rl()
        # child_ops = {
        #         "global_step": child_model.global_step,
        #         "loss": child_model.loss,
        #         "train_op": child_model.train_op,
        #         "lr": child_model.lr,
        #         "grad_norm": child_model.grad_norm,
        #         "train_acc": child_model.train_acc,
        #         "optimizer": child_model.optimizer,
        #         "num_train_batches": child_model.num_train_batches,
        #         }
        # random init a pool of paths
        # self.num_cells possible connections, +1 chosen ops
        # path_pool, dag_pool = self._init_pool_as_paths()
        path_pool = self._init_pool_as_paths()
        path_pool_acc = np.zeros(shape=(self.path_pool_size), dtype=float)
       
        saver = tf.train.Saver(max_to_keep=2)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
          FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)
       
        hooks = [checkpoint_saver_hook]
        if FLAGS.child_sync_replicas:
            sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)
       
        logger.info("-" * 80)
        logger.info("Starting session")
        config = tf.ConfigProto(allow_soft_placement=True)
        #                         log_device_placement=True)
        merged = tf.summary.merge_all()


        # print("Variables:")
        # for var in tf.trainable_variables():
        #     print(var)
        with tf.train.SingularMonitoredSession(
          config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        # with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            start_time = time.time()
            # train each path, get validation acc
            # get the initial population
            self.iteration = 0
            self.epoch = 0

            def _train_it(extra_epoch_num):
                t_epoch_num = self.epoch + extra_epoch_num
                while True:
                    for i in range(self.path_pool_size):
                        feed_dict = {child_ops["dag_arc"]: self._path2dag(path_pool[i][:self.num_cells]),
                                child_ops["reduce_arc"]: self._path2dag(path_pool[i][self.num_cells:])}
                        if self.iteration % 100 == 1:
                            run_ops = [
                                    merged,
                                    child_ops["loss"],
                                    child_ops["lr"],
                                    child_ops["grad_norm"],
                                    child_ops["train_acc"],
                                    child_ops["train_op"],
                                    ]
                            summary, loss, lr, gn, tr_acc, _ = sess.run(run_ops,
                                    feed_dict=feed_dict, options=run_options,
                                    run_metadata=run_metadata)
                            train_writer.add_run_metadata(run_metadata,
                                                  'step{0}'.format(self.iteration))
                            train_writer.add_summary(summary, self.iteration)
                            logger.info("Finish t_step {0}".format(self.iteration))
                        else:
                            run_ops = [
                                    child_ops["loss"],
                                    child_ops["lr"],
                                    child_ops["grad_norm"],
                                    child_ops["train_acc"],
                                    child_ops["train_op"],
                                    ]
                            loss, lr, gn, tr_acc, _ = sess.run(run_ops,
                                    feed_dict=feed_dict)

                        global_step = sess.run(child_ops["global_step"])
                        self.iteration = global_step

                        if FLAGS.child_sync_replicas:
                            actual_step = global_step * FLAGS.num_aggregate
                        else:
                            actual_step = global_step
                        self.epoch = actual_step // child_ops["num_train_batches"]
                        curr_time = time.time()
                        if global_step % FLAGS.log_every == 0:
                            logger.info("Global Step at {}".format(global_step))
                            log_string = ""
                            log_string += "epoch={:<6d}".format(self.epoch)
                            log_string += "ch_step={:<6d}".format(global_step)
                            log_string += " loss={:<8.6f}".format(loss)
                            log_string += " lr={:<8.4f}".format(lr)
                            log_string += " |g|={:<8.4f}".format(gn)
                            log_string += " tr_acc={:<3d}/{:>3d}".format(
                                tr_acc, FLAGS.batch_size)
                            log_string += " mins={:<10.2f}".format(
                                float(curr_time - start_time) / 60)
                            logger.info(log_string)

                            if FLAGS.child_fixed_arc is None:
                                child_ops["eval_func"](sess, "valid", feed_dict=feed_dict)
                            child_ops["eval_func"](sess, "test", feed_dict=feed_dict)

                    if self.epoch >= t_epoch_num:
                        for i in range(self.path_pool_size):
                            feed_dict = {child_ops["dag_arc"]: self._path2dag(path_pool[i][:self.num_cells]),
                                    child_ops["reduce_arc"]: self._path2dag(path_pool[i][self.num_cells:])}
                            # feed_dict = {child_ops["dag_arc"]: dag_pool[i]}
                            valid_acc = sess.run(child_ops["valid_rl_acc"],
                                                 feed_dict=feed_dict)
                            path_pool_acc[i] = valid_acc

                            logger.info("path_acc {0}: {1}".format(i, valid_acc))
                        break

            logger.info("Start init the pool and cal valid_acc")
            _train_it(FLAGS.num_epochs)
            logger.info("Finish init the path pool")


            def _is_new_path(path):
                for p in path_pool:
                    if (p == path).all():
                        return False
                return True

            def _is_new_path_in_pool(path, pool):
                for p in pool:
                    if (p == path).all():
                        return False
                return True

            # start evolving
            logger.info("Start evolving...")
            evolve_iter = 0
            while evolve_iter < self.max_generation:
                evolve_iter += 1
                # select top-k
                logger.info("Evolving {0}:".format(evolve_iter))
                logger.info("Pathes: {0}".format(list(enumerate(path_pool))))
                logger.info("Pathes acc: {0}".format(list(enumerate(path_pool_acc))))
                top_k_ind_acc = utils.find_top_k_ind(path_pool_acc,
                                                     self.k_init_selection)
                logger.info("Select top K seeds: {0}".format(top_k_ind_acc))
                seeds = [x for x, _ in top_k_ind_acc]
                candidate_paths = []

                # apply mutation
                for ind in seeds:
                    for i in range(self.num_cells_double):
                        for j in range(self.opt_num+1):
                            tmp_path = np.copy(path_pool[ind])
                            # tmp = tmp_path[i]
                            tmp_path[i] = j  # np.random.randint(0, self.opt_num+1)
                            if self._check_path(tmp_path) and \
                                    _is_new_path(tmp_path) and \
                                    _is_new_path_in_pool(tmp_path, candidate_paths):
                                candidate_paths.append(tmp_path)
                            # tmp_path[i] = tmp

                def _crossover(path1, path2, point):
                    assert(point != 0 and point < self.num_cells_double)
                    return np.concatenate((path1[:point], path2[point:])), \
                            np.concatenate((path2[:point], path1[point:]))

                # apply crossover
                for ind1 in range(len(path_pool)-1):
                    for ind2 in range(ind1+1, len(path_pool)):
                        for point in range(1, self.num_cells_double):
                            cpath1, cpath2 = _crossover(path_pool[ind1],
                                                        path_pool[ind2], point)
                            for cpath in [cpath1, cpath2]:
                                if self._check_path(cpath) and \
                                        _is_new_path(cpath) and \
                                        _is_new_path_in_pool(cpath,
                                                             candidate_paths):
                                    candidate_paths.append(cpath)

                # predict and select best k
                logger.info("Candidate Paths: {0}".format(candidate_paths))
                candidate_accs = []
                for i, cpath in enumerate(candidate_paths):
                    # cdag = self._path2dag(cpath)
                    feed_dict = {child_ops["dag_arc"]: self._path2dag(cpath[:self.num_cells]),
                            child_ops["reduce_arc"]: self._path2dag(cpath[self.num_cells:])}
                    # feed_dict = {child_ops["dag_arc"]: cdag}
                    valid_acc = sess.run(child_ops["valid_rl_acc"],
                                         feed_dict=feed_dict)
                    logger.info('Candidate {0} acc: {1}'.format(i, valid_acc))
                    candidate_accs.append(valid_acc)

                top_k_candidates = utils.find_top_k_ind(candidate_accs,
                                                        self.k_best_selection)
                logger.info("Top K Candidates: {0}".format(top_k_candidates))

                # replace the worse with candidates
                for tk_ind, tk_acc in top_k_candidates:
                    path_pool = np.append(path_pool, [candidate_paths[tk_ind]], axis=0)
                    path_pool_acc = np.append(path_pool_acc, [tk_acc])
                bad_k_paths = utils.find_rtop_k_ind(path_pool_acc,
                                                    len(top_k_candidates))
                                                    # self.k_best_selection)
                logger.info("Removing: {0}".format(bad_k_paths))
                del_inx = [ind for ind, _ in bad_k_paths]
                # del_inx.sort(reverse=True)
                # for d_ind in del_inx:
                #     del path_pool[d_ind]
                #     del path_pool_acc[d_ind]
                tmp_path_pool = []
                # tmp_dag_pool = []
                tmp_path_pool_acc = []
                for ind in range(len(path_pool)):
                    if ind not in del_inx:
                        tmp_path_pool.append(path_pool[ind])
                        # tmp_dag_pool.append(self._path2dag(path_pool[ind]))
                        tmp_path_pool_acc.append(path_pool_acc[ind])
                path_pool = np.array(tmp_path_pool)
                # dag_pool = np.array(tmp_dag_pool)
                path_pool_acc = np.array(tmp_path_pool_acc)

                if evolve_iter % FLAGS.train_every_generations == 0:
                    logger.info("Train evolving iteration {}".format(evolve_iter))
                    _train_it(FLAGS.num_epochs_evolve)
                logger.info("Finish evolving iteration {}".format(evolve_iter))

            logger.info("Final Pathes: {0}".format(list(enumerate(path_pool))))
            logger.info("Final Pathes acc: {0}".format(list(enumerate(path_pool_acc))))

            # build dags from paths
            # insert some short paths first

            # path_pool = [  [1, 0, 0, 0, 0, 1, 0,
            #                 2, 0, 0, 0, 0, 0, 0,
            #                 0, 1, 0, 0, 0, 3, 0,
            #                 2, 0, 0, 0, 0, 0, 0,
            #                 0, 0, 0, 1, 0, 2, 1, ],
            #                [1, 0, 0, 0, 0, 0, 0,
            #                 2, 0, 0, 0, 0, 0, 0,
            #                 0, 1, 0, 0, 0, 1, 0,
            #                 2, 0, 0, 0, 0, 0, 0,
            #                 0, 0, 0, 1, 0, 2, 1,
            #                 ]]
            def _merge_dag(da, db):
                opt_ind = self.cd_opt_ind
                end_ind = self.cd_end_ind
                d2a = np.reshape(da, (self.num_cells, self.cd_length))
                d2b = np.reshape(db, (self.num_cells, self.cd_length))
                d2c = np.zeros((self.num_cells, self.cd_length),
                               dtype=np.int32)
                for ind in range(self.num_cells):
                    if d2a[ind][opt_ind] == 0 and \
                            d2b[ind][opt_ind] == 0:
                        d2c[ind][opt_ind] = 0
                        d2c[ind][0] = 2
                        continue
                    if d2a[ind][opt_ind] == 0:
                        d2a[ind][0] = 0
                    if d2b[ind][opt_ind] == 0:
                        d2b[ind][0] = 0
                    for jnd in range(opt_ind):
                        if d2a[ind][jnd] != 0 or d2b[ind][jnd] != 0:
                            d2c[ind][jnd] = 1
                    # How do we decide the operator?
                    # Choose the first, the better?
                    if d2a[ind][opt_ind] != 0:
                        d2c[ind][opt_ind] = d2a[ind][opt_ind]
                    else:
                        d2c[ind][opt_ind] = d2b[ind][opt_ind]

                    # is End or not
                    if d2a[ind][end_ind] == 1 or \
                            d2b[ind][end_ind] == 1:
                        d2c[ind][end_ind] = 1
                return d2c.flatten()

            being_better = True
            sort_acc = sorted(enumerate(path_pool_acc),
                              key=lambda p: p[1], reverse=True)
            check_list = np.zeros((len(path_pool)), dtype=np.bool)
            check_list[0] = True
            final_dag = self._path2dag(path_pool[sort_acc[0][0]][:self.num_cells])
            final_dag_reduce = self._path2dag(path_pool[sort_acc[0][0]][self.num_cells:])
            logger.info("Merge dag{}".format(np.reshape(final_dag,
                (self.num_cells, self.cd_length))))
            logger.info("Merge reduce dag{}".format(np.reshape(final_dag_reduce,
                (self.num_cells, self.cd_length))))
            final_acc = path_pool_acc[sort_acc[0][0]]
            while being_better:
                choose_dag = final_dag
                choose_dag_reduce = final_dag_reduce
                choose_acc = final_acc
                choose_ind = 0
                being_better = False
                for ind in range(1, len(sort_acc)):
                    if check_list[ind]:
                        continue
                    tmp_dag = _merge_dag(final_dag,
                            self._path2dag(path_pool[sort_acc[ind][0]][:self.num_cells]))
                    tmp_dag_reduce = _merge_dag(final_dag_reduce,
                            self._path2dag(path_pool[sort_acc[ind][0]][self.num_cells:]))
                    # feed_dict = {child_ops["dag_arc"]: tmp_dag}
                    feed_dict = {child_ops["dag_arc"]: tmp_dag,
                            child_ops["reduce_arc"]: tmp_dag_reduce}
                    valid_acc = sess.run(child_ops["valid_rl_acc"],
                                         feed_dict=feed_dict)
                    logger.info("Merge {0} acc {1}".format(sort_acc[ind][0], valid_acc))
                    if valid_acc > choose_acc:
                        being_better = True
                        choose_ind = ind
                        choose_dag = tmp_dag
                        choose_dag_reduce = tmp_dag_reduce
                        choose_acc = valid_acc
                check_list[choose_ind] = True
                final_dag = choose_dag
                final_dag_reduce = choose_dag_reduce
                final_acc = choose_acc
                logger.info("Merge dag{}".format(np.reshape(final_dag,
                    (self.num_cells, self.cd_length))))
                logger.info("Merge reduce dag{}".format(np.reshape(final_dag_reduce,
                    (self.num_cells, self.cd_length))))
            logger.info("Final set {}".format(list(enumerate(check_list))))
            logger.info("Final valid acc{}".format(final_acc))
            logger.info("Final dag{}".format(np.reshape(final_dag,
                (self.num_cells, self.cd_length))))
            logger.info("Final reduce dag{}".format(np.reshape(final_dag_reduce,
                (self.num_cells, self.cd_length))))
            child_ops["eval_func"](sess, "test",
                                   feed_dict={child_ops["dag_arc"]: final_dag,
                                              child_ops["reduce_arc"]: final_dag_reduce})

            train_writer.close()


