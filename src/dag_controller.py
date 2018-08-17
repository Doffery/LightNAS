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
import math
from scipy.special import comb


flags = tf.app.flags
FLAGS = flags.FLAGS
logger = utils.logger


class DagController:
    def __init__(self,
                 num_cells,
                 num_layers,
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
        self.num_layers = num_layers
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
        return_val = True
        if path.tolist().count(1) == 0 and \
                path.tolist().count(2) == 0:
            return_val = False
        return return_val

    def _check_path_double(self, path):
        return self._check_path(path[:self.num_cells]) and \
                self._check_path(path[self.num_cells:])

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

    def _init_ops_pool(self):
        ops_pool = []
        # ops_pool_acc = []
        sample_dict = {}
        num = 0
        while num < self.path_pool_size:
            ops_dag = np.zeros(self.num_cells_double, dtype=np.int32)
            for i in range(self.num_cells_double):
                ops_dag[i] = random.randint(1, self.opt_num)
            ops_str = np.array2string(ops_dag)
            if ops_str not in sample_dict and self._check_path_double(ops_dag):
                sample_dict[ops_str] = 1
                ops_pool.append(ops_dag)
                # ops_pool_acc.append(0.0)
                num += 1
        logger.info("Sampled ops dag: {}".format(ops_pool))
        return ops_pool  # , ops_pool_acc

    def _merge_dag(self, da, db):
        opt_ind = self.cd_opt_ind
        end_ind = self.cd_end_ind
        d2a = np.copy(np.reshape(da, (self.num_cells, self.cd_length)))
        d2b = np.copy(np.reshape(db, (self.num_cells, self.cd_length)))
        # logger.info(d2b[0][-1])
        # logger.info(d2b[0][-1] == 1)
        # logger.info(d2b[0][-2])
        # logger.info(d2b[0][-2] == 1)
        if d2a[0][-1] == 1 and d2a[0][-2] == 0:
            d2a[0][-2] = -1
        if d2b[0][-1] == 1 and d2b[0][-2] == 0:
            d2b[0][-2] = -1
        d2c = np.zeros((self.num_cells, self.cd_length),
                       dtype=np.int32)
        for ind in range(self.num_cells):
            if d2a[ind][opt_ind] == 0 and d2b[ind][opt_ind] == 0:
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
        if d2c[0][-2] == -1:
            d2c[0][-2] = 0
        return d2c.flatten()

    def evolve_ops_dag(self, child_ops, generator_ops):
        logger.info("-" * 80)
        logger.info("Starting session")
        config = tf.ConfigProto(allow_soft_placement=True)
        saver = tf.train.Saver(max_to_keep=2)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)

        hooks = [checkpoint_saver_hook]
        if FLAGS.child_sync_replicas:
            sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
        #     hooks.append(sync_replicas_hook)
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = generator_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)

        merged = tf.summary.merge_all()
        time_tag = str(int(time.time()))
        logger.info("Summary Tag {}".format(time_tag))
        sess = tf.train.SingularMonitoredSession(config=config,
                                hooks=hooks, checkpoint_dir=FLAGS.output_dir)
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir+time_tag,
                                             sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # start evolving
        logger.info("Start evolving...")
        evolve_iter = 0
        self.iteration = 0
        start_time = time.time()

        def _train_it_path(extra_epoch_num, train_path_pool):
            _train_it(extra_epoch_num,
                      [(self._path2dag(path[:self.num_cells]),
                          self._path2dag(path[self.num_cells:])) \
                                for path in train_path_pool])

        def _train_it(extra_epoch_num, train_pool):
            t_step_num = self.iteration + \
                int(extra_epoch_num*child_ops["num_train_batches"])
            # t_epoch_num = self.epoch + extra_epoch_num
            while True:
                for i in range(len(train_pool)):
                    feed_dict = {child_ops["dag_arc"]: train_pool[i][0],
                                 child_ops["reduce_arc"]: train_pool[i][1]}
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

                if self.iteration >= t_step_num:
                    break

        def _eval_ops_dag(sess, ops_dag, child_ops, generator_ops):
            logger.info("Start evaluating {}".format(ops_dag))
            feed_dict = {generator_ops["conv_ops"]: ops_dag[:self.num_cells],
                         generator_ops["reduce_ops"]: ops_dag[self.num_cells:]}
            run_ops = [
                merged,
                generator_ops["sample_arc"],
                generator_ops["entropy"],
                generator_ops["lr"],
                generator_ops["grad_norm"],
                generator_ops["valid_acc"],
                generator_ops["baseline"],
                generator_ops["skip_rate"],
                generator_ops["train_op"],
            ]
            pool = []
            pool_acc = []
            self.num_cand_path = 8
            for i in range(self.num_cand_path):
                # generator_step = sess.run(generator_ops["train_op"], 
                #         feed_dict=feed_dict)
                # logger.info(generator_step)
                summary, arc, entropy, lr, gn, val_acc, bl, skip, _ = sess.run(run_ops, 
                                        feed_dict=feed_dict, options=run_options,
                                        run_metadata=run_metadata)
                # train_writer.add_run_metadata(run_metadata,
                #                       'step{0}{1}'.format(self.iteration, i))
                # train_writer.add_summary(summary, self.iteration)
                logger.info("Sampled Arc: ")
                logger.info(np.reshape(arc, (self.num_cells, self.cd_length)))
                logger.info(val_acc)
                logger.info(entropy)
                logger.info(lr)
                logger.info(gn)
                logger.info(bl)
                logger.info(skip)
                pool.append(arc)
                pool_acc.append(val_acc)

            def _merge_extensive(path_pool, path_pool_acc):
                # try every possibility of combining 2,3 pathes
                final_dag = self._path2dag([0 for x in range(self.num_cells)])
                final_dag_reduce = self._path2dag([0 for x in range(self.num_cells)])
                best_acc = 0
                for it in range(1, 1 << self.num_cand_path):
                    path_num = bin(it).count("1") 
                    if path_num < 2 or path_num > 3:
                        continue
                    tmp_dag = self._path2dag([0 for x in range(self.num_cells)])
                    tmp_dag_reduce = self._path2dag([0 for x in range(self.num_cells)])
                    for i in range(self.num_cand_path):
                        if (1 << i) & it != 0:
                            tmp_dag = self._merge_dag(tmp_dag, path_pool[i][0])
                            tmp_dag_reduce = self._merge_dag(tmp_dag_reduce, path_pool[i][1])
                    feed_dict = {child_ops["dag_arc"]: tmp_dag,
                            child_ops["reduce_arc"]: tmp_dag_reduce}
                    valid_acc = sess.run(child_ops["valid_rl_acc"],
                                         feed_dict=feed_dict)
                    logger.info(bin(it))
                    logger.info(tmp_dag)
                    logger.info(tmp_dag_reduce)
                    logger.info(valid_acc)
                    if valid_acc > best_acc:
                        final_dag = tmp_dag
                        final_dag_reduce = tmp_dag_reduce
                        best_acc = valid_acc
                return best_acc, final_dag


            def _merge_strategy_greedy(path_pool, path_pool_acc):
                being_better = True
                sort_acc = sorted(enumerate(path_pool_acc),
                                  key=lambda p: p[1], reverse=True)
                check_list = np.zeros((len(path_pool)), dtype=np.bool)
                check_list[0] = True
                final_dag = path_pool[sort_acc[0][0]][0]
                final_dag_reduce = path_pool[sort_acc[0][0]][1]
                logger.info("Merge dag{}".format(np.reshape(final_dag,
                    (self.num_cells, self.cd_length))))
                logger.info("Merge reduce dag{}".format(np.reshape(final_dag_reduce,
                    (self.num_cells, self.cd_length))))
                # Choose the best is not always a good idea?
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
                        tmp_dag = self._merge_dag(final_dag,
                                path_pool[sort_acc[ind][0]][0])
                        tmp_dag_reduce = self._merge_dag(final_dag_reduce,
                                path_pool[sort_acc[ind][0]][1])
                        # logger.info(path_pool[sort_acc[ind][0]][0])
                        # logger.info(tmp_dag)
                        # logger.info(path_pool[sort_acc[ind][0]][1])
                        # logger.info(tmp_dag_reduce)
                        # _train_it(0.5, [(tmp_dag, tmp_dag_reduce)])
                        # feed_dict = {child_ops["dag_arc"]: tmp_dag}
                        feed_dict = {child_ops["dag_arc"]: tmp_dag,
                                child_ops["reduce_arc"]: tmp_dag_reduce}
                        valid_acc = sess.run(child_ops["valid_rl_acc"],
                                             feed_dict=feed_dict)
                        # logger.info("Merge {0} acc {1}".format(sort_acc[ind][0], valid_acc))
                        # logger.info(path_pool[sort_acc[ind][0]])
                        # logger.info(tmp_dag)
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
                logger.info("Final valid acc{}".format(final_acc))
                logger.info("Final dag{}".format(np.reshape(final_dag,
                    (self.num_cells, self.cd_length))))
                logger.info("Final reduce dag{}".format(np.reshape(final_dag_reduce,
                (self.num_cells, self.cd_length))))
                best_acc = final_acc
                best_dag = (final_dag, final_dag_reduce)
                return best_acc, best_dag

            return _merge_extensive(pool, pool_acc)

        ops_pool = self._init_ops_pool()
        ops_pool_acc = []
        init_dags = []
        for op in ops_pool:
            best_acc, best_dag = _eval_ops_dag(sess, op, child_ops, generator_ops)
            # ops_pool_acc.append(best_acc)
            init_dags.append(best_dag)
        _train_it(0.5, init_dags)
        for op in ops_pool:
            best_acc, best_dag = _eval_ops_dag(sess, op, child_ops, generator_ops)
            ops_pool_acc.append(best_acc)
            # init_dags.append(best_dag)

        while evolve_iter < self.max_generation:
            evolve_iter += 1
            # select top-k
            logger.info("Evolving {0}:".format(evolve_iter))
            logger.info("Ops: {0}".format(list(enumerate(ops_pool))))
            logger.info("Ops acc: {0}".format(list(enumerate(ops_pool_acc))))
            seeds = []
            top_k_ind_acc = utils.find_top_k_ind(ops_pool_acc,
                                                 self.k_init_selection)
            logger.info("Select top K seeds {0}".format(top_k_ind_acc))
            seeds.extend([x for x, _ in top_k_ind_acc])
            candidate_ops = []

            def _is_new_ops_dag(path):
                for p in ops_pool:
                    if (p == path).all():
                        return False
                return True

            def _is_new_ops_dag_in_pool(path, pool):
                for p in pool:
                    if (p == path).all():
                        return False
                return True

            # apply mutation
            for ind in seeds:
                for i in range(self.num_cells_double):
                    for j in range(1, self.opt_num):
                        tmp_path = np.copy(ops_pool[ind])
                        # tmp = tmp_path[i]
                        tmp_path[i] = j  # np.random.randint(0, self.opt_num+1)
                        if self._check_path_double(tmp_path) and \
                                _is_new_ops_dag(tmp_path) and \
                                _is_new_ops_dag_in_pool(tmp_path, candidate_ops):
                            candidate_ops.append(tmp_path)
                        # tmp_path[i] = tmp

            def _crossover(path1, path2, point):
                assert(point != 0 and point < self.num_cells_double)
                return np.concatenate((path1[:point], path2[point:])), \
                        np.concatenate((path2[:point], path1[point:]))

            # apply crossover
            top_k_path_acc = utils.find_top_k_ind(ops_pool_acc,
                                                  self.k_init_selection)
            for ind1 in range(len(top_k_path_acc)-1):
                for ind2 in range(ind1+1, len(top_k_path_acc)):
                    pi1 = top_k_path_acc[ind1][0]
                    pi2 = top_k_path_acc[ind2][0]
                    for point in range(1, self.num_cells_double):
                        cpath1, cpath2 = _crossover(ops_pool[pi1],
                                                    ops_pool[pi2], point)
                        for cpath in [cpath1, cpath2]:
                            if self._check_path_double(cpath) and \
                                    _is_new_ops_dag(cpath) and \
                                    _is_new_ops_dag_in_pool(cpath,
                                                         candidate_ops):
                                candidate_ops.append(cpath)

            # _train_it_path(1, candidate_ops)
            train_cand_set = []
            # predict and select best k
            logger.info("Candidate Paths: {0}".format(candidate_ops))
            candidate_accs = []
            candidate_ops_dag = []
            for i, cpath in enumerate(candidate_ops):
                # cdag = self._path2dag(cpath)
                # feed_dict = {child_ops["dag_arc"]: cdag}
                valid_acc, best_dag = _eval_ops_dag(sess, cpath, 
                                                  child_ops, generator_ops)
                logger.info('Candidate {0} acc: {1}'.format(i, valid_acc))
                # child_ops["eval_func"](sess, "test", feed_dict=feed_dict)
                # bucket_ind = _path_length(cpath)
                # candidate_path_buckets[bucket_ind].append(cpath)
                # candidate_bucket_accs[bucket_ind].append(valid_acc)
                candidate_accs.append(valid_acc)
                candidate_ops_dag.append(best_dag)

            top_k_candidates = utils.find_top_k_ind(candidate_accs,
                                                    self.k_best_selection)
            logger.info("B Top K Candidates: {}".format(
                top_k_candidates))

            # replace the worse with candidates
            for tk_ind, tk_acc in top_k_candidates:
                train_cand_set.append(candidate_ops_dag[tk_ind])
                ops_pool.append(candidate_ops[tk_ind])
                ops_pool_acc.append(candidate_accs[tk_ind])
            bad_k_paths = utils.find_rtop_k_ind(ops_pool_acc,
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
            for ind in range(len(ops_pool)):
                if ind not in del_inx:
                    tmp_path_pool.append(ops_pool[ind])
                    # tmp_dag_pool.append(self._path2dag(path_pool[ind]))
                    tmp_path_pool_acc.append(ops_pool_acc[ind])
            ops_pool = tmp_path_pool
            ops_pool_acc = tmp_path_pool_acc
            # ops_pool = np.array(tmp_path_pool)
            # ops_pool_acc = np.array(tmp_path_pool_acc)

            # It means train after selection here
            # _train_it(0.5, train_cand_set)
            ops_pool_acc = []
            init_dags = []
            for op in ops_pool:
                best_acc, best_dag = _eval_ops_dag(sess, op, child_ops, generator_ops)
                # ops_pool_acc.append(best_acc)
                init_dags.append(best_dag)
            _train_it(0.5, init_dags)
            for op in ops_pool:
                best_acc, best_dag = _eval_ops_dag(sess, op, child_ops, generator_ops)
                ops_pool_acc.append(best_acc)
                # init_dags.append(best_dag)

            # if evolve_iter % FLAGS.train_every_generations == 0:
            #     logger.info("Train evolving iteration {}".format(evolve_iter))
            #     _train_it(FLAGS.num_epochs_evolve)
            # logger.info("Finish evolving iteration {}".format(evolve_iter))

        logger.info("Final OpsDag: {0}".format(list(enumerate(ops_pool))))
        logger.info("Final OpsDag acc: {0}".format(list(enumerate(ops_pool_acc))))
        return 0

    def _find_best_dag():
        # find the best dag base on the ops dag
        return 0

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
