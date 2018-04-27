from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time

import numpy as np
import tensorflow as tf
import utils


flags = tf.app.flags
FLAGS = flags.FLAGS
logger = utils.get_logger()


class PathController:
    def __init__(self,
                 num_cells,
                 opt_num,
                 path_pool_size,
                 k_init_selection_num,
                 k_best_selection_num,
                 max_generation):
        logger.info("*" * 80)
        logger.info('Start building controller')
        self.num_cells = num_cells
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

    def _init_pool_as_paths(self, path_pool):
        path_pool = np.where(True, path_pool, -1)
        path_pool[0][0] = 1

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
        path_pool = np.zeros(shape=(self.path_pool_size,
                                    self.num_cells*self.num_cells), dtype=np.int32)
        self._init_pool_as_paths(path_pool)
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
        logger.info("Start init the pool and cal valid_acc")
        with tf.train.SingularMonitoredSession(
          config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        # with tf.Session() as sess:
            start_time = time.time()
            # train each path, get validation acc
            # get the initial population
            while True:
                for i in range(self.path_pool_size):
                    feed_dict = {child_ops["path_arc"]: path_pool[i]}
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

                    if FLAGS.child_sync_replicas:
                        actual_step = global_step * FLAGS.num_aggregate
                    else:
                        actual_step = global_step
                    epoch = actual_step // child_ops["num_train_batches"]
                    curr_time = time.time()
                    if global_step % FLAGS.log_every == 0:
                        log_string = ""
                        log_string += "epoch={:<6d}".format(epoch)
                        log_string += "ch_step={:<6d}".format(global_step)
                        log_string += " loss={:<8.6f}".format(loss)
                        log_string += " lr={:<8.4f}".format(lr)
                        log_string += " |g|={:<8.4f}".format(gn)
                        log_string += " tr_acc={:<3d}/{:>3d}".format(
                            tr_acc, FLAGS.batch_size)
                        log_string += " mins={:<10.2f}".format(
                            float(curr_time - start_time) / 60)
                        logger.info(log_string)

                    valid_acc = sess.run(child_ops["valid_rl_acc"], 
                                         feed_dict=feed_dict)
                    path_pool_acc[i] = valid_acc
                    logger.info("path_acc {0}: {1}".format(i, valid_acc))

                    logger.info("Epoch {}: Eval".format(epoch))
                    if FLAGS.child_fixed_arc is None:
                        child_ops["eval_func"](sess, "valid", feed_dict=feed_dict)
                    child_ops["eval_func"](sess, "test", feed_dict=feed_dict)

                if epoch >= FLAGS.num_epochs:
                    break
            logger.info("Finish init the path pool")

            # start evolving
            # select top-k
            top_k_ind_acc = utils.find_top_k_ind(path_pool_acc,
                                                 self.k_init_selection)
            candidates = []

            # apply mutation
            for ind, acc in top_k_ind_acc:
                tmp_path = path_pool[ind]
                for i in range(self.num_cells):
                    tmp = tmp_path[i]
                    tmp_path[i] = np.random.randint(-1, self.opt_num)
                    candidates.append(tmp_path)
                    tmp_path[i] = tmp

            # apply crossover
            # for ind1, acc1 in top_k_ind_acc:
            #     tmp_path1 = path_pool[ind1]
            #     for ind2, acc2 in top_k_ind_acc:
            #         tmp_path2 = path_pool[ind2]

            # predict and select best k

    def build_path_pool_out_tensor(self, child_ops):
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
        path_pool = np.zeros(shape=(self.path_pool_size,
                                    self.num_cells), dtype=np.int32)
        self._init_pool(path_pool)
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
        logger.info("Start init the pool and cal valid_acc")
        with tf.train.SingularMonitoredSession(
          config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        # with tf.Session() as sess:
            start_time = time.time()
            # train each path, get validation acc
            # get the initial population
            while True:
                for i in range(self.path_pool_size):
                    feed_dict = {child_ops["path_arc"]: path_pool[i]}
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

                    if FLAGS.child_sync_replicas:
                        actual_step = global_step * FLAGS.num_aggregate
                    else:
                        actual_step = global_step
                    epoch = actual_step // child_ops["num_train_batches"]
                    curr_time = time.time()
                    if global_step % FLAGS.log_every == 0:
                        log_string = ""
                        log_string += "epoch={:<6d}".format(epoch)
                        log_string += "ch_step={:<6d}".format(global_step)
                        log_string += " loss={:<8.6f}".format(loss)
                        log_string += " lr={:<8.4f}".format(lr)
                        log_string += " |g|={:<8.4f}".format(gn)
                        log_string += " tr_acc={:<3d}/{:>3d}".format(
                            tr_acc, FLAGS.batch_size)
                        log_string += " mins={:<10.2f}".format(
                            float(curr_time - start_time) / 60)
                        logger.info(log_string)

                    valid_acc = sess.run(child_ops["valid_rl_acc"], 
                                         feed_dict=feed_dict)
                    path_pool_acc[i] = valid_acc
                    logger.info("path_acc {0}: {1}".format(i, valid_acc))

                    logger.info("Epoch {}: Eval".format(epoch))
                    if FLAGS.child_fixed_arc is None:
                        child_ops["eval_func"](sess, "valid", feed_dict=feed_dict)
                    child_ops["eval_func"](sess, "test", feed_dict=feed_dict)

                if epoch >= FLAGS.num_epochs:
                    break
            logger.info("Finish init the path pool")

            # start evolving
            # select top-k
            top_k_ind_acc = utils.find_top_k_ind(path_pool_acc,
                                                 self.k_init_selection)
            candidates = []

            # apply mutation
            for ind, acc in top_k_ind_acc:
                tmp_path = path_pool[ind]
                for i in range(self.num_cells):
                    tmp = tmp_path[i]
                    tmp_path[i] = np.random.randint(-1, self.opt_num)
                    candidates.append(tmp_path)
                    tmp_path[i] = tmp

            # apply crossover
            # for ind1, acc1 in top_k_ind_acc:
            #     tmp_path1 = path_pool[ind1]
            #     for ind2, acc2 in top_k_ind_acc:
            #         tmp_path2 = path_pool[ind2]

            # predict and select best k


    def build_path_pool(self, generator):
        # arc_seq = tf.TensorArray(tf.int32, size=self.num_cells * 4)
        # generator.build_valid_rl()
        path_pool = None
        # pool_acc = None
        for i in range(5):
            tmp_arc = self._build_sampler()
            self.sample_path_arc = tmp_arc
            path_pool = tf.concat([path_pool, tmp_arc], 0)
            # call path_generator functions
            # tmp_acc = (tf.to_float(generator.valid_shuffle_acc) /
            #            tf.to_float(generator.batch_size))
            # pool_acc = tf.concat([pool_acc, tmp_acc], 0)
        self.init_path_pool = path_pool
        # self.init_pool_acc = pool_acc

        '''
        def _condition(generation_id, *args):
            return tf.less(generation_id, self.max_generation)

        def _body(generation_id):
            # do mutation and crossover, get the next generation
            return generation_id

        loop_vars = [
                tf.constant(1, dtype=tf.int32, name="generation_id")
                ]

        loop_outputs = tf.while_loop(_condition, _body, loop_vars,
                                     parallel_iterations=1)
        '''
