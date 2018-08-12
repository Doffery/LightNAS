import os
#import cPickle as pickle
import pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

import utils
# from utils import Logger
from utils import DEFINE_boolean
from utils import DEFINE_float
from utils import DEFINE_integer
from utils import DEFINE_string
from utils import print_user_flags
from path_controller import PathController
from path_generator import PathGenerator
from dag_executor import DagExecutor
from dag_generator import DagGenerator
from dag_controller import DagController
import data_utils
from data_utils import read_data

flags = tf.app.flags
FLAGS = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = utils.logger


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("summaries_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_string("search_for", None, "Must be [macro|micro]")

DEFINE_integer("num_gpus", 1, "")
DEFINE_integer("num_cpus", 1, "")
DEFINE_integer("batch_size", 32, "")

DEFINE_integer("num_epochs_evolve", 3, "")

DEFINE_integer("num_epochs", 300, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_integer("child_num_cells", 8, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 48, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 4, "")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", None, "for lr schedule")
DEFINE_integer("child_lr_T_mul", None, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr_max", None, "for lr schedule")
DEFINE_float("child_lr_min", None, "for lr schedule")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")


DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", False, "")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")
DEFINE_boolean("controller_use_critic", False, "")


DEFINE_integer("cd_length", 7, "cell_descriptor_length")
# DEFINE_integer("opt_num", 4, "num of ops can be selected")
DEFINE_integer("path_pool_size", 10, "")
DEFINE_integer("k_init_selection_num", 2, "")
DEFINE_integer("k_best_selection_num", 2, "")
DEFINE_integer("max_generation", 50, "")

DEFINE_integer("train_every_generations", 5, "train again after n generations")
DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every", 23, "How many steps to log")
# DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels):
    """
    Args:
      images: dict with keys {"train", "valid", "test"}.
      labels: dict with keys {"train", "valid", "test"}.
    """
 
    assert FLAGS.search_for is not None, "Please specify --search_for"
 
    # if FLAGS.search_for == "micro":
    # ControllerClass = PathController
    # ChildClass = PathGenerator
    ChildClass = DagExecutor
    # else:
    #  ControllerClass = GeneralController
    #  ChildClass = GeneralChild
 
    child_model = ChildClass(
      images,
      labels,
      num_gpus=FLAGS.num_gpus,
      num_cpus=FLAGS.num_cpus,
      use_aux_heads=FLAGS.child_use_aux_heads,
      cutout_size=FLAGS.child_cutout_size,
      whole_channels=FLAGS.controller_search_whole_channels,
      num_layers=FLAGS.child_num_layers,
      num_cells=FLAGS.child_num_cells,
      cd_length=FLAGS.child_num_cells+2,
      num_branches=FLAGS.child_num_branches,
      fixed_arc=FLAGS.child_fixed_arc,
      out_filters_scale=FLAGS.child_out_filters_scale,
      out_filters=FLAGS.child_out_filters,
      keep_prob=FLAGS.child_keep_prob,
      drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
      num_epochs=FLAGS.num_epochs,
      l2_reg=FLAGS.child_l2_reg,
      data_format=FLAGS.data_format,
      batch_size=FLAGS.batch_size,
      clip_mode="norm",
      grad_bound=FLAGS.child_grad_bound,
      lr_init=FLAGS.child_lr,
      lr_dec_every=FLAGS.child_lr_dec_every,
      lr_dec_rate=FLAGS.child_lr_dec_rate,
      lr_cosine=FLAGS.child_lr_cosine,
      lr_max=FLAGS.child_lr_max,
      lr_min=FLAGS.child_lr_min,
      lr_T_0=FLAGS.child_lr_T_0,
      lr_T_mul=FLAGS.child_lr_T_mul,
      optim_algo="momentum",
      sync_replicas=FLAGS.child_sync_replicas,
      num_aggregate=FLAGS.child_num_aggregate,
      num_replicas=FLAGS.child_num_replicas,
    )

    generator_model = DagGenerator(
        num_cells=FLAGS.child_num_cells,
        num_branches=FLAGS.child_num_branches,
        lstm_size=32,
        lstm_num_layers=1,
        lstm_keep_prob=1.0,
        tanh_constant=FLAGS.controller_tanh_constant,
        op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
        temperature=FLAGS.controller_temperature,
        lr_init=FLAGS.controller_lr,
        lr_dec_start=0,
        lr_dec_every=1000000,  # never decrease learning rate
        l2_reg=FLAGS.controller_l2_reg,
        entropy_weight=FLAGS.controller_entropy_weight,
        bl_dec=FLAGS.controller_bl_dec,
        use_critic=FLAGS.controller_use_critic,
        optim_algo="sgd",
        sync_replicas=FLAGS.controller_sync_replicas,
        num_aggregate=FLAGS.controller_num_aggregate,
        num_replicas=FLAGS.controller_num_replicas)

    child_model.initialize(generator_model)
    generator_model.build_trainer(child_model)
    # controller_model.build_trainer(child_model)

    child_ops = {
      "global_step": child_model.global_step,
      "loss": child_model.loss,
      "train_op": child_model.train_op,
      "lr": child_model.lr,
      "grad_norm": child_model.grad_norm,
      "train_acc": child_model.train_acc,
      "optimizer": child_model.optimizer,
      "valid_rl_acc": child_model.valid_rl_acc,
      "valid_acc": child_model.valid_acc,
      # "path_arc": child_model.path_arc,
      "dag_arc": child_model.dag_arc,
      "reduce_arc": child_model.reduce_arc,
      "num_train_batches": child_model.num_train_batches,
      # "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
      "eval_func": child_model.eval_once,
    }

    generator_ops = {
      "train_step": generator_model.train_step,
      "loss": generator_model.loss,
      "train_op": generator_model.train_op,
      "lr": generator_model.lr,
      "grad_norm": generator_model.grad_norm,
      "valid_acc": generator_model.valid_acc,
      "optimizer": generator_model.optimizer,
      "baseline": generator_model.baseline,
      "entropy": generator_model.sample_entropy,
      "conv_ops": generator_model.conv_ops,
      "reduce_ops": generator_model.reduce_ops,
      "sample_arc": generator_model.sample_arc,
      "skip_rate": generator_model.skip_rate,
    }

    ops = {
      "child": child_ops,
      "generator": generator_ops,
      # "controller": controller_ops,
      # "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
      # "eval_func": child_model.eval_once,
      # "num_train_batches": child_model.num_train_batches,
    }
 
    return ops


def train():
    logger.info('start...')
    if FLAGS.child_fixed_arc is None:
        images, labels = read_data(FLAGS.data_path)
    else:
        images, labels = read_data(FLAGS.data_path, num_valids=0)

    logger.info("Original Image Shape: {0}".format(images['train'].shape))
 
    g = tf.Graph()
    with g.as_default():
        ops = get_ops(images, labels)
        child_ops = ops["child"]
        generator_ops = ops["generator"]

        dc = DagController(
                num_cells=FLAGS.child_num_cells,
                num_layers=FLAGS.child_num_layers,
                cd_length=FLAGS.child_num_cells+2,
                opt_num=FLAGS.child_num_branches,
                path_pool_size=FLAGS.path_pool_size,
                k_init_selection_num=FLAGS.k_init_selection_num,
                k_best_selection_num=FLAGS.k_best_selection_num,
                max_generation=FLAGS.max_generation)
        if FLAGS.child_fixed_arc is None:
            dc.evolve_ops_dag(child_ops, generator_ops)
        else:
            dc.eval_dag_arc(child_ops)

        # pc = PathController(
        #         num_cells=FLAGS.child_num_cells,
        #         num_layers=FLAGS.child_num_layers,
        #         cd_length=FLAGS.child_num_cells+2,
        #         opt_num=FLAGS.opt_num,
        #         path_pool_size=FLAGS.path_pool_size,
        #         k_init_selection_num=FLAGS.k_init_selection_num,
        #         k_best_selection_num=FLAGS.k_best_selection_num,
        #         max_generation=FLAGS.max_generation)
        # # pc.build_path_pool_out_tensor(child_ops)
        # # pc.build_path_pool(10)
        # if FLAGS.child_fixed_arc is None:
        #     pc.build_path_pool_full_arc(child_ops)
        # else:
        #     pc.eval_dag_arc(child_ops)



def main(_):
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    logger.info("-" * 80)
    if not os.path.isdir(FLAGS.output_dir):
        logger.info("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.info("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
 
    # logger.info("-" * 80)
    # log_file = os.path.join(FLAGS.output_dir, "stdout")
    # logger.info("Logging to {}".format(log_file))
    # sys.stdout = Logger(log_file)
 
    utils.print_user_flags()
    train()
    logger.info('End.')
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

if __name__ == "__main__":
    tf.app.run()
