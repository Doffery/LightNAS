import numpy as np
import tensorflow as tf
import utils

from utils import get_train_ops
from common_ops import stack_lstm
from common_ops import blstm
logger = utils.logger

# This class manages the RNN to generate paths 
# also maybe combine paths to dag
class DagGenerator():

    def __init__(self,
                 num_branches=4,
                 num_cells=6,
                 num_rand_head=3,
                 lstm_size=32,
                 lstm_num_layers=1, # one layer is enough?
                 lstm_keep_prob=1.0,
                 batch_size=8,
                 tanh_constant=None,
                 op_tanh_reduce=1.0,
                 temperature=None,
                 lr_init=1e-3,
                 lr_dec_start=0,
                 lr_dec_every=100,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 entropy_weight=None,
                 clip_mode=None,
                 grad_bound=None,
                 use_critic=False,
                 bl_dec=0.999,
                 optim_algo="adam",
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 name="generator"):
        logger.info("-" * 80)
        logger.info("Building ConvController")

        # self.search_for = search_for
        # self.search_whole_channels = search_whole_channels
        self.num_cells = num_cells + num_rand_head
        self.num_branches = num_branches

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers 
        self.lstm_keep_prob = lstm_keep_prob
        self.batch_size = batch_size
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.temperature = temperature
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.use_critic = use_critic
        self.bl_dec = bl_dec

        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.name = name

        self._create_params()
        self.conv_ops = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_cells))
        self.reduce_ops = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_cells))
        arc_seq_1, entropy_1, log_prob_1, c, h = self._build_sampler(self.conv_ops)
        arc_seq_2, entropy_2, log_prob_2, _, _ = self._build_sampler(self.reduce_ops, prev_c=c, prev_h=h)
        # sample_arc = (self._to_dag(arc_seq_1, self.conv_ops), 
        #                    self._to_dag(arc_seq_2, self.reduce_ops))
        # self.arcs.append(sample_arc)
        self.sample_arc = (arc_seq_1, arc_seq_2)
        self.entropies = entropy_1 + entropy_2
        self.log_probs = log_prob_1 + log_prob_2

    def _to_dag(self, path, ops):
        ops = tf.multiply(ops, path)

        # if path is empty, add an identity layer
        not_empty = tf.reduce_sum(path)
        path = tf.cond(tf.equal(not_empty, 0),
                       lambda: tf.one_hot(0, self.num_cells, dtype=tf.int32),
                       lambda: path)
        layers = []
        for i in range(self.num_cells):
            pre_layer = tf.one_hot(0, self.num_cells, dtype=tf.int32)
            for j in range(i):
                pre_layer = tf.cond(tf.equal(path[j], 0),
                       lambda: pre_layer,
                       lambda: tf.one_hot(j+1, self.num_cells, dtype=tf.int32))
            tmp = tf.add(tf.one_hot(0, self.num_cells, dtype=tf.int32),
                        tf.one_hot(0, self.num_cells, dtype=tf.int32))
            pre_layer = tf.cond(tf.equal(path[i], 0),
                   lambda: tmp, lambda: pre_layer)
            end = tf.cond(tf.equal(path[i], 0),
                    lambda: tf.constant([0], dtype=tf.int32),
                    lambda: tf.constant([1], dtype=tf.int32))
            for k in range(i+1, self.num_cells):
                end = tf.cond(tf.equal(path[k], 0),
                        lambda: end, lambda: tf.constant([0], dtype=tf.int32))
            layers.append(tf.concat([pre_layer, ops[i:i+1], end], 0))
    
        return tf.concat(layers, 0)

    def _create_params(self):
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope("lstm"):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with tf.variable_scope("layer_{}".format(layer_id)):
                        w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
                        self.w_lstm.append(w)

            # self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
            self.out_size = 2
            with tf.variable_scope("emb"):
                self.w_emb = tf.get_variable("w", [self.num_branches, self.lstm_size])
            with tf.variable_scope("softmax"):
                self.w_soft = tf.get_variable("w", [self.lstm_size, self.out_size])
                b_init = np.array([10.0, 0.0], dtype=np.float32)
                self.b_soft = tf.get_variable(
                    "b", [1, self.out_size],
                    initializer=tf.constant_initializer(b_init))

                b_soft_no_learn = np.array(
                    [0.25, -0.25], dtype=np.float32)
                b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.out_size])
                self.b_soft_no_learn = tf.constant(b_soft_no_learn, dtype=tf.float32)

            '''
            with tf.variable_scope("attention"):
                self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
                self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
                self.v_attn = tf.get_variable("v", [self.lstm_size, 1])
            '''

    def _build_sampler(self, inputs, prev_c=None, prev_h=None, use_bias=False):
        """Build the sampler ops and the log_prob ops."""

        logger.info("-" * 80)
        logger.info("Build controller sampler")

        arc_seq = tf.TensorArray(tf.int32, size=self.num_cells*self.batch_size)  # 4
        # arc_seq = tf.reshape(arc_seq, [self.batch_size, self.num_cells])
        if prev_c is None:
            assert prev_h is None, "prev_c and prev_h must both be None"
            prev_c = tf.zeros([self.batch_size, 1, self.lstm_size], tf.float32)
            prev_h = tf.zeros([self.batch_size, 1, self.lstm_size], tf.float32)
        # inputs = self.g_emb

        '''
        for layer_id in range(1):
            next_c, next_h = stack_lstm(inputs[layer_id], prev_c, prev_h, self.w_lstm)
            print(type(next_c))
            print(type(next_h))
            print(type(next_h[-1]))
            print("###########################")
            prev_c, prev_h = next_c, next_h
        '''

        def _condition(layer_id, *args):
            return tf.less(layer_id, self.num_cells)  # + 2

        def _body(layer_id, inputs, prev_c, prev_h, arc_seq,
                  entropy, log_prob):
            start_id = 1 * (layer_id)  # - 2
            inp = tf.nn.embedding_lookup(self.w_emb, inputs[:, layer_id])
            # for i in range(1):    # index, choose with attention or only softmax?
            logger.info(prev_h)
            next_c, next_h = blstm(inp, prev_c, prev_h, self.w_lstm[0], self.batch_size)
            # next_c, next_h = stack_lstm(inp, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            for i in range(self.batch_size):
                logits = tf.matmul(next_h[i], self.w_soft) + self.b_soft
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    op_tanh = self.tanh_constant / self.op_tanh_reduce
                    logits = op_tanh * tf.tanh(logits)
                if use_bias:
                    logits += self.b_soft_no_learn
                op_id = tf.multinomial(logits, 1)
                op_id = tf.to_int32(op_id)
                op_id = tf.reshape(op_id, [1])
                arc_seq = arc_seq.write(i*self.num_cells+start_id, op_id)
                curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=op_id)
                log_prob += curr_log_prob*tf.one_hot(i, self.batch_size, dtype=tf.float32)
                curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.nn.softmax(logits)))
                entropy += curr_ent*tf.one_hot(i, self.batch_size, dtype=tf.float32)

            return (layer_id + 1, inputs, next_c, next_h,
                    arc_seq, entropy, log_prob)

        loop_vars = [
            tf.constant(0, dtype=tf.int32, name="layer_id"),
            inputs,
            prev_c,
            prev_h,
            arc_seq,
            tf.zeros([self.batch_size], dtype=tf.float32, name="entropy"),
            tf.zeros([self.batch_size], dtype=tf.float32, name="log_prob"),
        ]
        
        loop_outputs = tf.while_loop(_condition, _body, loop_vars,
                                     parallel_iterations=1)

        arc_seq = loop_outputs[-3].stack()
        arc_seq = tf.reshape(arc_seq, [-1])
        entropy = loop_outputs[-2] ###
        log_prob = loop_outputs[-1]
        # entropy = tf.reduce_sum(loop_outputs[-2]) ###
        # log_prob = tf.reduce_sum(loop_outputs[-1])

        last_c = loop_outputs[-5]
        last_h = loop_outputs[-4]

        # Combine arc_seq(use ops or not) and input(ops template) to a path dag

        arc_seq = tf.Print(arc_seq, [arc_seq, 'arc_seq'], message='Debug: ', summarize=100)
        return arc_seq, entropy, log_prob, last_c, last_h

    def build_trainer(self, child_model):
        child_model.build_valid_rl()
        # self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
        #                                     tf.to_float(child_model.batch_size))
        # self.reward = self.valid_acc  # tf.placeholder(tf.float32, shape=(1))
        self.reward_data = tf.placeholder(tf.float32, shape=(self.batch_size))
        self.reward = self.reward_data

        if self.entropy_weight is not None:
            self.reward += self.entropy_weight * self.entropies

        self.sample_log_prob = self.log_probs
        self.baseline = tf.Variable([0.0 for i in range(self.batch_size)], trainable=False)
        baseline_update = tf.assign_sub(
            self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

        with tf.control_dependencies([baseline_update]):
            self.reward = tf.identity(self.reward)

        self.loss = self.sample_log_prob * (self.reward - self.baseline)
        self.loss = tf.reduce_sum(self.loss)

        tf.summary.scalar('rl loss', self.loss)

        self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")

        tf_variables = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        logger.info("-" * 80)
        for var in tf_variables:
            logger.info(var)

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            self.loss,
            tf_variables,
            self.train_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas)

        self.skip_rate = tf.constant(0.0, dtype=tf.float32)

