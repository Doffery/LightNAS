import utils

import os
import sys

import numpy as np
import tensorflow as tf

from src.models import Model
from src.image_ops import conv
from src.image_ops import fully_connected
from src.image_ops import batch_norm
from src.image_ops import batch_norm_with_mask
from src.image_ops import relu
from src.image_ops import max_pool
from src.image_ops import drop_path
from src.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.utils import get_grads
from src.utils import get_train_ops_re
from src.utils import average_gradients
from src.common_ops import create_weight


logger = utils.logger


class DagExecutor(Model):
    def __init__(self,
                 images,
                 labels,
                 num_gpus=1,
                 num_cpus=1,
                 use_aux_heads=False,
                 cutout_size=None,
                 fixed_arc=None,
                 num_layers=2,
                 num_cells=5,
                 cd_length=7,
                 out_filters=24,
                 keep_prob=1.0,
                 drop_path_keep_prob=None,
                 batch_size=32,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=0.1,
                 lr_dec_start=0,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 **kwargs
                 ):
        """
        """

        super(self.__class__, self).__init__(
            images,
            labels,
            cutout_size=cutout_size,
            batch_size=batch_size,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            lr_dec_start=lr_dec_start,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            keep_prob=keep_prob,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas,
            data_format=data_format,
            name=name)

        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.use_aux_heads = use_aux_heads
        self.num_epochs = num_epochs
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.drop_path_keep_prob = drop_path_keep_prob
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.fixed_arc = fixed_arc

        self.cd_length = cd_length
        self.cd_opt_ind = self.cd_length-2
        self.cd_end_ind = self.cd_length-1

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")

        if self.drop_path_keep_prob is not None:
            assert num_epochs is not None, "Need num_epochs to drop_path"

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]

        if self.use_aux_heads:
            self.aux_head_indices = [self.pool_layers[-1] + 1]
        logger.info('Finished init PathGenerator')
        
    def _factorized_reduction(self, x, out_filters, stride, is_training):
        """Reduces the shape of x without information loss due to striding."""
        assert out_filters % 2 == 0, (
                "Need even number of filters when using this factorized reduction.")
        if stride == 1:
            with tf.variable_scope("path_conv"):
                inp_c = self._get_C(x)
                w = create_weight("w", [1, 1, inp_c, out_filters])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                                 data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
                return x

        stride_spec = self._get_strides(stride)
        # Skip path 1
        path1 = tf.nn.avg_pool(
                x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path1_conv"):
            inp_c = self._get_C(path1)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "VALID",
                                                     data_format=self.data_format)
    
        # Skip path 2
        # First pad with 0"s on the right and bottom, then shift the filter to
        # include those 0"s that were added.
        if self.data_format == "NHWC":
            pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
            path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
            concat_axis = 3
        else:
            pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
            path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
            concat_axis = 1
    
        path2 = tf.nn.avg_pool(
                path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path2_conv"):
            inp_c = self._get_C(path2)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "VALID",
                                                     data_format=self.data_format)
    
        # Concat and apply BN
        final_path = tf.concat(values=[path1, path2], axis=concat_axis)
        final_path = batch_norm(final_path, is_training,
                                                        data_format=self.data_format)

        return final_path

    def _get_C(self, x):
        """
        Args:
            x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            # logger.info('Where am I ?2')
            # logger.info(x)
            # logger.info(x.get_shape())
            return x.get_shape()[1].value
        else:
            #logger.info('Where am I ?3')
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):
        """
        Args:
            x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        return x.get_shape()[2].value

    def _get_strides(self, stride):
        """
        Args:
            x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _apply_drop_path(self, x, layer_id):
        drop_path_keep_prob = self.drop_path_keep_prob

        layer_ratio = float(layer_id + 1) / (self.num_layers + 2)
        drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

        step_ratio = tf.to_float(self.global_step + 1) / tf.to_float(self.num_train_steps)
        step_ratio = tf.minimum(1.0, step_ratio)
        drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

        x = drop_path(x, drop_path_keep_prob)
        return x

    def _flat_filter_size(self, layer, out_filters, is_training):
        """Makes sure layers[0] and layers[1] have the same shapes."""

        # hw = [self._get_HW(layer) for layer in layers]
        # c = [self._get_C(layer) for layer in layers]
        hw = self._get_HW(layer)
        c = self._get_C(layer)

        with tf.variable_scope("calibrate"):
            # x = layers[0]
            x = layer
            # if hw[0] != hw[1]:
            #     assert hw[0] == 2 * hw[1]
            #     with tf.variable_scope("pool_x"):
            #         x = tf.nn.relu(x)
            #         x = self._factorized_reduction(x, out_filters, 2, is_training)
            # elif c[0] != out_filters:
            if c != out_filters:
                with tf.variable_scope("pool_x"):
                    # w = create_weight("w", [1, 1, c[0], out_filters])
                    w = create_weight("w", [1, 1, c, out_filters])
                    x = tf.nn.relu(x)
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
                    x = batch_norm(x, is_training, data_format=self.data_format)

            # y = layers[1]
            # if c[1] != out_filters:
            #     with tf.variable_scope("pool_y"):
            #         w = create_weight("w", [1, 1, c[1], out_filters])
            #         y = tf.nn.relu(y)
            #         y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
            #                                          data_format=self.data_format)
            #         y = batch_norm(y, is_training, data_format=self.data_format)
        # return [x, y]
        return x

    def _maybe_adjust_channel(self, layer, out_filters, is_training):
        """Makes sure layers[0] and layers[1] have the same shapes."""

        c = self._get_C(layer)

        with tf.variable_scope("adjust"):
            x = layer
            if c != out_filters:
                with tf.variable_scope("pool_x"):
                    x = tf.nn.relu(x)
                    x = self._factorized_reduction(x, out_filters, 2, is_training)

        return x

    def _maybe_calibrate_size(self, layers, out_filters, is_training):
        """Makes sure layers[0] and layers[1] have the same shapes."""

        hw = [self._get_HW(layer) for layer in layers]
        c = [self._get_C(layer) for layer in layers]

        with tf.variable_scope("calibrate"):
            x = layers[0]
            if hw[0] != hw[1]:
                assert hw[0] == 2 * hw[1]
                with tf.variable_scope("pool_x"):
                    x = tf.nn.relu(x)
                    x = self._factorized_reduction(x, out_filters, 2, is_training)
            elif c[0] != out_filters:
                with tf.variable_scope("pool_x"):
                    w = create_weight("w", [1, 1, c[0], out_filters])
                    x = tf.nn.relu(x)
                    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                                     data_format=self.data_format)
                    x = batch_norm(x, is_training, data_format=self.data_format)

            y = layers[1]
            if c[1] != out_filters:
                with tf.variable_scope("pool_y"):
                    w = create_weight("w", [1, 1, c[1], out_filters])
                    y = tf.nn.relu(y)
                    y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                                                     data_format=self.data_format)
                    y = batch_norm(y, is_training, data_format=self.data_format)
        return [x, y]


    def _model(self, images, is_training, reuse=False):
        """Compute the logits given the images."""
        # tf.get_variable_scope().reuse_variables()

        if self.fixed_arc is None:
            is_training = True

        logger.info("Model image shape:{0}".format(images))
        # tf.Print(images, [images[0]], 'image data: ')

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
        # with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs, we do not need two, or even one?
            with tf.variable_scope("stem_conv"):
                # w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                w = create_weight("w", [3, 3, 3, self.out_filters])
                logger.info("Verify reuse Weight w {0}".format(w.name))
                x = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))
            # Need Mod
            # layers = [x, x]
            
            # inital layer
            ilayer = x
            logger.info('Prelayer: {0}'.format(ilayer))

            # tf.Print(ilayer, [ilayer])

            # building layers in the micro space
            out_filters = self.out_filters
            # Need Mod, layers+2?
            for layer_id in range(self.num_layers + 2):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if layer_id not in self.pool_layers:
                        if self.fixed_arc is None:
                            x = self._enas_layer(
                               layer_id, ilayer, self.dag_arc, out_filters)
                             #     layer_id, layers, self.dag_arc, out_filters)
                        else:
                            x = self._fixed_layer(
                                layer_id, ilayer, self.dag_arc, out_filters, 1, is_training,
                                normal_or_reduction_cell="normal")
                            # layer_id, layers, self.dag_arc, out_filters, 1, is_training,
                    else:
                        out_filters *= 2
                        if self.fixed_arc is None:
                            x = self._factorized_reduction(x, out_filters, 2, is_training)
                            # layers = [layers[-1], x]
                            x = self._enas_layer(
                                layer_id, x, self.reduce_arc, out_filters)
                                # layer_id, layers, self.reduce_arc, out_filters)
                        else:
                            x = self._factorized_reduction(x, out_filters, 2, is_training)
                            x = self._fixed_layer(
                                layer_id, x, self.reduce_arc, out_filters, 1, is_training,
                                normal_or_reduction_cell="reduction")
                                # layer_id, layers, self.reduce_arc, out_filters, 2, is_training,
                    logger.info("Layer {0:>2d}: {1}".format(layer_id, x))
                    ilayer = x

                    # tf.Print(x, [layer_id, x])
                    # layers = [layers[-1], x]

                # auxiliary heads
                self.num_aux_vars = 0
                if (self.use_aux_heads and
                        layer_id in self.aux_head_indices
                        and is_training):
                    logger.info("Using aux_head at layer {0}".format(layer_id))
                    with tf.variable_scope("aux_head"):
                        aux_logits = tf.nn.relu(x)
                        aux_logits = tf.layers.average_pooling2d(
                            aux_logits, [5, 5], [3, 3], "VALID",
                            data_format=self.actual_data_format)
                        with tf.variable_scope("proj"):
                            inp_c = self._get_C(aux_logits)
                            w = create_weight("w", [1, 1, inp_c, 128])
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=True,
                                                    data_format=self.data_format)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope("avg_pool"):
                            inp_c = self._get_C(aux_logits)
                            hw = self._get_HW(aux_logits)
                            w = create_weight("w", [hw, hw, inp_c, 768])
                            aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                                      data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=True,
                                                    data_format=self.data_format)
                            aux_logits = tf.nn.relu(aux_logits)

                        with tf.variable_scope("fc"):
                            aux_logits = global_avg_pool(aux_logits,
                                                         data_format=self.data_format)
                            inp_c = aux_logits.get_shape()[1].value
                            w = create_weight("w", [inp_c, 10])
                            aux_logits = tf.matmul(aux_logits, w)
                            self.aux_logits = aux_logits

                    aux_head_variables = [
                        var for var in tf.trainable_variables() if (
                            var.name.startswith(self.name) and "aux_head" in var.name)]
                    self.num_aux_vars = count_model_params(aux_head_variables)
                    logger.info("Aux head uses {0} params".format(self.num_aux_vars))

            x = tf.nn.relu(x)
            x = global_avg_pool(x, data_format=self.data_format)
            if is_training and self.keep_prob is not None and self.keep_prob < 1.0:
                x = tf.nn.dropout(x, self.keep_prob)
            logger.info(x)
            with tf.variable_scope("fc"):
                inp_c = x.get_shape()[1].value
                w = create_weight("w", [inp_c, 10])
                x = tf.matmul(x, w)
        return x

    def _fixed_conv(self, x, f_size, out_filters, stride, is_training,
                                    stack_convs=2):
        """Apply fixed convolution.

        Args:
            stacked_convs: number of separable convs to apply.
        """

        for conv_id in range(stack_convs):
            inp_c = self._get_C(x)
            if conv_id == 0:
                strides = self._get_strides(stride)
            else:
                strides = [1, 1, 1, 1]

            with tf.variable_scope("sep_conv_{}".format(conv_id)):
                w_depthwise = create_weight("w_depth", [f_size, f_size, inp_c, 1])
                w_pointwise = create_weight("w_point", [1, 1, inp_c, out_filters])
                x = tf.nn.relu(x)
                x = tf.nn.separable_conv2d(
                    x,
                    depthwise_filter=w_depthwise,
                    pointwise_filter=w_pointwise,
                    strides=strides, padding="SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)

        return x

    def _fixed_combine(self, layers, ends, out_filters, is_training,
                                         normal_or_reduction_cell="normal"):
        """Adjust if necessary.

        Args:
            layers: a list of tf tensors of size [NHWC] of [NCHW].
            ends: a numpy tensor, [1] means is an end.
        """
        num_ends = tf.reduce_sum(ends)
        possible_end_length = self.num_cells+1
        at_least_one_end = tf.Assert(tf.greater(num_ends, 0), [ends,
                                 'No end for this layer!!!'], 100)

        out_hw = min([self._get_HW(layer)
                     for i, layer in enumerate(layers) if ends[i] == 1])
        out = []

        with tf.variable_scope("final_combine"):
            for i, layer in enumerate(layers):
                if ends[i] == 1:
                    hw = self._get_HW(layer)
                    if hw > out_hw:
                        assert hw == out_hw * 2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
                        with tf.variable_scope("calibrate_{0}".format(i)):
                            x = self._factorized_reduction(layer, out_filters, 2, is_training)
                    else:
                        x = layer
                    out.append(x)
            logger.info(out)

            # if self.data_format == "NHWC":
            #     out = tf.concat(out, axis=3)
            # elif self.data_format == "NCHW":
            #     out = tf.concat(out, axis=1)
            # else:
            #     raise ValueError("Unknown data_format '{0}'".format(self.data_format))
            # logger.info(out)

        out = tf.stack(out, axis=0)

        with tf.control_dependencies([at_least_one_end]):
            out = tf.reduce_mean(out, 0)
            logger.info(out)
            # with tf.variable_scope("final_conv"):
            #     w = create_weight("w", [possible_end_length, out_filters * out_filters])
            #     w = w[:num_ends]
            #     # w = tf.gather(w, indices, axis=0)
            #     w = tf.reshape(w, [1, 1, num_ends * out_filters, out_filters])
            #     out = tf.nn.relu(out)
            #     out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
            #                        data_format=self.data_format)
            #     out = batch_norm(out, is_training=True, data_format=self.data_format)

        # out = tf.reshape(out, tf.shape(prev_layers[0]))
        out = tf.reshape(out, tf.shape(layers[0]))

        return out

    def _fixed_layer(self, layer_id, pre_layer, arc, out_filters, stride,
                                     is_training, normal_or_reduction_cell="normal"):
        logger.info("Arc {}".format(arc))
        """
        Args:
            prev_layers: cache of previous layers. for skip connections
            is_training: for batch_norm
        """

        # assert len(prev_layers) == 2
        # layers = [prev_layers[0], prev_layers[1]]
        # layers = self._maybe_calibrate_size(layers, out_filters,
        #                                     is_training=is_training)
        layers = [pre_layer]

        # What is this for?
        with tf.variable_scope("layer_base"):
            x = layers[0]
            inp_c = self._get_C(x)
            w = create_weight("w", [1, 1, inp_c, out_filters])
            x = tf.nn.relu(x)
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                             data_format=self.data_format)
            x = batch_norm(x, is_training, data_format=self.data_format)
            layers[0] = x

        ends = np.zeros([self.num_cells + 1], dtype=np.int32)
        f_sizes = [3, 5]
        for cell_id in range(self.num_cells):
            with tf.variable_scope("cell_{}".format(cell_id)):
                start_idx = (self.cd_length) * cell_id
                prev_idxs = arc[start_idx:start_idx+self.cd_opt_ind]
                x_op = arc[start_idx+self.cd_opt_ind]
                if arc[start_idx+self.cd_end_ind] == 1:
                    ends[cell_id+1] = 1
                link_layers = []
                num_used_layers = self.num_cells-tf.size(tf.where(tf.equal(prev_idxs, 0)))
                at_least_one = tf.Assert(tf.greater(num_used_layers, 0), [prev_idxs,
                                         'No previous layer chosen!!!'], 100)

                def _not_selected():
                    return tf.zeros(shape=tf.shape(layers[0]),
                                    dtype=tf.float32)

                def _seleted(l):
                    return l

                for i in range(cell_id+1):
                    slayer = tf.cond(tf.equal(prev_idxs[i], 0), _not_selected,
                                     lambda: _seleted(layers[i]))
                    link_layers.append(slayer)
                layer_add_sum = tf.add_n(link_layers)
                x = layer_add_sum
                with tf.control_dependencies([at_least_one]):
                    x = tf.divide(x, tf.to_float(num_used_layers))
                # x_id = arc[4 * cell_id]
                # used[x_id] += 1
                # x_op = arc[4 * cell_id + 1]
                # x = layers[x_id]
                x_stride = stride
                with tf.variable_scope("x_conv"):
                    if x_op in [1, 2]:
                        f_size = f_sizes[x_op-1]
                        x = self._fixed_conv(x, f_size, out_filters, x_stride, is_training)
                    elif x_op in [3, 4]:
                        inp_c = self._get_C(x)
                        if x_op == 3:
                            x = tf.layers.average_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            x = tf.layers.max_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            x = tf.nn.relu(x)
                            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                                             data_format=self.data_format)
                            x = batch_norm(x, is_training, data_format=self.data_format)
                    else:
                        inp_c = self._get_C(x)
                        if x_stride > 1:
                            assert x_stride == 2
                            x = self._factorized_reduction(x, out_filters, 2, is_training)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            x = tf.nn.relu(x)
                            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                            x = batch_norm(x, is_training, data_format=self.data_format)
                    if (x_op in [0, 1, 2, 3] and
                            self.drop_path_keep_prob is not None and
                            is_training):
                        x = self._apply_drop_path(x, layer_id)

                out = x
                layers.append(out)
        out = self._fixed_combine(layers, ends, out_filters, is_training,
                                                            normal_or_reduction_cell)

        return out

    # def _enas_dag_cell(self, x, curr_cell, prev_cell, op_id, out_filters):
    def _enas_dag_cell(self, prev_layers, curr_cell,
                       prev_idxs, op_id, out_filters):
        """Performs an enas operation specified by op_id."""
        # num_possible_inputs = curr_cell + 1
        # add and average all prev cells to x

        # prev_idxs = tf.Print(prev_idxs, [prev_idxs, 'prev_idx'],
        #                      message='Debug: ', summarize=100)

        def _not_selected():
            return tf.zeros(shape=tf.shape(prev_layers[0]),
                            dtype=tf.float32)

        def _seleted(l):
            return l

        link_layers = []
        num_used_layers = self.num_cells-tf.size(tf.where(tf.equal(prev_idxs, 0)))
        at_least_one = tf.Assert(tf.greater(num_used_layers, 0), [prev_idxs,
                                 'No previous layer chosen!!!'], 100)
        for i in range(curr_cell+1):
            slayer = tf.cond(tf.equal(prev_idxs[i], 0), _not_selected,
                             lambda: _seleted(prev_layers[i]))
            link_layers.append(slayer)
        layer_add_sum = tf.add_n(link_layers)
        x = layer_add_sum
        with tf.control_dependencies([at_least_one]):
            x = tf.divide(x, tf.to_float(num_used_layers))
        # x = prev_layers[0]
        # logger.info('Choose pre_layer: {0}'.format(x))

        with tf.variable_scope("avg_pool"):
            avg_pool = tf.layers.average_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            avg_pool_c = self._get_C(avg_pool)
            # logger.info(avg_pool_c)
            if avg_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    # w = create_weight(
                    #     "w", [num_possible_inputs, avg_pool_c * out_filters])
                    # w = w[prev_cell]
                    w = create_weight(
                        "w", [avg_pool_c * out_filters])
                    w = tf.reshape(w, [1, 1, avg_pool_c, out_filters])
                    avg_pool = tf.nn.relu(avg_pool)
                    avg_pool = tf.nn.conv2d(avg_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    avg_pool = batch_norm(avg_pool, is_training=True,
                                          data_format=self.data_format)

        with tf.variable_scope("max_pool"):
            max_pool = tf.layers.max_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            max_pool_c = self._get_C(max_pool)
            if max_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    # w = create_weight(
                    #     "w", [num_possible_inputs, max_pool_c * out_filters])
                    # w = w[prev_cell]
                    w = create_weight(
                        "w", [max_pool_c * out_filters])
                    w = tf.reshape(w, [1, 1, max_pool_c, out_filters])
                    max_pool = tf.nn.relu(max_pool)
                    max_pool = tf.nn.conv2d(max_pool, w, strides=[1, 1, 1, 1],
                                            padding="SAME", data_format=self.data_format)
                    max_pool = batch_norm(max_pool, is_training=True,
                                          data_format=self.data_format)

        x_c = self._get_C(x)
        if x_c != out_filters:
            with tf.variable_scope("x_conv"):
                # w = create_weight("w", [num_possible_inputs, x_c * out_filters])
                # w = w[prev_cell]
                w = create_weight("w", [x_c * out_filters])
                w = tf.reshape(w, [1, 1, x_c, out_filters])
                x = tf.nn.relu(x)
                x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                                                 data_format=self.data_format)
                x = batch_norm(x, is_training=True, data_format=self.data_format)

        out = [ x,  # This is for default useless linkage
                self._enas_dag_conv(x, curr_cell, 3, out_filters),
                self._enas_dag_conv(x, curr_cell, 5, out_filters),
                avg_pool,
                max_pool,
                # x,
        ]

        out = tf.stack(out, axis=0)
        out = out[op_id, :, :, :, :]
        return out

    # def _enas_daga_conv(self, x, curr_cell, prev_cell, filter_size, out_filters,
    #                              stack_conv=2):
    def _enas_dag_conv(self, x, curr_cell, filter_size, out_filters,
                                 stack_conv=2):
        """Performs an enas convolution specified by the relevant parameters."""

        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
            # num_possible_inputs = curr_cell + 2
            for conv_id in range(stack_conv):
                with tf.variable_scope("stack_{0}".format(conv_id)):
                    # create params and pick the correct path
                    inp_c = self._get_C(x)
                    w_depthwise = create_weight(
                        "w_depth", [filter_size * filter_size * inp_c])
                    w_depthwise = tf.reshape(
                        w_depthwise, [filter_size, filter_size, inp_c, 1])

                    w_pointwise = create_weight(
                        "w_point", [inp_c * out_filters])
                    w_pointwise = tf.reshape(w_pointwise, [1, 1, inp_c, out_filters])

                    with tf.variable_scope("bn"):
                        zero_init = tf.initializers.zeros(dtype=tf.float32)
                        one_init = tf.initializers.ones(dtype=tf.float32)
                        offset = create_weight(
                            "offset", [out_filters],
                            initializer=zero_init)
                        scale = create_weight(
                            "scale", [out_filters],
                            initializer=one_init)

                    # the computations
                    x = tf.nn.relu(x)
                    x = tf.nn.separable_conv2d(
                        x,
                        depthwise_filter=w_depthwise,
                        pointwise_filter=w_pointwise,
                        strides=[1, 1, 1, 1], padding="SAME",
                        data_format=self.data_format)
                    x, _, _ = tf.nn.fused_batch_norm(
                        x, scale, offset, epsilon=1e-5, data_format=self.data_format,
                        is_training=True)
        return x


    # def _enas_layer(self, layer_id, prev_layers, arc, out_filters):
    def _enas_layer(self, layer_id, prev_layer, arc, out_filters):
        """
        Args:
            layer_id: current layer
            prev_layers: cache of previous layers. for skip connections
            start_idx: where to start looking at. technically, we can infer this
                from layer_id, but why bother...
        """

        # assert len(prev_layers) == 2, "need exactly 2 inputs"
        # layers = [prev_layers[0], prev_layers[1]]
        # layers = self._maybe_calibrate_size(layers, out_filters, is_training=True)
        layers = [prev_layer]
        used = []
        possible_end_length = self.num_cells+1
        for cell_id in range(self.num_cells):
            prev_layers = layers
            # prev_layers = tf.stack(layers, axis=0)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                # with tf.variable_scope("x"):
                # x_id = arc[2 * cell_id]
                # x_op = arc[2 * cell_id + 1]
                # x = prev_layers[x_id, :, :, :, :]
                # x = self._enas_cell(x, cell_id, x_id, x_op, out_filters)
                start_idx = (self.cd_length) * cell_id
                x = self._enas_dag_cell(prev_layers, cell_id,
                                    arc[start_idx:start_idx+self.cd_opt_ind],
                                    arc[start_idx+self.cd_opt_ind], out_filters)
                # x_used = tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32)
                x_used = tf.cond(tf.equal(arc[start_idx+self.cd_end_ind], 0),
                        lambda: tf.zeros([possible_end_length], tf.int32),
                        lambda: tf.one_hot(cell_id+1, depth=possible_end_length,
                                            dtype=tf.int32))

                # with tf.variable_scope("y"):
                #     y_id = arc[4 * cell_id + 2]
                #     y_op = arc[4 * cell_id + 3]
                #     y = prev_layers[y_id, :, :, :, :]
                #     y = self._enas_cell(y, cell_id, y_id, y_op, out_filters)
                #     y_used = tf.one_hot(y_id, depth=self.num_cells + 2, dtype=tf.int32)

                # out = x + y
                # used.extend([x_used, y_used])
                # layers.append(out)
                used.append(x_used)
                layers.append(x)
        # out = x

        tf.summary.tensor_summary('used', used)
        used = tf.add_n(used)
        # used = tf.Print(used, [used], 'Used: ', summarize=100)

        '''
        dag2d = tf.reshape(self.dag_arc, [self.num_cells, self.num_cells+1])
        used = tf.slice(dag2d, [0, 0], [self.num_cells, 0])
        '''

        indices = tf.where(tf.equal(used, 1))
        indices = tf.to_int32(indices)
        indices = tf.reshape(indices, [-1])
        num_outs = tf.size(indices)
        out = tf.stack(layers, axis=0)
        out = tf.gather(out, indices, axis=0)

        at_least_one_end = tf.Assert(tf.greater(num_outs, 0), 
                                     [num_outs, indices])  # at least one

        with tf.control_dependencies([at_least_one_end]):
            out = tf.reduce_mean(out, 0)
        # inp = prev_layer
        # if self.data_format == "NHWC":
        #     N = tf.shape(inp)[0]
        #     H = tf.shape(inp)[1]
        #     W = tf.shape(inp)[2]
        #     C = tf.shape(inp)[3]
        #     out = tf.transpose(out, [1, 2, 3, 0, 4])
        #     out = tf.reshape(out, [N, H, W, num_outs * out_filters])
        # elif self.data_format == "NCHW":
        #     N = tf.shape(inp)[0]
        #     C = tf.shape(inp)[1]
        #     H = tf.shape(inp)[2]
        #     W = tf.shape(inp)[3]
        #     out = tf.transpose(out, [1, 0, 2, 3, 4])
        #     out = tf.reshape(out, [N, num_outs * out_filters, H, W])
        # else:
        #     raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        # with tf.control_dependencies([at_least_one_end]):
        #     with tf.variable_scope("final_conv"):
        #         w = create_weight("w", [possible_end_length, out_filters * out_filters])
        #         w = tf.gather(w, indices, axis=0)
        #         w = tf.reshape(w, [1, 1, num_outs * out_filters, out_filters])
        #         out = tf.nn.relu(out)
        #         out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
        #                            data_format=self.data_format)
        #         out = batch_norm(out, is_training=True, data_format=self.data_format)

        # out = tf.reshape(out, tf.shape(prev_layers[0]))
        out = tf.reshape(out, tf.shape(prev_layer))
        return out


    def _enas_layer_path(self, layer_id, prev_layer, path, out_filters):
        """
        Args:
            layer_id: current layer
            prev_layers: cache of previous layers. for skip connections
            start_idx: where to start looking at. technically, we can infer this
                from layer_id, but why bother...
        """

        # assert len(prev_layers) == 2, "need exactly 2 inputs"
        # layers = [prev_layers[0], prev_layers[1]]
        # layers = self._maybe_calibrate_size(layers, out_filters, is_training=True)
        layers = [prev_layer]
        used = []
        possible_end_length = self.num_cells+1
        for cell_id in range(self.num_cells):
            prev_layers = layers
            # prev_layers = tf.stack(layers, axis=0)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                # with tf.variable_scope("x"):
                # x_id = arc[2 * cell_id]
                # x_op = arc[2 * cell_id + 1]
                # x = prev_layers[x_id, :, :, :, :]
                # x = self._enas_cell(x, cell_id, x_id, x_op, out_filters)
                start_idx = cell_id
                for ind in range(cell_id):
                    pre_list = tf.cond(tf.equal(path[start_idx], 0),
                            lambda: tf.zeros([possible_end_length], tf.int32))
                x = self._enas_dag_cell(prev_layers, cell_id,
                                    arc[start_idx:start_idx+self.cd_opt_ind],
                                    arc[start_idx], out_filters)
                # x_used = tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32)
                x_used = tf.cond(tf.equal(arc[start_idx+self.cd_end_ind], 0),
                        lambda: tf.zeros([possible_end_length], tf.int32),
                        lambda: tf.one_hot(cell_id+1, depth=possible_end_length,
                                            dtype=tf.int32))

                # with tf.variable_scope("y"):
                #     y_id = arc[4 * cell_id + 2]
                #     y_op = arc[4 * cell_id + 3]
                #     y = prev_layers[y_id, :, :, :, :]
                #     y = self._enas_cell(y, cell_id, y_id, y_op, out_filters)
                #     y_used = tf.one_hot(y_id, depth=self.num_cells + 2, dtype=tf.int32)

                # out = x + y
                # used.extend([x_used, y_used])
                # layers.append(out)
                used.append(x_used)
                layers.append(x)
        # out = x

        tf.summary.tensor_summary('used', used)
        used = tf.add_n(used)
        # used = tf.Print(used, [used], 'Used: ', summarize=100)

        '''
        dag2d = tf.reshape(self.dag_arc, [self.num_cells, self.num_cells+1])
        used = tf.slice(dag2d, [0, 0], [self.num_cells, 0])
        '''

        indices = tf.where(tf.equal(used, 1))
        indices = tf.to_int32(indices)
        indices = tf.reshape(indices, [-1])
        num_outs = tf.size(indices)
        out = tf.stack(layers, axis=0)
        out = tf.gather(out, indices, axis=0)

        at_least_one_end = tf.Assert(tf.greater(num_outs, 0), 
                                     [num_outs, indices])  # at least one

        inp = prev_layer
        if self.data_format == "NHWC":
            N = tf.shape(inp)[0]
            H = tf.shape(inp)[1]
            W = tf.shape(inp)[2]
            C = tf.shape(inp)[3]
            out = tf.transpose(out, [1, 2, 3, 0, 4])
            out = tf.reshape(out, [N, H, W, num_outs * out_filters])
        elif self.data_format == "NCHW":
            N = tf.shape(inp)[0]
            C = tf.shape(inp)[1]
            H = tf.shape(inp)[2]
            W = tf.shape(inp)[3]
            out = tf.transpose(out, [1, 0, 2, 3, 4])
            out = tf.reshape(out, [N, num_outs * out_filters, H, W])
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        with tf.control_dependencies([at_least_one_end]):
            with tf.variable_scope("final_conv"):
                w = create_weight("w", [possible_end_length, out_filters * out_filters])
                w = tf.gather(w, indices, axis=0)
                w = tf.reshape(w, [1, 1, num_outs * out_filters, out_filters])
                out = tf.nn.relu(out)
                out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
                                   data_format=self.data_format)
                out = batch_norm(out, is_training=True, data_format=self.data_format)

        # out = tf.reshape(out, tf.shape(prev_layers[0]))
        out = tf.reshape(out, tf.shape(prev_layer))
        return out

    # override
    def _build_train(self):
        logger.info("-" * 80)
        logger.info("Build train graph")
        tower_grads = []
        tower_grad_norm = []
        choose_pu = "/gpu"
        num_pu = self.num_gpus
        if self.num_gpus == 0:
            choose_pu = "/cpu"
            num_pu = self.num_cpus
        for i in range(num_pu):  # FLAGS.num_gpus
            with tf.device(choose_pu+':%d' % i):
                logits = self._model(self.x_train, is_training=True)
                log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.y_train)
                self.loss = tf.reduce_mean(log_probs)

                if self.use_aux_heads:
                    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.aux_logits, labels=self.y_train)
                    self.aux_loss = tf.reduce_mean(log_probs)
                    train_loss = self.loss + 0.4 * self.aux_loss
                else:
                    train_loss = self.loss

                self.train_preds = tf.argmax(logits, axis=1)
                self.train_preds = tf.to_int32(self.train_preds)
                self.train_acc = tf.equal(self.train_preds, self.y_train)
                self.train_acc = tf.to_int32(self.train_acc)
                self.train_acc = tf.reduce_sum(self.train_acc)

                tf.summary.scalar('train acc', self.train_acc)

                tf_variables = [
                    var for var in tf.trainable_variables() if (
                        var.name.startswith(self.name) and "aux_head" not in var.name)]

                self.num_vars = count_model_params(tf_variables)
                logger.info("Model has {0} params".format(self.num_vars))
                grads, grad_norm = get_grads(train_loss,
                                             tf_variables,
                                             clip_mode=self.clip_mode,
                                             grad_bound=self.grad_bound,
                                             l2_reg=self.l2_reg)
                tower_grads.append(grads)
                tower_grad_norm.append(grad_norm)
        grads = average_gradients(tower_grads)
        self.grad_norm = tf.reduce_mean(tower_grad_norm)
        tf_variables = [
            var for var in tf.trainable_variables() if (
                var.name.startswith(self.name) and "aux_head" not in var.name)]
        self.train_op, self.lr, self.optimizer = get_train_ops_re(
            grads,
            tf_variables,
            self.global_step,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            lr_cosine=self.lr_cosine,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            lr_T_0=self.lr_T_0,
            lr_T_mul=self.lr_T_mul,
            num_train_batches=self.num_train_batches,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas)

    # override
    def _build_valid(self):
        if self.x_valid is not None:
            logger.info("-" * 80)
            logger.info("Build valid graph")
            logits = self._model(self.x_valid, False, reuse=True)
            self.valid_preds = tf.argmax(logits, axis=1)
            self.valid_preds = tf.to_int32(self.valid_preds)
            self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
            self.valid_acc = tf.to_int32(self.valid_acc)
            self.valid_acc = tf.reduce_sum(self.valid_acc)

            tf.summary.scalar('validation acc', self.valid_acc)

    # override
    def _build_test(self):
        logger.info("-" * 80)
        logger.info("Build test graph")
        logits = self._model(self.x_test, False, reuse=True)
        self.test_preds = tf.argmax(logits, axis=1)
        self.test_preds = tf.to_int32(self.test_preds)
        self.test_acc = tf.equal(self.test_preds, self.y_test)
        self.test_acc = tf.to_int32(self.test_acc)
        self.test_acc = tf.reduce_sum(self.test_acc)

        tf.summary.scalar('test acc', self.test_acc)

    # override
    def build_valid_rl(self, shuffle=False):
        logger.info("-" * 80)
        logger.info("Build valid graph on shuffled data")
        with tf.device("/cpu:0"):
            # shuffled valid data: for choosing validation model
            if not shuffle and self.data_format == "NCHW":
                self.images["valid_original"] = np.transpose(
                    self.images["valid_original"], [0, 3, 1, 2])
            x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
                [self.images["valid_original"], self.labels["valid_original"]],
                batch_size=self.batch_size,
                capacity=25000,
                enqueue_many=True,
                min_after_dequeue=0,
                num_threads=16,
                seed=self.seed,
                allow_smaller_final_batch=True,
            )

            def _pre_process(x):
                x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
                x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
                x = tf.image.random_flip_left_right(x, seed=self.seed)
                if self.data_format == "NCHW":
                    x = tf.transpose(x, [2, 0, 1])
                return x

            if shuffle:
                x_valid_shuffle = tf.map_fn(
                    _pre_process, x_valid_shuffle, back_prop=False)

        logits = self._model(x_valid_shuffle, is_training=True, reuse=True)
        valid_shuffle_preds = tf.argmax(logits, axis=1)
        valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
        self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
        self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
        self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)
        self.valid_rl_acc = (tf.to_float(self.valid_shuffle_acc) /
                             tf.to_float(self.batch_size))
        

    def initialize(self, generator_model):
        if self.fixed_arc is None:
            # self.dag_arc, self.reduce_arc = generator_model.sample_arc
            self.dag_arc = tf.placeholder(tf.int32, shape=(
                                          self.num_cells*self.cd_length))
            self.reduce_arc = tf.placeholder(tf.int32, shape=(
                                          self.num_cells*(self.cd_length)))
        else:
            fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
            self.dag_arc = fixed_arc[:self.cd_length * self.num_cells]
            self.reduce_arc = fixed_arc[self.cd_length * self.num_cells:]
        # self.path_arc = tf.placeholder(tf.int32, shape=(5))
        # if self.fixed_arc is None:
        #     # self.path_arc = tf.placeholder(tf.int32, shape=(self.num_cells))
        #     self.dag_arc = tf.placeholder(tf.int32, shape=(
        #                                   self.num_cells*self.cd_length))
        #     self.reduce_arc = tf.placeholder(tf.int32, shape=(
        #                                   self.num_cells*(self.cd_length)))
        #     self.dag_arc = tf.Print(self.dag_arc, [self.dag_arc],
        #                             'dag_arc: ', summarize=100)
        #     self.reduce_arc = tf.Print(self.reduce_arc, [self.reduce_arc],
        #                                'reduce_arc: ', summarize=100)
        #     # self.dag_end_points = tf.placeholder(tf.int32, shape=(
        #     #                               self.num_cells+1))
        #     # self.path_arc = controller_model.sample_path_arc
        #     # self.path_pool = []
        #     # self.path_pool.append(self.path_arc)
        # else:
        #     fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
        #     logger.info(fixed_arc)
        #     # fixed_arc.reshape()
        #     self.dag_arc = fixed_arc[:self.num_cells*self.cd_length]
        #     self.reduce_arc = fixed_arc[self.num_cells*self.cd_length:]

        # with tf.variable_scope(tf.get_variable_scope()):
        self._build_train()
        self._build_valid()
        # if self.fixed_arc is None:
        #     self.build_valid_rl()
        # else:
        #     self.valid_rl_acc = None
        self._build_test()
