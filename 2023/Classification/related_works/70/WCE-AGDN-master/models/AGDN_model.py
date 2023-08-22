

from sklearn.metrics import accuracy_score
import os
import time
import shutil
from tqdm import tqdm
from ops import generate_seg
from ops1 import deform_con2v, conv, hw_flatten
from datetime import timedelta
import numpy as np
from trainable_image_sampler import get_resampled_images
from high_low_res_data_provider import Data_set
# import pandas as pd  # used to write and read csv files.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # assign the GPU.

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, growth_rate, depth,
                 total_blocks, keep_prob, lamda,
                 weight_decay, nesterov_momentum, model_type,
                 should_save_logs, should_save_model, which_split,
                 renew_logs=False,
                 reduction=0.5,
                 bc_mode=True,
                 **kwargs):

        self.data_shape = (128, 128, 3)
        self.n_classes = 8
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        # the depth is consisted of layers in dense blocks and layers in transition layers.
        # the number of layers in different blocks.
        # self.layers_per_block = [6,12,24,16]
        self.layers_per_block = [4, 8, 12, 8]
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        self.which_split = which_split
        self.lamda = lamda

        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "totally %d layers." % (
                      model_type, self.total_blocks, self.depth))
        if bc_mode:
            # the layers in each block is consisted of bottleneck layers and composite layers,
            # so the number of composite layers should be half the total number.
            print("Build %s model with %d blocks, "
                  "totally %d layers." % (
                      model_type, self.total_blocks, self.depth))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        # self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        self.batch_size = 8

        self.num_train = 200
        self.num_test = 48

        # self.num_train = 80
        # self.num_test =600

        self.src_size = 336
        self.dst_size = 128

        self.margin = 0.05

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        # config = tf.compat.v1.ConfigProto()
        # restrict model GPU memory utilization to min required
        # config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session()

        # tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.compat.v1.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    # if the save_path exists, use the save path
    # else create a save path
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            # save_path = 'saves/%s' % self.model_identifier
            save_path = 'previous_ckpts/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}".format(
            self.model_type, self.growth_rate, self.depth)

    def save_model(self, global_step=None):
        # save the model to the save path.
        self.saver.save(self.sess, self.save_path, global_step=global_step)
        print("Model saved to %s" % self.save_path)

    def load_model(self):
        self.saver.restore(self.sess, self.save_path)
        # assert cannot load the model.
        if not os.path.exists(self.save_path + '.meta'):
            print("The save path does not exist!")
            return

        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss1, loss2, loss3, TE_loss, epoch, prefix, should_print=True):
        if should_print:
            print('')
            print("mean TE loss: %f" % (TE_loss))

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='loss1_%s' %
                             prefix, simple_value=float(loss1)),
            tf.Summary.Value(tag='loss2_%s' %
                             prefix, simple_value=float(loss2)),
        ])

        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='images_input1')

        self.high_res_images = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.src_size, self.src_size, 3],
            name='high_resolution_input_images')

        self.labels = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.n_classes],
            name='labels')

        self.gt_map = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.dst_size, self.dst_size, 1],
            name='gt_map')

        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # the function batch_norm, conv2d, dropout are defined in the following part.
        with tf.compat.v1.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)

            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.compat.v1.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, block, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block[block]):
            with tf.compat.v1.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and then average
        pooling
        """
        # call composite function with 1x1 kernel
        # reduce the number of feature maps by compression
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def TLFA(self, x, channels, de=4, scope='TLFA', trainable=True, reuse=False):
        """
        The Third-order Long-range Feature Aggregation module,
        containing deformable convolution and second-order covariance.
        """
        with tf.compat.v1.variable_scope(scope, reuse=reuse):

            c_num = channels // de

            f, offset1 = deform_con2v(
                x, num_outputs=c_num, kernel_size=3, stride=1, trainable=trainable,  name=scope+'f_conv', reuse=reuse)
            f = tf.nn.relu(f)  # [bs, h, w, c_num]

            h = conv(x, channels, kernel=1, stride=1, sn=True,
                     scope='h_conv')  # [bs, h, w, c_num]

            # compute the second-order spatial correlation.
            I_hat = (-1./c_num/c_num) * \
                tf.ones([c_num, c_num]) + (1./c_num) * tf.eye(c_num)
            I_hat = tf.tile(tf.expand_dims(I_hat, 0), [
                            x.shape[0], 1, 1])  # (8, c, c)
            correlation = tf.matmul(
                tf.matmul(hw_flatten(f), I_hat), hw_flatten(f), False, True)
            # correlation = tf.nn.softmax(correlation/(tf.sqrt(tf.cast(c_num, tf.float32))), axis = -1) # (8, hw, hw)
            correlation = tf.nn.softmax(correlation, axis=-1)

            # aggregated feature
            o = tf.matmul(correlation, hw_flatten(h))  # [bs, hw, C]

            gamma = tf.get_variable(
                "gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            output = gamma * o + x

            # print("TLFA output:", output)
        return output

    # after block4, convert the 7*7 feature map to 1*1 by average pooling.

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        spatial_features = output

        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # print(output.shape)
        # FC

        features = tf.reshape(output, [self.batch_size, -1])

        with tf.compat.v1.variable_scope("final_layer") as scope:
            output = self.conv2d(output, out_features=8, kernel_size=1)

            scope.reuse_variables()
            spatial_output = self.conv2d(
                spatial_features, out_features=8, kernel_size=1)
            spatial_pred = tf.nn.softmax(spatial_output)  # (16, 14, 14, 3)
            # spatial_pred = tf.reshape(spatial_pred, [self.batch_size, -1, 3]) # (16, 196, 3)
        # print(output.shape, spatial_pred.shape)

        logits = tf.reshape(output, [-1, self.n_classes])
        # print(features.shape, logits.shape)

        return features, logits, spatial_pred

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        # output = tf.contrib.layers.batch_norm(
        #     _input, scale=True, is_training=self.is_training,
        #     updates_collections=None)
        output = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None)(inputs=_input, training=self.is_training)

        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.keras.initializers.he_normal())
        # an initializer that generates tensors with unit variance.

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.keras.initializers.glorot_normal())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def rank_loss(self, pred1, pred2):
        # The ranking loss between two branches,
        # if prob2 > prob1 + margin, rank_loss = 0
        # if prob2 < prob1 + margin, rank_loss = prob1 - prob2 + margin
        prob1 = tf.reduce_sum(tf.multiply(
            self.labels, pred1), axis=1)  # (batch_size, 1)
        prob2 = tf.reduce_sum(tf.multiply(
            self.labels, pred2), axis=1)  # (batch_size, 1)
        rank_loss = tf.reduce_mean(tf.maximum(
            0.0, prob1 - prob2 + self.margin))  # scalar
        return rank_loss

    def compute_saliency(self, f_maps, mode="avg"):

        if mode == "avg":
            f_maps = tf.nn.relu(f_maps)
            s_map = tf.reduce_mean(f_maps, axis=-1, keepdims=True)
        elif mode == "max":
            f_maps = tf.nn.relu(f_maps)
            s_map = tf.reduce_max(f_maps, axis=-1, keepdims=True)
        elif mode == "sum_abs":
            f_maps = tf.abs(f_maps)
            s_map = tf.reduce_sum(f_maps, axis=-1, keepdims=True)

        # [batch_size, 8, 8, 1]
        s_map_min = tf.reduce_min(s_map, axis=[1, 2, 3], keepdims=True)
        s_map_max = tf.reduce_max(s_map, axis=[1, 2, 3], keepdims=True)
        s_map = tf.div(s_map - s_map_min + 1e-8, s_map_max -
                       s_map_min + 1e-8)  # [batch_size, 8, 8, 1]
        # s_map = tf.div(s_map, s_map_max)

        # s_map = tf.sigmoid(s_map)

        # used for image resampling.
        s_map = tf.image.resize_images(s_map, size=(31, 31))

        saliency_map = tf.tile(s_map, (1, 1, 1, 3))
        saliency_map = tf.image.resize_images(
            saliency_map, (128, 128))  # (8, 128, 128, 3)

        return s_map, saliency_map

    def select_samples(self, pred1, pred2, mode="net2_teach_net1"):
        """
        Select the samples that pred2 are more accurate than pred1,
        and also, pred2 make correct decision,
        then let the resampled att_maps1 of these samples be similar with att_maps2.
        """
        prob1 = tf.reduce_sum(self.labels * pred1, axis=1)
        prob2 = tf.reduce_sum(self.labels * pred2, axis=1)
        threshold = 0.5

        cond1 = tf.cast(tf.greater(
            prob2, tf.maximum(prob1, threshold)), tf.float32)
        cond2 = tf.cast(tf.equal(tf.argmax(pred2, 1),
                        tf.argmax(self.labels, 1)), tf.float32)
        cond3 = tf.cast(tf.greater(tf.argmax(self.labels, 1), 0), tf.float32)

        if mode == "net2_teach_net1":
            select = cond3
        elif mode == "net1_teach_net2":
            select = tf.multiply(tf.multiply(cond1, cond2), cond3)

        return select

    def network(self, _input):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        with tf.compat.v1.variable_scope("Initial_convolution"):
            output = self.conv2d(
                _input,
                out_features=self.first_output_features,
                kernel_size=3, strides=[1, 1, 1, 1])
            # print(output.shape)

        with tf.compat.v1.variable_scope("Initial_pooling"):
            output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[
                                    1, 2, 2, 1], padding='SAME')
            # print(output.shape)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.compat.v1.variable_scope("Block_%d" % block):
                output = self.add_block(
                    block, output, growth_rate, layers_per_block)
            # print(output.shape)

            if block != self.total_blocks - 1:
                with tf.compat.v1.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)
                    # print(output.shape)
            #
            if block == 0:
                output = self.TLFA(output, int(
                    output.shape[-1]), de=4, scope='TLFA1', trainable=True, reuse=False)

            if block == 1:
                output = self.TLFA(output, int(
                    output.shape[-1]), de=4, scope='TLFA2', trainable=True, reuse=False)
                fmaps_b2 = output

            if block == 2:
                output = self.TLFA(output, int(
                    output.shape[-1]), de=4, scope='TLFA3', trainable=True, reuse=False)
                fmaps_b3 = output

            # if block == 3:
            #     output = self.TLFA(output, int(output.shape[-1]), de=2, scope='TLFA4', trainable=True, reuse=False)

        f_maps = output

        # the last block is followed by a "transition_to_classes" layer.
        with tf.compat.v1.variable_scope("Transition_to_classes"):
            features, logits, spatial_pred = self.transition_layer_to_classes(
                f_maps)

        return f_maps, fmaps_b2, fmaps_b3, features, logits, spatial_pred

    def _build_graph(self):

        with tf.compat.v1.variable_scope("net1"):
            f_maps1, fmaps1_b2, fmaps1_b3, self.features1, logits1, spatial_pred1 = self.network(
                self.images)
            self.pred1 = tf.nn.softmax(logits1)
            self.f_maps1 = tf.nn.relu(f_maps1)

            self.s_map1, self.saliency1 = self.compute_saliency(f_maps1)
            _, self.smaps1_b2 = self.compute_saliency(fmaps1_b2)
            _, self.smaps1_b3 = self.compute_saliency(fmaps1_b3)

        self.input2 = tf.stop_gradient(get_resampled_images(
            self.high_res_images,
            self.s_map1,
            self.batch_size,
            self.src_size,
            self.dst_size,
            padding_size=30,
            lamda=self.lamda))

        # print("input2:", self.input2.shape)
        # print("images:", self.images.shape)

        self.resampled_s_map1 = get_resampled_images(
            self.s_map1, self.s_map1, self.batch_size, 31, 8, padding_size=30, lamda=self.lamda)
        # print(self.resampled_s_map1.shape)

        resampled_saliency1 = tf.tile(self.resampled_s_map1, (1, 1, 1, 3))
        self.resampled_saliency1 = tf.image.resize_images(
            resampled_saliency1, (128, 128))  # (8, 128, 128, 3)
        self.resampled_seg_map1 = generate_seg(self.resampled_saliency1)

        with tf.compat.v1.variable_scope("net2"):
            f_maps2, fmaps2_b2, fmaps2_b3, self.features2, logits2, spatial_pred2 = self.network(
                self.input2)
            self.pred2 = tf.nn.softmax(logits2)

            # Saliency of the original features of block4 and the SACA feature.
            self.s_map2, self.saliency2 = self.compute_saliency(f_maps2)
            # used for TE-loss computation.
            self.s_map2 = tf.image.resize_images(self.s_map2, [8, 8])

        ##########################################################
        # The 3-rd branch: combine the features from net1 and net2,
        # and, then make predictions of the combined features.
        ##########################################################

        total_fmaps = tf.concat(axis=3, values=(f_maps1, f_maps2))

        with tf.compat.v1.variable_scope("Sum_branch"):
            _, logits, _ = self.transition_layer_to_classes(total_fmaps)
            self.pred = tf.nn.softmax(logits)

        # weighted loss:
        class_weights = tf.constant([1, 1, 1])
        weights = tf.gather(class_weights, tf.argmax(self.labels, axis=-1))

        cross_entropy1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits1, weights=weights, label_smoothing=0.1))
        self.cross_entropy1 = cross_entropy1

        self.kd_loss1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.pred, logits1, weights=weights, label_smoothing=0.1))

        cross_entropy2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits2, weights=weights, label_smoothing=0.1))
        self.cross_entropy2 = cross_entropy2

        self.sum_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits, weights=weights, label_smoothing=0.1))

        self.grads = tf.gradients(self.cross_entropy2, self.images)[0]
        # print('gradients from loss2 to input1:', self.grads)

        # force the pred2 more accurate than pred1
        self.rank_loss_2_1 = self.rank_loss(self.pred1, self.pred2)

        rank_grad = tf.gradients(self.rank_loss_2_1, self.input2)[0]
        # print("gradients from rank_loss to input2:", rank_grad)

        self.TE_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis=[1, 2, 3])))
        # self.TE_loss = self.spatial_kl_loss(self.resampled_s_map1, self.s_map2)

        # Selective MSE loss
        self.net2_teach_net1 = self.select_samples(
            self.pred1, self.pred2, mode="net2_teach_net1")
        self.DAC_loss1 = tf.reduce_mean(self.net2_teach_net1 * tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis=[1, 2, 3])))

        self.net1_teach_net2 = self.select_samples(
            self.pred2, self.pred1, mode="net1_teach_net2")
        self.DAC_loss2 = tf.reduce_mean(self.net1_teach_net2 * tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis=[1, 2, 3])))

        self.grad1 = tf.gradients(self.sum_loss, self.images)[0]
        # print("gradients from sum loss to input1:", self.grad1)

        self.grad2 = tf.gradients(self.sum_loss, self.input2)[0]
        # print("gradients from sum loss to input2:", self.grad2)

        # regularize the variables that needs to be trained in Net1 or Net2.
        # var_list = [var for var in tf.trainable_variables()]
        var_list1 = [var for var in tf.trainable_variables(
        ) if var.name.split('/')[0] == 'net1']
        var_list2 = [var for var in tf.trainable_variables(
        ) if var.name.split('/')[0] == 'net2']
        var_list3 = [var for var in tf.trainable_variables() if var.name.split('_')[
            0] == 'Sum']

        # print ("--------------------", len(var_list))
        # for var in tf.trainable_variables():
        #     print var.name.split('/')[0]

        # l2_loss = tf.add_n([tf.nn.l2_loss(var)
        #                    for var in tf.trainable_variables()])

        l2_loss1 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1'])
        l2_loss2 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2'])
        l2_loss3 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('_')[0] == 'Sum'])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)

        self.train_step1 = optimizer.minimize(
            self.cross_entropy1 + l2_loss1 * self.weight_decay, var_list=var_list1)  # + 0.5 * self.DAC_loss1
        self.train_step2 = optimizer.minimize(
            self.cross_entropy2 + l2_loss2 * self.weight_decay, var_list=var_list2)  # + 0.5 * self.DAC_loss2

        self.train_step3 = optimizer.minimize(
            self.sum_loss + l2_loss3 * self.weight_decay, var_list=var_list3)

        correct_prediction1 = tf.equal(
            tf.argmax(self.pred1, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction1 = correct_prediction1
        self.accuracy1 = tf.reduce_mean(
            tf.cast(correct_prediction1, tf.float32))
        # self.precision1, self.recall1, self.F1_1 = self.net_measure(prediction1, self.labels)
        # self.precision1 = tf.convert_to_tensor(self.precision1)

        correct_prediction2 = tf.equal(
            tf.argmax(self.pred2, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction2 = correct_prediction2
        self.accuracy2 = tf.reduce_mean(
            tf.cast(correct_prediction2, tf.float32))
        # self.precision2, self.recall2, self.F1_2 = self.net_measure(prediction2, self.labels)

        self.correct_prediction = tf.equal(
            tf.argmax(self.pred, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        # self.batch_size = batch_size
        # reduce the lr at epoch1 and epoch2.
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        reduce_lr_epoch_3 = train_params['reduce_lr_epoch_3']
        reduce_lr_epoch_4 = train_params['reduce_lr_epoch_4']

        total_start_time = time.time()

        loss1_all_epochs = []
        loss2_all_epochs = []
        loss3_all_epochs = []

        acc1_all_epochs = []
        acc2_all_epochs = []
        acc_all_epochs = []

        """
        Only save the model with highest accuracy2
        """
        best_acc = 0.0
        data_train = "./downstream_folds/train/"
        data_val = "./downstream_folds/val/"
        self.Data_train = Data_set(batch_size=batch_size, which_split=self.which_split,
                                   shuffle=True, name='train', data_train=data_train)
        self.Data_test = Data_set(batch_size=batch_size, which_split=self.which_split,
                                  shuffle=True, name='test1', data_train=data_val)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord=coord)

        try:

            for epoch in range(1, n_epochs + 1):
                start_time = time.time()

                print("\n", '-' * 30, "Train epoch: %d" %
                      epoch, '-' * 30, '\n')
                if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2 \
                        or epoch == reduce_lr_epoch_3 or epoch == reduce_lr_epoch_4:
                    learning_rate = learning_rate / 10
                    print("Decrease learning rate, new lr = %f" %
                          learning_rate)

                print("Training...")
                start_train = time.time()
                loss1, loss2, loss3, TE_loss, rank_loss = self.train_one_epoch(
                    batch_size, learning_rate)
                train_time = time.time() - start_train
                total_train_time = int(n_epochs * train_time)
                print("Training time per epoch: %s, Total training time: %s" % (
                    str(timedelta(seconds=train_time)),
                    str(timedelta(seconds=total_train_time))))

                if self.should_save_logs:
                    self.log_loss_accuracy(
                        loss1, loss2, loss3, TE_loss, epoch, prefix='train')

                print("Rank loss:", rank_loss)

                loss1_all_epochs.append(loss1)
                loss2_all_epochs.append(loss2)
                loss3_all_epochs.append(loss3)

                if train_params.get('validation_set', False):
                    print("Validation...")
                    start_val = time.time()
                    total_pred1, total_pred2, total_pred, total_labels = self.test(
                        batch_size)
                    val_time_per_image = 1000 * \
                        (time.time() - start_val)/self.num_test
                    print("Validation time per image:", str(
                        timedelta(seconds=val_time_per_image)))

                    total_pred1 = total_pred1[0].tolist()
                    total_pred2 = total_pred2[0].tolist()
                    total_pred = total_pred[0].tolist()
                    total_labels = total_labels[0].tolist()

                    acc1 = accuracy_score(total_labels, total_pred1)
                    acc2 = accuracy_score(total_labels, total_pred2)
                    acc = accuracy_score(total_labels, total_pred)

                    # print(total_labels)
                    # print(total_pred)

                    # print(acc1, acc2, acc)

                    if self.should_save_logs:
                        self.log_loss_accuracy(
                            loss1, loss2, loss3, TE_loss, epoch, prefix='valid')

                    acc1_all_epochs.append(acc1)
                    acc2_all_epochs.append(acc2)
                    acc_all_epochs.append(acc)

                time_per_epoch = time.time() - start_time
                seconds_left = int((n_epochs - epoch) * time_per_epoch)
                print("Time per epoch: %s, Est. complete in: %s" % (
                    str(timedelta(seconds=time_per_epoch)),
                    str(timedelta(seconds=seconds_left))))

                if self.should_save_model:
                    if epoch >= 0 and epoch % 1 == 0:
                        self.save_model(global_step=epoch)

                if acc > best_acc:
                    best_acc = acc
                    self.save_model(global_step=epoch)
                    print("Save model at epoch %d" % epoch)

            total_training_time = time.time() - total_start_time
            print("\nTotal training time: %s" % str(timedelta(
                seconds=total_training_time)))

        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
            coord.join(threads)

    def train_one_epoch(self, batch_size, learning_rate):

        # train_features_path = "./feature_visualization/train_features.txt"
        # train_labels_path = "./feature_visualization/train_labels.txt"

        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_TE_loss = []
        total_rank_loss = []

        total_prob1 = []
        total_prob2 = []

        total_pred1 = []
        total_pred2 = []
        total_labels1 = []

        for i in range(self.num_train // batch_size):
            self.train_high_image_batch, self.train_low_image_batch, self.train_label_batch = self.Data_train.next_iter()

            # print("type", self.train_low_image_batch)
            high_images, low_images, labels = self.sess.run([
                self.train_high_image_batch, self.train_low_image_batch, self.train_label_batch])

            # the class_labels for features in Net1 are 0,1,2
            class_labels1 = np.argmax(labels, axis=1).astype(np.int32)
            # the class_labels for features in Net2 are 3,4,5
            class_labels2 = class_labels1 + 8

            feed_dict = {
                self.images: low_images,
                self.high_res_images: high_images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }

            fetches = [self.train_step1, self.train_step2, self.train_step3, self.features1, self.features2,
                       self.cross_entropy1, self.cross_entropy2, self.sum_loss, self.TE_loss, self.rank_loss_2_1,
                       self.accuracy1, self.accuracy2, self.pred1, self.pred2]  # , self.train_step3

            results = self.sess.run(fetches, feed_dict=feed_dict)
            _, _, _, features1, features2, loss1, loss2, loss3, TE_loss, rank_loss, acc1, acc2, pred1, pred2 = results

            features = np.vstack((features1, features2))
            class_labels = np.hstack((class_labels1, class_labels2))
            # print features.shape, class_labels.shape

            if i == 0:
                total_features = features
                total_labels = class_labels
            else:
                total_features = np.append(total_features, features, axis=0)
                total_labels = np.append(total_labels, class_labels)

            # print(pred)
            total_loss1.append(loss1)
            total_loss2.append(loss2)
            total_loss3.append(loss3)
            total_TE_loss.append(TE_loss)
            total_rank_loss.append(rank_loss)

            prob1 = np.sum(np.multiply(pred1, labels), axis=1)  # [batch_size]
            prob2 = np.sum(np.multiply(pred2, labels), axis=1)

            total_prob1.append(prob1)
            total_prob2.append(prob2)

            total_pred1.append(np.argmax(pred1, axis=1))
            total_pred2.append(np.argmax(pred2, axis=1))
            total_labels1.append(np.argmax(labels, axis=1))

            if self.should_save_logs:
                self.batches_step += 1
                # save loss and accuracy into Summary
                self.log_loss_accuracy(
                    loss1, loss2, loss3, TE_loss, self.batches_step,
                    prefix='per_batch', should_print=False)

        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)
        mean_loss3 = np.mean(total_loss3)
        mean_TE_loss = np.mean(total_TE_loss)
        mean_rank_loss = np.mean(total_rank_loss)

        return mean_loss1, mean_loss2, mean_loss3, mean_TE_loss, \
            mean_rank_loss

    def test(self, batch_size):
        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_TE_loss = []

        total_prob1 = []
        total_prob2 = []

        total_pred1 = []
        total_pred2 = []
        total_pred = []
        total_labels = []
        gt_batch = np.zeros((batch_size, 128, 128, 1))

        for i in range(self.num_test // batch_size):
            self.test_high_image_batch, self.test_low_image_batch, self.test_label_batch = self.Data_test.next_iter()
            test_high_images, test_low_images, test_labels = self.sess.run([
                self.test_high_image_batch, self.test_low_image_batch, self.test_label_batch])

            # if i > 24:
            #     for num in range(batch_size):
            #         gt_batch[num] = np.expand_dims(
            #             cv2.imread(os.path.join(dst_gt_path, str(i * batch_size + num) + '.jpg'), 0), axis=-1)

            feed_dict = {
                self.images: test_low_images,
                self.high_res_images: test_high_images,
                self.labels: test_labels,
                self.is_training: False,
                self.gt_map: gt_batch,
            }

            fetches = [self.input2,
                       self.cross_entropy1,
                       self.cross_entropy2,
                       self.sum_loss,
                       self.TE_loss,
                       self.saliency1,
                       self.saliency2,
                       self.smaps1_b2,
                       self.smaps1_b3,
                       self.pred1,
                       self.pred2,
                       self.pred]

            input2, loss1, loss2, loss3, TE_loss, s_map1, s_map2, smap1_b2, smap1_b3, pred1, pred2, pred = \
                self.sess.run(fetches, feed_dict=feed_dict)

            total_loss1.append(loss1)
            total_loss2.append(loss2)
            total_loss3.append(loss3)
            total_TE_loss.append(TE_loss)

            prob1 = np.sum(np.multiply(pred1, test_labels),
                           axis=1)  # [batch_size]
            prob2 = np.sum(np.multiply(pred2, test_labels), axis=1)

            total_prob1.append(prob1)
            total_prob2.append(prob2)

            pred1 = np.argmax(pred1, axis=1)
            pred2 = np.argmax(pred2, axis=1)
            pred = np.argmax(pred, axis=1)
            labels = np.argmax(test_labels, axis=1)

            total_pred1.append(pred1)
            total_pred2.append(pred2)
            total_pred.append(pred)
            total_labels.append(labels)

        return total_pred1, total_pred2, total_pred, total_labels