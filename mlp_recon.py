# The Auto-encoder object

# Yu Sun, CIG, WUSTL, 2019

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import scipy.io as spio
import tensorflow as tf
import logging

from NeuralNetwork.models import util
from skimage.transform import radon, iradon


class MLP(object):
    """
    data_kargs:
        nx, ny, (nz) ~ 2D/3D spatial size of the image
        ic ~ input data channel size
        oc ~ ground truth channel size

    net_kargs:
        skip_layers ~ a list of layer number to put the skip connection
        encoder_layer_num ~ number of encoding layers
        decoder_layer_num ~ number of decoding layers (each layer halves the number of neurons)
        feature_num ~ number of hidden neurons in each layer
        ffm ~ the type of Fourier feature layer
        L ~ total number of frequrencies expanded in ffm

    train_kargs:
        batch_size ~ the size of training batch
        valid_size ~ the size of valid batch
        learning_rate ~ could be a list of learning rate corresponding to differetent epoches
        epoches ~ number of epoches
        is_restore ~ True / False
        prediction_path ~ where to save predicted results. No saves if set to None. (also used to save validation)
        save_epoch ~ save model every save_epochs

    """

    def __init__(self,
                 data_kargs=None,
                 net_kargs={},
                 gpu_ratio=0.2):
        if data_kargs is None:
            data_kargs = {'ic': 2, 'oc': 1, 'num_dete': 512, 'num_proj': 90}
        tf.reset_default_graph()
        # tf.logging.set_verbosity(tf.logging.DEBUG)

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config=config)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # dictionary of key args
        self.data_kargs = data_kargs

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.data_kargs['ic']])
        self.y = tf.placeholder(tf.float32, shape=[None, self.data_kargs['oc']])
        self.lr = tf.placeholder(tf.float32)

        # config
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_ratio
        self.config.allow_soft_placement = True
        self.config.gpu_options.allow_growth = True

        # define the architecture
        self.xhat = self.net(**net_kargs)
        self.loss, self.avg_snr = self._get_measure()

    def net(self,
            ffm='loglinear',
            skip_layers=range(2, 16, 2),
            encoder_layer_num=16,
            decoder_layer_num=1,
            feature_num=256,
            L=10):

        # input layer
        in_node = self.x

        for l in range(L):
            if ffm is 'linear':
                cur_freq = tf.concat([tf.sin((l + 1) * 0.5 * np.pi * in_node),
                                      tf.cos((l + 1) * 0.5 * np.pi * in_node)], axis=-1)
            elif ffm is 'loglinear':
                cur_freq = tf.concat([tf.sin(2 ** l * np.pi * in_node),
                                      tf.cos(2 ** l * np.pi * in_node)], axis=-1)
            if l is 0:
                tot_freq = cur_freq
            else:
                tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)
        in_node = tot_freq

        with tf.variable_scope('MLP'):
            # input encoder
            for layer in range(encoder_layer_num):
                if layer in skip_layers:
                    in_node = tf.concat([in_node, tot_freq], -1)
                in_node = tf.layers.dense(in_node, feature_num, activation=tf.nn.relu)

            # output decoder
            for layer in range(decoder_layer_num):
                in_node = tf.layers.dense(in_node, feature_num // 2 ** (layer + 1), activation=None)

            # final layer
            output = tf.layers.dense(in_node, self.data_kargs['oc'], activation=None)

        return output

    def grad(self):
        return tf.gradients(self.loss, self.x)[0]

    def predict(self,
                model_path,
                x_test):

        with tf.Session(config=self.config) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            # set phase to False for every prediction
            prediction = sess.run(self.xhat, feed_dict={self.x: x_test})

        return prediction

    def save(self,
             sess,
             model_path):

        # saver = tf.train.Saver(
        #     var_list=[v for v in tf.global_variables(scope='MLP')])
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self,
                sess,
                model_path):

        # saver = tf.train.Saver(
        #     var_list=[v for v in tf.global_variables(scope='MLP')])  # tf_1.12
        saver = tf.train.Saver()  # tf_1.4
        saver.restore(sess, model_path)
        tf.logging.info("Model restored from file: %s" % model_path)

    def _record_summary(self, summary_writer, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        summary_writer.add_summary(summary, step)

    def train(self,
              output_path,
              train_provider,
              valid_provider,
              batch_size=128,
              valid_size="full",
              epochs=1000,
              learning_rate=0.001,
              is_restore=False,
              prediction_path='predict',
              save_epoch=1):

        batch_size_valid = 512
        #         initial_learning_rate = 0.01  # The initial learning rate
        #         epochs = 1000  # The total number of epochs
        #         decay_factor = 0.9  # The factor by which the learning rate should decay
        #         decay_epoch_interval = 5  # The number of epochs between each decay

        #         # Calculate the learning rate for each epoch
        #         learning_rate = np.array([initial_learning_rate * (decay_factor ** (epoch // decay_epoch_interval))
        #                                    for epoch in range(epochs)])

        abs_output_path, abs_prediction_path = self._path_checker(
            output_path, prediction_path, is_restore)

        # Set up logging to file and console
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        log_file_path = os.path.join(abs_output_path, 'training_log.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # define the optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss)

        # create output path
        directory = os.path.join(abs_output_path, "final/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, "model")
        if epochs == 0:
            tf.logging.info('Parameter [epoch] is zero. Programm terminated.')
            quit()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # tensorflow
        with tf.Session(config=config) as sess:

            # initialize the session
            sess.run(tf.global_variables_initializer())

            if is_restore:
                model = tf.train.get_checkpoint_state(abs_output_path)
                if model and model.model_checkpoint_path:
                    self.restore(sess, model.model_checkpoint_path)

            # initialize summary_writer
            summary_writer = tf.summary.FileWriter(
                abs_output_path, graph=sess.graph)
            tf.logging.info('Start Training')

            # select validation dataset (1 is dummy placeholder)
            valid_x, valid_y = valid_provider(valid_size, 1, fix=True)

            # tracking the model with the highest snr
            best = 0

            # main loop for training
            global_step = 1
            raw_iters = train_provider.file_count / batch_size
            iters_per_epoch = int(
                raw_iters) + 1 if raw_iters > int(raw_iters) else int(raw_iters)

            for epoch in range(epochs):
                logging.info(f"Starting Epoch {epoch + 1}/{epochs}")
                print(f"Starting Epoch {epoch + 1}/{epochs}...")

                train_loss_sum = 0.0
                train_snr_sum = 0.0
                valid_loss_sum = 0.0
                valid_snr_sum = 0.0

                # reshuffle the order of feeeding data
                # train_provider.reset()
                # Ensure valid_steps is properly initialized and computed

                valid_steps = max(int(np.ceil(valid_provider.file_count / batch_size)), 1)

                for iter in range(iters_per_epoch):

                    # extract training data
                    batch_x, batch_y = train_provider(batch_size, iter)

                    # learning rate
                    if type(learning_rate) is np.ndarray:
                        lr = learning_rate[epoch]
                    elif type(learning_rate) is float:
                        lr = learning_rate
                    else:
                        tf.logging.info(
                            'Learning rate should be a list of double or a double scalar.')
                        quit()

                    # run backprop
                    _, loss, avg_snr = sess.run(
                        [self.optimizer, self.loss, self.avg_snr],
                        feed_dict={self.x: batch_x, self.y: batch_y, self.lr: lr})

                    train_loss_sum += loss
                    train_snr_sum += avg_snr

                    #                     summary_writer.add_summary(train_summary, global_step)
                    global_step += 1
                    # Compute average loss and SNR for training
                    #                     avg_train_loss = train_loss_sum / iters_per_epoch
                    #                     avg_train_snr = train_snr_sum / iters_per_epoch

                    if (iter + 1) % 10000 == 0 or iter == iters_per_epoch - 1:
                        print(f"Epoch {epoch + 1}/{epochs}, Iteration {iter + 1}/{iters_per_epoch}")

                    # record global step
                    global_step = global_step + 1

                # Validation logic starts here

                valid_loss_sum = 0.0
                valid_snr_sum = 0.0
                valid_steps = max(np.ceil(valid_provider.file_count / batch_size_valid).astype(int), 1)

                # valid_steps = np.ceil(valid_provider.file_count / batch_size).astype(int)

                for step in range(valid_steps):
                    valid_x, valid_y = valid_provider(batch_size_valid, step, fix=True)
                    v_loss, v_snr = sess.run([self.loss, self.avg_snr],
                                             feed_dict={self.x: valid_x, self.y: valid_y})

                    valid_loss_sum += v_loss
                    valid_snr_sum += v_snr

                avg_valid_loss = valid_loss_sum / valid_steps
                avg_valid_snr = valid_snr_sum / valid_steps

                logger.info(
                    f"Epoch {epoch + 1}: Training - Average Loss: {loss:.4f}, Average SNR: {avg_snr:.4f}")
                print(
                    f"Epoch {epoch + 1}:  Training - Average Loss: {loss:.4f}, Average SNR: {avg_snr:.4f}")

                logger.info(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_valid_loss:.4f}, Average SNR: {avg_valid_snr:.4f}")
                print(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_valid_loss:.4f}, Average SNR: {avg_valid_snr:.4f}")

                if avg_valid_snr >= best:
                    best = avg_valid_snr
                    self.save(sess, save_path)

                    self._record_summary(
                        summary_writer, 'best_snr', best, epoch + 1)

                # save model
                if (epoch + 1) % save_epoch == 0:
                    directory = os.path.join(
                        abs_output_path, "{}_model/".format(epoch + 1))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, "model")
                    self.save(sess, path)

            #                 validation_summary = tf.Summary(value=[
            #                     tf.Summary.Value(tag="validation_loss", simple_value=avg_valid_loss),
            #                     tf.Summary.Value(tag="validation_snr", simple_value=avg_valid_snr)
            #                 ])
            #                 summary_writer.add_summary(validation_summary, epoch)
            #                 summary_writer.flush()

            #             summary_writer.close()
            tf.logging.info('Training Ends')
            if 'sess' in globals() and sess:
                sess.close()
                print('Open TensorFlow session has been closed.')

    ###### Private Functions ######

    def _get_measure(self):

        # define the loss
        grad = tf.gradients(self.xhat, self.x)[0]
        loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.xhat - self.y),1))

        # compute average SNR
        ratio = tf.reduce_sum(tf.square(self.xhat)) / \
            tf.reduce_sum(tf.square(self.xhat - self.y))
        avg_snr = 10*self._log(ratio, 10)
        return loss, avg_snr
    def _output_valstats(self,
                         sess,
                         summary_writer,
                         step,
                         batch_x,
                         batch_y,
                         name,
                         save_path):

        xhat, loss, avg_snr = sess.run([self.xhat, self.loss, self.avg_snr],
                                       feed_dict={self.x: batch_x,
                                                  self.y: batch_y})

        self._record_summary(
            summary_writer, 'validation_loss', loss, step)
        self._record_summary(
            summary_writer, 'validation_snr', avg_snr, step)

        tf.logging.info(
            "Validation Statistics, Validation Loss= {:.4f}, Validation SNR= {:.4f}".format(loss, avg_snr))
        return avg_snr

    @staticmethod
    def _log(x,
             base):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
        return numerator / denominator

    @staticmethod
    def _path_checker(output_path,
                      prediction_path,
                      is_restore):
        abs_prediction_path = os.path.abspath(prediction_path)
        abs_output_path = os.path.abspath(output_path)

        if not is_restore:
            tf.logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            tf.logging.info("Removing '{:}'".format(abs_output_path))
            shutil.rmtree(abs_output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            tf.logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(abs_output_path):
            tf.logging.info("Allocating '{:}'".format(abs_output_path))
            os.makedirs(abs_output_path)

        return abs_output_path, abs_prediction_path

    @staticmethod
    def _record_summary(writer,
                        name,
                        value,
                        step):

        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()