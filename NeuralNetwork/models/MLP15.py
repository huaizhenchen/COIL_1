from __future__ import print_function, division, absolute_import, unicode_literals

import logging
import os
import shutil
import numpy as np
import scipy.io as spio
import tensorflow as tf
import pywt
from NeuralNetwork.models import util
from skimage.transform import radon, iradon
from tensorflow.python.keras import regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import pywt

import tensorflow as tf
import numpy as np

import tensorflow as tf
def factorial_tf(n):
    """Compute factorial using TensorFlow for GPU acceleration."""
    return tf.math.reduce_prod(tf.range(1, n + 1, dtype=tf.float32))

def zernike_tf1(m, n, rho, theta):

    # Initialize R
    R = tf.zeros_like(rho, dtype=tf.float32)

    # Calculate Zernike radial polynomial
    for k in range((n - abs(m)) // 2 + 1):
        R += ((-1) ** k * factorial_tf(n - k)) / (
            factorial_tf(k) * factorial_tf((n + abs(m)) // 2 - k) * factorial_tf((n - abs(m)) // 2 - k)
        ) * tf.pow(rho, n - 2 * k)

    # Zernike polynomial (real part)
    result = R * tf.cos(m * theta)
    return result
def zernike_tf2(m, n, rho, theta):

    # Initialize R
    R = tf.zeros_like(rho, dtype=tf.float32)

    # Calculate Zernike radial polynomial
    for k in range((n - abs(m)) // 2 + 1):
        R += ((-1) ** k * factorial_tf(n - k)) / (
            factorial_tf(k) * factorial_tf((n + abs(m)) // 2 - k) * factorial_tf((n - abs(m)) // 2 - k)
        ) * tf.pow(rho, n - 2 * k)

    # Zernike polynomial (real part)
    result = R * tf.sin(m * theta)
    return result

def apply_zernike_transform_tf(data, zernike_degree=10):
    """
    Apply Zernike transform to both the first two and last two dimensions
    using TensorFlow for GPU acceleration.
    """
    
    def zernike_transform_single(data_point):
        # First two dimensions
        rho1 = tf.sqrt(data_point[0]**2 + data_point[1]**2)
        theta1 = tf.atan2(data_point[1], data_point[0])
        zernike_coeffs1 = []

        # Apply Zernike polynomial to the first two dimensions
        for n in range(zernike_degree + 1):
            for m in range(-n, n + 1, 2):
                if n >= abs(m) and (n - m) % 2 == 0:
                    zernike_coeffs1.append(zernike_tf1(m, n, rho1, theta1))
                    zernike_coeffs1.append(zernike_tf2(m, n, rho1, theta1))

        # Last two dimensions
        rho2 = tf.sqrt(data_point[2]**2 + data_point[3]**2)
        theta2 = tf.atan2(data_point[3], data_point[2])
        zernike_coeffs2 = []

        # Apply Zernike polynomial to the last two dimensions
        for n in range(zernike_degree + 1):
            for m in range(-n, n + 1, 2):
                if n >= abs(m) and (n - m) % 2 == 0:
                    zernike_coeffs2.append(zernike_tf1(m, n, rho2, theta2))
                    zernike_coeffs2.append(zernike_tf2(m, n, rho2, theta2))

        # Concatenate the Zernike coefficients for both parts
        return tf.concat([tf.stack(zernike_coeffs1), tf.stack(zernike_coeffs2)], axis=-1)
    
    # Use tf.map_fn to apply the transformation to each row of the input tensor
    return tf.map_fn(lambda x: zernike_transform_single(x), data)
# Coefficients for db8 wavelet transform
db8_low = [
    0.2303778133088964, 0.7148465705529154, 0.6308807679298587, -0.0279837694168599,
    -0.1870348117190931, 0.0308413818359869, 0.0328830116666778, -0.0105974017850690
]

db8_high = [
    -0.0105974017850690, -0.0328830116666778, 0.0308413818359869, 0.1870348117190931,
    -0.0279837694168599, -0.6308807679298587, 0.7148465705529154, -0.2303778133088964
]

# Scaling function (low-pass filter) for db8
def scaling_function(x):
    coeffs = tf.constant(db8_low, dtype=tf.float32)
    filter_size = len(db8_low)
    
    # Reshape 2D input [batch_size, features] to 3D [batch_size, features, 1] to add channel dimension
    x = tf.expand_dims(x, axis=-1)
    
    # Apply convolution with 'same' padding to maintain the original size
    return tf.nn.conv1d(x, tf.reshape(coeffs, [filter_size, 1, 1]), stride=1, padding='SAME')

# Wavelet function (high-pass filter) for db8
def wavelet_function(x):
    coeffs = tf.constant(db8_high, dtype=tf.float32)
    filter_size = len(db8_high)
    
    # Reshape 2D input [batch_size, features] to 3D [batch_size, features, 1] to add channel dimension
    x = tf.expand_dims(x, axis=-1)
    
    # Apply convolution with 'same' padding to maintain the original size
    return tf.nn.conv1d(x, tf.reshape(coeffs, [filter_size, 1, 1]), stride=1, padding='SAME')

# Applying db8 wavelet transform
def apply_manual_wavelet_transform(data, L):
    wavelet_transformed = []
    current_data = data
    
    for l in range(L):
        low_pass = scaling_function(current_data)  # Low-pass filtering (scaling coefficients)
        high_pass = wavelet_function(current_data)  # High-pass filtering (wavelet coefficients)
        
        # Remove extra dimensions added by conv1d
        low_pass = tf.squeeze(low_pass, axis=-1)
        high_pass = tf.squeeze(high_pass, axis=-1)
        
        # Concatenating low-pass and high-pass results for the next level
        current_data = low_pass  # For next iteration, work on low-pass filtered signal
        wavelet_transformed.append(tf.concat([low_pass, high_pass], axis=-1))
    
    return tf.concat(wavelet_transformed, axis=-1)

def positional_encoding(in_node, num_encoding_functions=10):
    """
    使用位置编码对输入进行编码。

    参数：
    - in_node: 输入张量，形状为 (..., D)，其中 D 是输入维度。
    - num_encoding_functions: 使用的编码函数数量。

    返回：
    - 编码后的张量，形状为 (..., D * num_encoding_functions * 2)。
    """
    D = tf.cast(in_node.shape[-1], tf.float32)  # 将 D 转换为 tf.float32
    batch_shape = tf.shape(in_node)[:-1]

    # 生成频率列表
    i = tf.range(num_encoding_functions, dtype=tf.float32)
    
    # 将 10000 显式转换为 float32
    angle_rates = 1 / tf.pow(tf.cast(10000, tf.float32), (2 * i) / D)

    # 计算正弦和余弦编码
    angle_rads = in_node[..., tf.newaxis] * angle_rates
    sin_encoding = tf.sin(angle_rads)
    cos_encoding = tf.cos(angle_rads)

    # 拼接编码
    pos_encoding = tf.concat([sin_encoding, cos_encoding], axis=-1)

    # 重塑张量
    new_shape = tf.concat([batch_shape, [-1]], axis=0)
    pos_encoding = tf.reshape(pos_encoding, new_shape)
    
    return pos_encoding


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
             data_kargs={'ic': 2, 'oc': 1, 'num_dete': 512, 'num_proj': 90},
             net_kargs={},
             gpu_ratio=0.2,
             weight_path='data/all1.mat'):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # dictionary of key args
        self.data_kargs = data_kargs

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.data_kargs['ic']])
        self.y = tf.placeholder(tf.float32, shape=[None, self.data_kargs['oc']])
        self.lr = tf.placeholder(tf.float32)

        # Load the weight matrix and split it into 100 [3600, 1] blocks
        full_weight = spio.loadmat(weight_path)['a']
        num_blocks = 9
        block_size = 1296

        # Ensure the weight matrix is (600, 600) and flatten it into a 1D array of shape (360000,)
        reshaped_weight = full_weight.reshape(-1)

        # Split the flattened weight matrix into 100 blocks of shape [3600, 1]
        self.weight_blocks = [
            reshaped_weight[i * block_size: (i + 1) * block_size].reshape(block_size, 1)
            for i in range(num_blocks)
        ]

        # Horizontally flip the weight matrix and flatten it
        flipped_weight = np.fliplr(full_weight).reshape(-1)

        # Split the flipped weight matrix into 100 blocks of shape [3600, 1]
        flipped_weight_blocks = [
            flipped_weight[i * block_size: (i + 1) * block_size].reshape(block_size, 1)
            for i in range(num_blocks)
        ]

        # Append the flipped blocks to the original blocks
        self.weight_blocks.extend(flipped_weight_blocks)

        # config
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_ratio

        # define the architecture
        self.xhat = self.net(**net_kargs)
        self.current_weight_ph = tf.placeholder(tf.float32, shape=[1296, 1])
        self.loss, self.avg_snr = self._get_measure(self.current_weight_ph)



    def net(self,
            ffm='positional_encoding',  # 默认编码方式
            skip_layers=range(2, 16, 2),
            encoder_layer_num=16,
            decoder_layer_num=1,
            feature_num=256,
            L=10,  # 对于位置编码，L 是编码函数的数量
            l2_reg=0.01,
            dropout_rate=0.01,
            dia_degree=45):  # 添加用于 exp_diag 的角度

        # 输入层
        in_node = self.x

        if ffm == 'wavelet':
            # 自定义小波变换
            in_node = apply_manual_wavelet_transform(in_node, L)
            tot_freq = in_node
        if ffm == 'zernike':
            # Apply Zernike transform to the full 4D input using TensorFlow function
            in_node = apply_zernike_transform_tf(in_node, zernike_degree=L)
            tot_freq = in_node  # Store transformed frequencies for later use
        elif ffm == 'positional_encoding':
            # 位置编码的实现，使用 sin 和 cos 函数
            position_dim = tf.shape(in_node)[-1]  # 获取输入的维度
            batch_size = tf.shape(in_node)[0]

            # 初始化存储总的频率特征
            tot_freq = None

            for l in range(L):  # L 决定生成多少层频率编码
                # 计算缩放因子
                scaling_factor = 1 / tf.pow(10000.0, (2 * l) / tf.cast(position_dim, tf.float32))

                # 生成 sin 和 cos 编码
                pos_encoding_sin = tf.sin(in_node * scaling_factor)
                pos_encoding_cos = tf.cos(in_node * scaling_factor)

                # 合并 sin 和 cos 的编码
                cur_freq = tf.concat([pos_encoding_sin, pos_encoding_cos], axis=-1)

                if tot_freq is None:
                    tot_freq = cur_freq
                else:
                    tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)

            # 更新 in_node，使其包含所有的频率编码
            in_node = tot_freq

        elif ffm == 'exp_diag':  # 实现 exp_diag 编码
            # 对角线正弦编码实现，使用旋转矩阵
            angles = np.arange(0, 180, dia_degree) * np.pi / 180
            s = np.sin(angles)[:, np.newaxis]
            c = np.cos(angles)[:, np.newaxis]
            fourier_mapping = np.concatenate((s, c), axis=1).T  # 生成旋转矩阵
            fourier_mapping = tf.cast(fourier_mapping, dtype=tf.float32)
            # 对四维坐标的前两维（即 x 和 y 坐标）进行编码
            xy_freq = tf.matmul(in_node[:, :2], fourier_mapping)
            lxly_freq=tf.matmul(in_node[:, 2:], fourier_mapping)
            tot_freq = None
            for l in range(L):
                # 对 xy 坐标应用不同的频率生成 sin 和 cos 编码
                cur_freq = tf.concat(
                    [
                        tf.sin(2 ** l * np.pi * xy_freq),
                        tf.cos(2 ** l * np.pi * xy_freq),
                    ],
                    axis=-1,
                )

                if tot_freq is None:
                    tot_freq = cur_freq
                else:
                    tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)
            for l in range(L):
                # 对 xy 坐标应用不同的频率生成 sin 和 cos 编码
                cur_freq = tf.concat(
                    [
                        tf.sin(2 ** l * np.pi * lxly_freq),
                        tf.cos(2 ** l * np.pi * lxly_freq),
                    ],
                    axis=-1,
                )

                tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)            
            # 将所有的频率特征加入到输入节点
            in_node = tot_freq

        else:
            # 处理其他特征编码机制
            for l in range(L):
                if ffm == 'linear':
                    scaling_factor = 1 / 2**l
                    cur_freq = tf.concat([tf.sin(scaling_factor * (l + 1) * 0.5 * np.pi * in_node),
                                          tf.cos(scaling_factor * (l + 1) * 0.5 * np.pi * in_node)], axis=-1)

                elif ffm == 'loglinear':
                    cur_freq = tf.concat([tf.sin(2 ** l * np.pi * in_node),
                                          tf.cos(2 ** l * np.pi * in_node)], axis=-1)

                if l == 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)

            in_node = tot_freq

        # 构建 MLP 网络
        with tf.variable_scope('MLP'):
            # 输入编码器
            for layer in range(encoder_layer_num):
                if layer in skip_layers:
                    in_node = tf.concat([in_node, tot_freq], -1)
                input_before_layer = in_node
                in_node = tf.layers.dense(in_node, feature_num, activation=tf.nn.relu,
                                          kernel_regularizer=regularizers.l2(l2_reg))
                if layer not in skip_layers and layer > 0:
                    in_node += input_before_layer  # 残差连接
                in_node = tf.layers.dropout(in_node, rate=dropout_rate)

            # 输出解码器
            for layer in range(decoder_layer_num):
                in_node = tf.layers.dense(in_node, feature_num // 2 ** (layer + 1), activation=None,
                                          kernel_regularizer=regularizers.l2(l2_reg))

            # 最后一层
            output = tf.layers.dense(in_node, self.data_kargs['oc'], activation=None,
                                     kernel_regularizer=regularizers.l2(l2_reg))

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
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self,
                sess,
                model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        tf.logging.info("Model restored from file: %s" % model_path)

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

        batch_size_valid = 1296
        abs_output_path, abs_prediction_path = self._path_checker(
            output_path, prediction_path, is_restore)

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

        directory = os.path.join(abs_output_path, "final/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, "model")
        if epochs == 0:
            tf.logging.info('Parameter [epoch] is zero. Programm terminated.')
            quit()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # 计算每个 epoch 的迭代次数
            raw_iters = train_provider.file_count / batch_size
            iters_per_epoch = int(raw_iters) + 1 if raw_iters > int(raw_iters) else int(raw_iters)

            # 定义优化器，基于 `self.loss`
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # 初始化所有变量（包括优化器变量）
            sess.run(tf.global_variables_initializer())

            if is_restore:
                model = tf.train.get_checkpoint_state(abs_output_path)
                if model and model.model_checkpoint_path:
                    self.restore(sess, model.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(
                abs_output_path, graph=sess.graph)
            tf.logging.info('Start Training')

            valid_x, valid_y = valid_provider(valid_size, 1)

            best = 0
            global_step = 1

            for epoch in range(epochs):
                logging.info(f"Starting Epoch {epoch + 1}/{epochs}")
                print(f"Starting Epoch {epoch + 1}/{epochs}...")

                train_loss_sum = 0.0
                train_snr_sum = 0.0

                train_provider.reset()

                for iter in range(iters_per_epoch):
                    batch_x, batch_y = train_provider(batch_size, iter)
                    current_weight = self.weight_blocks[iter % len(self.weight_blocks)]

                    if type(learning_rate) is np.ndarray:
                        lr = learning_rate[epoch]
                    elif type(learning_rate) is float:
                        lr = learning_rate
                    else:
                        tf.logging.info(
                            'Learning rate should be a list of double or a double scalar.')
                        quit()

                    # 运行反向传播，并将动态权重传入计算图
                    _, loss, avg_snr = sess.run([optimizer, self.loss, self.avg_snr],
                                                feed_dict={self.x: batch_x,
                                                           self.y: batch_y,
                                                           self.lr: lr,
                                                           self.current_weight_ph: current_weight})

                    train_loss_sum += loss
                    train_snr_sum += avg_snr

                    if (iter + 1) % 1000 == 0 or iter == iters_per_epoch - 1:
                        print(f"Epoch {epoch + 1}/{epochs}, Iteration {iter + 1}/{iters_per_epoch}")

                    self._record_summary(
                        summary_writer, 'training_loss', loss, global_step)
                    self._record_summary(
                        summary_writer, 'training_snr', avg_snr, global_step)

                    global_step += 1

                avg_epoch_loss = train_loss_sum / iters_per_epoch
                avg_epoch_snr = train_snr_sum / iters_per_epoch
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Average SNR: {avg_epoch_snr:.4f}")
                print(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_epoch_loss:.4f}, Average SNR: {avg_epoch_snr:.4f}")
                # Validation logic
                valid_loss_sum = 0.0
                valid_snr_sum = 0.0
                valid_steps = max(np.ceil(valid_provider.file_count / batch_size_valid).astype(int), 1)

                for step in range(valid_steps):
                    valid_x, valid_y = valid_provider(batch_size_valid, step)
                    v_loss, v_snr = sess.run([self.loss, self.avg_snr],
                                             feed_dict={self.x: valid_x, self.y: valid_y, self.current_weight_ph: current_weight})
                    valid_loss_sum += v_loss
                    valid_snr_sum += v_snr

                avg_valid_loss = valid_loss_sum / valid_steps
                avg_valid_snr = valid_snr_sum / valid_steps

                logger.info(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_valid_loss:.8f}, Average SNR: {avg_valid_snr:.4f}")
                print(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_valid_loss:.8f}, Average SNR: {avg_valid_snr:.4f}")

                if avg_valid_snr >= best:
                    best = avg_valid_snr
                    self.save(sess, save_path)

                if (epoch + 1) % save_epoch == 0:
                    directory = os.path.join(
                        abs_output_path, "{}_model/".format(epoch + 1))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, "model")
                    self.save(sess, path)

            tf.logging.info('Training Ends')
            if 'sess' in globals() and sess:
                sess.close()
                print('Open TensorFlow session has been closed.')


#     def _get_measure(self, current_weight_ph ):
#         # 将权重矩阵应用于MSE计算
#         weighted_mse_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(current_weight_ph  * tf.square(self.xhat - self.y), 1))
#         loss = weighted_mse_loss
#         ratio = tf.reduce_sum(tf.square(self.y)) / tf.reduce_sum(tf.square(self.xhat - self.y))
#         avg_snr = 10 * self._log(ratio, 10)
#         return loss, avg_snr
    def _get_measure(self, current_weight_ph):
        # 将权重矩阵应用于 MAE 计算
        weighted_mae_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(current_weight_ph * tf.abs(self.xhat - self.y), 1))
        loss = weighted_mae_loss
        ratio = tf.reduce_sum(tf.square(self.y)) / tf.reduce_sum(tf.square(self.xhat - self.y))
        avg_snr = 10 * self._log(ratio, 10)
        return loss, avg_snr

#     def _get_measure(self, current_weight_ph):
#         # 加权 MSE 损失
#         weighted_mse_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(current_weight_ph * tf.square(self.xhat - self.y), 1))

#         # 计算一维梯度
#         gradient_xhat = tf.abs(self.xhat[:, 1:] - self.xhat[:, :-1])  # 计算预测值的梯度
#         gradient_y = tf.abs(self.y[:, 1:] - self.y[:, :-1])  # 计算真实值的梯度

#         # 计算梯度差异作为边缘感知损失
#         gradient_loss = tf.reduce_mean(tf.square(gradient_xhat - gradient_y))

#         # 总损失 = MSE 损失 + 梯度损失
#         loss = weighted_mse_loss + 0.1 * gradient_loss

#         # 计算信噪比 (SNR)
#         ratio = tf.reduce_sum(tf.square(self.y)) / tf.reduce_sum(tf.square(self.xhat - self.y))
#         avg_snr = 10 * self._log(ratio, 10)

#         return loss, avg_snr






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
