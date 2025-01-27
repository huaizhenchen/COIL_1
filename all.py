import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
from skimage.metrics import structural_similarity as ssim
import scipy.io as spio
import shutil
import logging
from NeuralNetwork import Provider
def process_mat_file_normalized(file_path):
    data = spio.loadmat(file_path, squeeze_me=True)
    LEDCood_f = data['LEDCood_f']  # Assume the parameter vector is named LEDCood_f
    images = data['I_Raw']  # Assume image data is stored in the images field

    # Initialize an empty list to store processed data
    processed_data = []

    # Find the min and max for each parameter in LEDCood_f across all images
    min_val = np.min(LEDCood_f, axis=0)
    max_val = np.max(LEDCood_f, axis=0)

    # Normalize LEDCood_f and x, y coordinates to range [0, 1]
    normalized_LEDCood_f = (LEDCood_f - min_val) / (max_val - min_val)

    # Print the shape of images to verify dimensions
    print(images.shape[0], images.shape[1], images.shape[2])
    for z in range(images.shape[2]):
        print("z: ", z)
        for x in range(images.shape[0]):
            for y in range(images.shape[1]):
                normalized_x = x / (images.shape[0] - 1)  # Normalizing to [0, 1]
                normalized_y = y / (images.shape[1] - 1)  # Normalizing to [0, 1]
                input_vector = np.append(normalized_LEDCood_f[z], [normalized_x, normalized_y])
                pixel_value = images[x, y, z]
                processed_data.append(np.append(input_vector, pixel_value))

    # Convert the processed data into a numpy array
    processed_data = np.array(processed_data, dtype=np.float32)
    np.save('data/processed_data.npy', processed_data)
    return processed_data

def split_data_gap(processed_data, num_total_imgs=120, img_size=1200*1200):
    """
    Split the processed data into training, validation, and test sets with custom intervals.

    Parameters:
    - processed_data: The processed dataset, assumed to be a flat array where each image is a block.
    - num_total_imgs: Total number of images in the dataset.
    - img_size: Size of each image in the dataset, assumed to be a flat block.

    Returns:
    - train_data: Data from images chosen for training.
    - val_data: Data from images chosen for validation.
    - test_data: Data from images chosen for testing.
    """

    # Create indices for each set
    indices = np.arange(num_total_imgs)
    np.random.shuffle(indices)
    train_indices = indices[::2]  # Even indices for training (0, 2, 4, ...)
    remaining_indices = indices[1::2]  # Odd indices remaining (1, 3, 5, ...)
    
    # Split the remaining indices into validation and test sets
    val_indices = remaining_indices[::2]  # Alternate starting from first remaining index (1, 5, 9, ...)
    test_indices = remaining_indices[1::2]  # Alternate starting from second remaining index (3, 7, 11, ...)

    # Extract data based on calculated indices
    train_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in train_indices])
    val_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in val_indices])
    test_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in test_indices])

    # Optionally save data to files
    np.save("data/test_data.npy", test_data)
    np.save("data/train_data.npy", train_data)
    np.save("data/val_data.npy", val_data)

    return train_data, val_data
class MLP(object):
    def __init__(self,
                 data_kargs={'ic': 2, 'oc': 1},
                 net_kargs={},
                 gpu_ratio=0.2):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # dictionary of key args
        self.data_kargs = data_kargs

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.data_kargs['ic']])
        self.y = tf.placeholder(tf.float32, shape=[None, self.data_kargs['oc']])
        self.lr = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        # config
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_ratio

        # define the architecture
        self.xhat = self.net(**net_kargs)
        self.loss, self.avg_snr = self._get_measure()

        # add L2 regularization
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss += 0.001 * l2_loss  # Adjust the regularization factor as needed

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
            if ffm == 'linear':
                cur_freq = tf.concat([tf.sin((l + 1) * 0.5 * np.pi * in_node),
                                      tf.cos((l + 1) * 0.5 * np.pi * in_node)], axis=-1)
            elif ffm == 'loglinear':
                cur_freq = tf.concat([tf.sin(2 ** l * np.pi * in_node),
                                      tf.cos(2 ** l * np.pi * in_node)], axis=-1)
            if l == 0:
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
                in_node = tf.layers.dropout(in_node, rate=0.5, training=self.is_training)  # Add Dropout layer

            # output decoder
            for layer in range(decoder_layer_num):
                in_node = tf.layers.dense(in_node, feature_num // 2 ** (layer + 1), activation=None)

            # final layer
            output = tf.layers.dense(in_node, self.data_kargs['oc'], activation=None)

        return output

    def augment_data(self, batch_x):
        # Data augmentation by adding small noise to the parameter vectors and coordinates
        noise = tf.random_normal(tf.shape(batch_x), mean=0.0, stddev=0.01)
        augmented_batch_x = batch_x + noise
        return augmented_batch_x

    def _get_measure(self):
        # define the loss
        grad = tf.gradients(self.xhat, self.x)[0]
        loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.xhat - self.y), 1))

        # compute average SNR using TensorFlow operations
        signal_power = tf.reduce_mean(tf.square(self.y))
        noise_power = tf.reduce_mean(tf.square(self.xhat - self.y))
        ratio = signal_power / noise_power
        avg_snr = 10 * self._log(ratio, 10)

        return loss, avg_snr

    @staticmethod
    def _log(x, base):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
        return numerator / denominator

    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def train(self,
              output_path,
              train_provider,
              valid_provider,
              batch_size=128,
              valid_size="full",
              epochs=1000,
              initial_learning_rate=0.001,
              is_restore=False,
              prediction_path='predict',
              save_epoch=1):

        batch_size_valid = 256
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

        # Define the optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss)

        # Create output path
        directory = os.path.join(abs_output_path, "final/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, "model")
        if epochs == 0:
            tf.logging.info('Parameter [epoch] is zero. Program terminated.')
            quit()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # Initialize the session
            sess.run(tf.global_variables_initializer())

            if is_restore:
                model = tf.train.get_checkpoint_state(abs_output_path)
                if model and model.model_checkpoint_path:
                    self.restore(sess, model.model_checkpoint_path)

            # Initialize summary_writer
            summary_writer = tf.summary.FileWriter(
                abs_output_path, graph=sess.graph)
            tf.logging.info('Start Training')

            # Select validation dataset (1 is dummy placeholder)
            valid_x, valid_y = valid_provider(valid_size, 1, fix=True)

            # Tracking the model with the highest snr
            best_val_loss = float('inf')

            # Main loop for training
            global_step = 1
            raw_iters = train_provider.file_count / batch_size
            iters_per_epoch = int(
                raw_iters) + 1 if raw_iters > int(raw_iters) else int(raw_iters)

            learning_rate = initial_learning_rate
            patience = 5  # Number of epochs to wait for improvement before reducing learning rate
            wait = 0

            for epoch in range(epochs):
                logging.info(f"Starting Epoch {epoch + 1}/{epochs}")
                print(f"Starting Epoch {epoch + 1}/{epochs}...")

                train_loss_sum = 0.0
                train_snr_sum = 0.0

                # Reshuffle the order of feeding data
                train_provider.reset()

                for iter in range(iters_per_epoch):

                    # Extract training data
                    batch_x, batch_y = train_provider(batch_size, iter)

                    # Augment data
                    augmented_batch_x = self.augment_data(batch_x)

                    # Run backpropagation
                    _, loss, avg_snr = sess.run([self.optimizer, self.loss, self.avg_snr],
                                                feed_dict={self.x: augmented_batch_x,
                                                           self.y: batch_y,
                                                           self.lr: learning_rate,
                                                           self.is_training: True})

                    train_loss_sum += loss
                    train_snr_sum += avg_snr

                    if (iter + 1) % 1000 == 0 or iter == iters_per_epoch - 1:
                        print(f"Epoch {epoch + 1}/{epochs}, Iteration {iter + 1}/{iters_per_epoch}")

                    # Record diagnosis data
                    self._record_summary(
                        summary_writer, 'training_loss', loss, global_step)
                    self._record_summary(
                        summary_writer, 'training_snr', avg_snr, global_step)

                    # Record global step
                    global_step += 1

                # Output statistics for epoch
                avg_epoch_loss = train_loss_sum / iters_per_epoch
                avg_epoch_snr = train_snr_sum / iters_per_epoch
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Average SNR: {avg_epoch_snr:.4f}")

                # Validation logic starts here

                # Save the current model
                self.save(sess, save_path)

                # Load the saved model
                saver = tf.train.Saver()
                saver.restore(sess, save_path)

                total_mse = 0
                total_ssim = 0
                total_snr = 0
                valid_steps = 30
                print(valid_steps)
                for step in range(30):
                    valid_x, valid_y = valid_provider(batch_size_valid, step, fix=True)

                    # Run inference
                    predictions = []

                    for j in range(0, valid_x.shape[0], batch_size):
                        batch_data = valid_x[j:j + batch_size]
                        batch_pred = sess.run(self.xhat, feed_dict={self.x: batch_data,
                                                                    self.is_training: False})
                        predictions.append(batch_pred)

                    predictions = np.concatenate(predictions).reshape(valid_y.shape)

                    # Calculate metrics
                    mse_value = np.mean((predictions - valid_y) ** 2)
                    signal_power = np.mean(valid_y ** 2)
                    noise_power = np.mean((predictions - valid_y) ** 2)
                    snr_value = 10 * np.log10(signal_power / noise_power)

                    # If the image is multi-channel, set multichannel=True
                    if predictions.ndim == 3 and predictions.shape[-1] > 1:
                        ssim_value = ssim(predictions, valid_y, data_range=valid_y.max() - valid_y.min(), multichannel=True)
                    else:
                        ssim_value = ssim(predictions, valid_y, data_range=valid_y.max() - valid_y.min(), multichannel=True)

                    total_mse += mse_value
                    total_snr += snr_value
                    total_ssim += ssim_value

                # Calculate averages
                avg_valid_loss = total_mse / 30
                avg_valid_snr = self._output_valstats(
                                    sess, summary_writer, epoch, valid_x, valid_y, 
                                    "epoch_{}.mat".format(epoch+1), abs_prediction_path)
                avg_valid_ssim = total_ssim / 30

                logger.info(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_valid_loss:.4f}, Average SNR: {avg_valid_snr:.4f}, Average SSIM: {avg_valid_ssim:.4f}")
                print(
                    f"Epoch {epoch + 1}: Validation - Average Loss: {avg_valid_loss:.4f}, Average SNR: {avg_valid_snr:.4f}, Average SSIM: {avg_valid_ssim:.4f}")

                # Save the best model based on validation SNR
                if avg_valid_loss < best_val_loss:
                    best_val_loss = avg_valid_loss
                    self.save(sess, save_path)
                    self._record_summary(summary_writer, 'best_loss', best_val_loss, epoch + 1)
                    wait = 0  # Reset wait counter if validation loss improves
                else:
                    wait += 1  # Increment wait counter if no improvement

                # Adjust learning rate if no improvement in validation loss for a number of epochs
                if wait >= patience:
                    learning_rate *= 0.5
                    print(f"Reducing learning rate to {learning_rate}")
                    logger.info(f"Reducing learning rate to {learning_rate}")
                    wait = 0  # Reset wait counter after reducing learning rate

                # Save model periodically
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
                                                  self.y: batch_y,
                                                  self.is_training: False})

        self._record_summary(
            summary_writer, 'validation_loss', loss, step)
        self._record_summary(
            summary_writer, 'validation_snr', avg_snr, step)

        tf.logging.info(
            "Validation Statistics, Validation Loss= {:.4f}, Validation SNR= {:.4f}".format(loss, avg_snr))
        return avg_snr

    @staticmethod
    def _log(x, base):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
        return numerator / denominator

    @staticmethod
    def _path_checker(output_path, prediction_path, is_restore):
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
    def _record_summary(writer, name, value, step):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()
data_kargs = {
    'ic': 4,  # Input channel size is 4 (original parameter vector plus pixel coordinates)
    'oc': 1  # Output channel size is 1 (predicted pixel value)
}

net_kargs = {
    'skip_layers': range(2, 16, 2),
    'encoder_layer_num': 24,
    'decoder_layer_num': 1,
    'feature_num': 1024,
    'ffm': 'linear',
    'L': 10
}
if __name__ == "__main__":

    # Data file path
    data_file_path = 'data/IRaw_CElegans_Shw.mat'

    processed_data_path = 'data/processed_data.npy'

    if os.path.exists(processed_data_path):
        print("data loading")
        processed_data = np.load(processed_data_path)
    else:
        print("data processing")
        processed_data = process_mat_file_normalized(data_file_path)
        # save_processed_data(processed_data, processed_data_path)

    print("data loaded")

    # Split the data into training and test sets
    train_data, val_data = split_data_gap(processed_data)

    print("data splitted")

    # Initialize data providers
    train_provider = Provider.StrictEpochProvider(train_data[:, :4], train_data[:, 4:], is_shuffle=False)
    valid_provider = Provider.StrictEpochProvider(val_data[:, :4], val_data[:, 4:], is_shuffle=False)

    print("data provided")

    # Network initialization parameters
    data_kargs = {'ic': 4, 'oc': 1}
    train_kargs = {
        'batch_size': 1024,
        'valid_size': 'full',
        'epochs': 1000,
        'initial_learning_rate': 0.0001,
        'is_restore': False,
        'prediction_path': 'data/predictions',
        'save_epoch': 1}

    # Initialize the network
    net = MLP(data_kargs=data_kargs, net_kargs=net_kargs)

    # Train the network
    output_path = 'save/trial/'
    prediction_path = 'save/trial/'
    net.train(output_path, train_provider, valid_provider, **train_kargs)

