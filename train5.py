import os
import pywt
import numpy as np
import tensorflow as tf
from NeuralNetwork.models.MLP15 import MLP
from NeuralNetwork import Provider as Provider
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import scipy.io as spio
from scipy.stats import pearsonr

# 指定 GPU
gpu_ind = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind  # 0,1,2,3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

###### Functions ######

def apply_wavelet_transform(data, wavelet='db4', level=4, mode='symmetric'):
    wavelet_transformed = []
    for i in range(data.shape[0]):
        try:
            coeffs = pywt.wavedec(data[i], wavelet=wavelet, level=level, mode=mode)
            transformed = np.concatenate(coeffs, axis=-1)
            wavelet_transformed.append(transformed)
        except Exception as e:
            print(f"Error transforming data at index {i}: {e}")
            wavelet_transformed.append(np.zeros(data[i].shape))  

    return np.array(wavelet_transformed)

# def process_mat_file_normalized(file_path):
#     data = spio.loadmat(file_path, squeeze_me=True)
#     LEDCood_f = data['combined_illum_ref']
#     images = data['Efield_phase_cropped']

#     processed_data = []
#     min_val = np.min(LEDCood_f, axis=0)
#     max_val = np.max(LEDCood_f, axis=0)
#     normalized_LEDCood_f = (LEDCood_f - min_val) / (max_val - min_val)

#     for z in range(images.shape[2]):
#         for x in range(images.shape[0]):
#             for y in range(images.shape[1]):
#                 normalized_x = x / (images.shape[0] - 1)
#                 normalized_y = y / (images.shape[1] - 1)
#                 input_vector = np.append(normalized_LEDCood_f[z], [normalized_x, normalized_y])
#                 pixel_value = images[x, y, z]
#                 processed_data.append(np.append(input_vector, pixel_value))

#     return np.array(processed_data, dtype=np.float32)
def process_mat_file_normalized(file_path):
    data = spio.loadmat(file_path, squeeze_me=True)
    LEDCood_f = data['Illum_ref']
    images = data['Cropped_Efield_amplitude']

    processed_data = []
    min_val = np.min(LEDCood_f, axis=0)
    max_val = np.max(LEDCood_f, axis=0)
    normalized_LEDCood_f = (LEDCood_f - min_val) / (max_val - min_val)

    min_pixel_val = np.min(images)
    max_pixel_val = np.max(images)
    normalized_images = (images - min_pixel_val) / (max_pixel_val - min_pixel_val)

    for z in range(images.shape[2]):
        for x in range(images.shape[0]):
            for y in range(images.shape[1]):
                normalized_x = x / (images.shape[0] - 1)
                normalized_y = y / (images.shape[1] - 1)
                input_vector = np.append(normalized_LEDCood_f[z], [normalized_x, normalized_y])
                pixel_value = normalized_images[x, y, z]
                processed_data.append(np.append(input_vector, pixel_value))

    return np.array(processed_data, dtype=np.float32)


def get_fourier_image(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)  # 移动频率为中心
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)  # 计算幅度谱
    return magnitude_spectrum

def split_data_gap_deprecated(processed_data, num_total_imgs=500, img_size=360*360):
    # Create indices and shuffle them
#     indices = np.arange(num_total_imgs)
#     np.random.seed(42)  # Set a random seed for reproducibility
#     np.random.shuffle(indices)

#     # Select 110 images for training and 10 for validation/test
#     train_indices = indices[:110]  # First 110 for training
#     val_indices = indices[110:]    # Remaining 10 for validation/test
    indices = np.arange(num_total_imgs)
    val_indices = np.array([238,239,241,242])  # 验证集索引
    train_indices = np.setdiff1d(indices, val_indices)

#     val_indices = np.arange(0, 499, 2)

#     train_indices = np.arange(1, 500, 2)



    # Collect the corresponding data for train and test sets
    train_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in train_indices])
    val_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in val_indices])

    # Save the datasets
    np.save("data/test_data.npy", val_data)
    np.save("data/train_data.npy", train_data)
    np.save("data/val_data.npy", val_data)

    return train_data, val_data

def exp_decay(initial_lr, final_lr, epochs):
    decay_rate = -np.log(final_lr / initial_lr) / epochs
    return initial_lr * np.exp(-decay_rate * np.arange(epochs))

def train_and_infer(train_data, val_data, learning_rate, encoder_layer_num, feature_num, output_dir, L):
    net_kargs = {
        'skip_layers': range(2, encoder_layer_num, 2),
        'encoder_layer_num': encoder_layer_num,
        'decoder_layer_num': 1,
        'feature_num': feature_num,
        'ffm': 'exp_diag',
        'L': L
    }

    data_kargs = {'ic': 4, 'oc': 1}
    train_provider = Provider.StrictEpochProvider(train_data[:, :4], train_data[:, 4:], is_shuffle=False)
    valid_provider = Provider.StrictEpochProvider(val_data[:, :4], val_data[:, 4:], is_shuffle=False)

    lr = exp_decay(learning_rate, learning_rate * 0.01, 300)
    train_kargs = {
        'batch_size': 1296,
        'valid_size': 'full',
        'epochs': 50,
        'learning_rate': lr,
        'is_restore': False,
        'prediction_path': 'data/predictions',
        'save_epoch': 50
    }

    net = MLP(data_kargs=data_kargs, net_kargs=net_kargs)

    output_path = os.path.join(output_dir, 'model')
    net.train(output_path, train_provider, valid_provider, **train_kargs)

    # 推理代码
    test_data_path = 'data/test_data.npy'
    test_data = np.load(test_data_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(output_path, '50_model', 'model')
        saver.restore(sess, checkpoint_path)

        num_images = 10
        predicted_images = np.zeros((360, 360, num_images), dtype=np.float32)
        total_mse, total_ssim, total_snr, total_pcc = 0, 0, 0, 0

        for i in range(num_images):
            input_data = test_data[i * 360 * 360:(i + 1) * 360 * 360, :-1]
            ground_truth = test_data[i * 360 * 360:(i + 1) * 360 * 360, -1].reshape(360, 360)

            predictions = []
            for j in range(0, input_data.shape[0], 1296):
                batch_data = input_data[j:j + 1296]
                batch_pred = sess.run(net.xhat, feed_dict={net.x: batch_data})
                predictions.append(batch_pred)

            predictions = np.concatenate(predictions).reshape(360, 360)
            predicted_images[:, :, i] = predictions

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            ground_truth_fourier = get_fourier_image(ground_truth)
            predicted_fourier = get_fourier_image(predictions)

            # 计算 MSE 和 PCC
            mse_value = np.mean((predictions - ground_truth) ** 2)
            flat_predictions = predictions.flatten()
            flat_ground_truth = ground_truth.flatten()
            pcc_value, _ = pearsonr(flat_predictions, flat_ground_truth)

            # 使用 ground_truth 的最小值和最大值作为 vmin 和 vmax
            vmin, vmax = ground_truth.min(), ground_truth.max()

            # 显示预测图像和ground truth图像，并在标题中展示MSE和PCC
            pred_display = axes[0, 0].imshow(predictions, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f"Predicted Image {i} - PCC: {pcc_value:.2f}, MSE: {mse_value:.2f}")
            axes[0, 0].axis('off')

            gt_display = axes[0, 1].imshow(ground_truth, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f"Ground Truth {i}")
            axes[0, 1].axis('off')

            pred_fourier_display = axes[1, 0].imshow(predicted_fourier, cmap='gray')
            axes[1, 0].set_title(f"Predicted Fourier {i}")
            axes[1, 0].axis('off')

            gt_fourier_display = axes[1, 1].imshow(ground_truth_fourier, cmap='gray')
            axes[1, 1].set_title(f"Ground Truth Fourier {i}")
            axes[1, 1].axis('off')

            plt.savefig(os.path.join(output_dir, f"image_{i}.png"))
            plt.close()

            total_mse += mse_value
            total_pcc += pcc_value

        avg_mse = total_mse / num_images
        avg_pcc = total_pcc / num_images

        with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
            f.write(f"Average MSE: {avg_mse}\n")
            f.write(f"Average PCC: {avg_pcc}\n")
param_combinations = [
    {'lr': 5e-4, 'enc_layer_num': 16, 'feat_num': 4096 ,'L': 4},
]

data_file_path = 'data/processed_fields.mat'
processed_data = process_mat_file_normalized(data_file_path)
train_data, val_data = split_data_gap_deprecated(processed_data)

# 循环遍历特定的超参数组合
for params in param_combinations:
    lr = params['lr']
    enc_layer_num = params['enc_layer_num']
    feat_num = params['feat_num']
    L = params['L']

    output_dir = f'inference/reshaped_fieleds_amplitude_enc_{enc_layer_num}_feat_{feat_num}_L_{L}__L2_4_1000_jiou_0-1_50_center240_fang_50'
    train_and_infer(train_data, val_data, lr, enc_layer_num, feat_num, output_dir, L)
    print(f"Finished training and inference for lr={lr}, encoder_layers={enc_layer_num}, feature_num={feat_num}, L={L}")

