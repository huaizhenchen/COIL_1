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
gpu_ind = '2'
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

def process_mat_file_normalized(file_path):
    data = spio.loadmat(file_path, squeeze_me=True)
    LEDCood_f = data['LEDCood_f']
    images = data['I_Raw']

    processed_data = []
    min_val = np.min(LEDCood_f, axis=0)
    max_val = np.max(LEDCood_f, axis=0)
    normalized_LEDCood_f = (LEDCood_f - min_val) / (max_val - min_val)

    for z in range(images.shape[2]):
        for x in range(images.shape[0]):
            for y in range(images.shape[1]):
                normalized_x = x / (images.shape[0] - 1)
                normalized_y = y / (images.shape[1] - 1)
                input_vector = np.append(normalized_LEDCood_f[z], [normalized_x, normalized_y])
                pixel_value = images[x, y, z]
                processed_data.append(np.append(input_vector, pixel_value))

    return np.array(processed_data, dtype=np.float32)

def split_data_gap_deprecated(processed_data, num_total_imgs=120, img_size=360*360):
    indices = np.arange(num_total_imgs)
    val_indices = np.array([112,113,114, 116,117,118])  # 验证集索引
    train_indices = np.setdiff1d(indices, val_indices)

    train_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in train_indices])
    val_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in val_indices])

    np.save("data/test_data.npy", val_data)
    np.save("data/train_data.npy", train_data)

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

    lr = exp_decay(learning_rate, learning_rate * 0.01, 1000)
    train_kargs = {
        'batch_size': 1296,
        'valid_size': 'full',
        'epochs': 100,
        'learning_rate': lr,
        'is_restore': False,
        'prediction_path': 'data/predictions',
        'save_epoch': 100
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
        checkpoint_path = os.path.join(output_path, '100_model', 'model')
        saver.restore(sess, checkpoint_path)

        num_images = 6
        predicted_images = np.zeros((360, 360, num_images), dtype=np.float32)
        total_mse, total_pcc = 0, 0

        # 提取第115号图像的 ground truth
        ground_truth_115 = train_data[112 * 360 * 360:113 * 360 * 360, -1].reshape(360, 360)

        for i in range(num_images):
            input_data = test_data[i * 360 * 360:(i + 1) * 360 * 360, :-1]
            ground_truth = test_data[i * 360 * 360:(i + 1) * 360 * 360, -1].reshape(360, 360)

            predictions = []
            for j in range(0, input_data.shape[0], 1296):
                batch_data = input_data[j:j + 1296]
                batch_pred = sess.run(net.xhat, feed_dict={net.x: batch_data})
                predictions.append(batch_pred)

            predictions = np.concatenate(predictions).reshape(360, 360)

            # 计算与真实值的MSE
            mse_value = np.mean((predictions - ground_truth) ** 2)

            # 计算与第115号图像的PCC
            flat_predictions = predictions.flatten()
            flat_ground_truth_115 = ground_truth_115.flatten()
            pcc_value, _ = pearsonr(flat_predictions, flat_ground_truth_115)

            total_mse += mse_value
            total_pcc += pcc_value

            # 绘制图像
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(ground_truth, cmap='jet', vmin=1, vmax=4)
            axs[0].set_title("Ground Truth")
            axs[0].axis('off')
            plt.colorbar(axs[0].imshow(ground_truth, cmap='jet', vmin=1, vmax=4), ax=axs[0])

            axs[1].imshow(predictions, cmap='jet', vmin=1, vmax=4)
            axs[1].set_title("Predictions")
            axs[1].axis('off')
            plt.colorbar(axs[1].imshow(predictions, cmap='jet', vmin=1, vmax=4), ax=axs[1])

            axs[2].imshow(np.abs(predictions - ground_truth), cmap='jet', vmin=1, vmax=4)
            axs[2].set_title("Absolute Error")
            axs[2].axis('off')
            plt.colorbar(axs[2].imshow(np.abs(predictions - ground_truth), cmap='jet', vmin=1, vmax=4), ax=axs[2])

            # 在图像上标注指标
            fig.suptitle(f"MSE: {mse_value:.4f}, PCC with Image 115: {pcc_value:.4f}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"image_{i}_results.png"))
            plt.close(fig)

        # 计算平均值
        avg_mse = total_mse / num_images
        avg_pcc = total_pcc / num_images

        # 打印平均值
        print(f"Average MSE with ground truth: {avg_mse:.4f}")
        print(f"Average PCC with Image 115: {avg_pcc:.4f}")

param_combinations = [
    {'lr': 1e-4, 'enc_layer_num': 16, 'feat_num': 4096, 'L': 4},
]

data_file_path = 'data/IRaw_CElegans_Shw_360.mat'
processed_data = process_mat_file_normalized(data_file_path)
train_data, val_data = split_data_gap_deprecated(processed_data)

for params in param_combinations:
    lr = params['lr']
    enc_layer_num = params['enc_layer_num']
    feat_num = params['feat_num']
    L = params['L']

    output_dir = f'inference/IRaw_CElegans_Shw_360_PCCto115_expdiag_enc_{enc_layer_num}_feat_{feat_num}_L_{L}_true112-118'
    train_and_infer(train_data, val_data, lr, enc_layer_num, feat_num, output_dir, L)
    print(f"Finished training and inference for lr={lr}, encoder_layers={enc_layer_num}, feature_num={feat_num}, L={L}")


