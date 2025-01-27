import os
import pywt
import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import scipy.io as spio
import h5py
import tensorflow as tf
from NeuralNetwork.models.MLP_wavelet import MLP
from NeuralNetwork import Provider2 as Provider
import cv2

# 指定 GPU
gpu_ind = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind  # 0,1,2,3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

###### Functions ######

def save_processed_data(processed_data, save_path='data/processed_data_2d.npy'):
    np.save(save_path, processed_data)

def apply_wavelet_transform(data, wavelet='sym4', level=2, mode='symmetric'):
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

def process_mat_file_normalized(file_path, denoise_method=None):
    data = spio.loadmat(file_path, squeeze_me=True)
    LEDCood_f = data['LEDCoord_f']  # 假设参数向量是 LEDCood_f
    images = data['I_Raw']  # 假设图像数据存储在 I_Raw 字段

    processed_data = []

    # 找到 LEDCood_f 中每个参数的最小值和最大值
    min_val = np.min(LEDCood_f, axis=0)
    max_val = np.max(LEDCood_f, axis=0)

    # 将 LEDCood_f 和 x, y 坐标归一化到 [0, 1] 范围
    normalized_LEDCood_f = (LEDCood_f - min_val) / (max_val - min_val)

    print(images.shape[0], images.shape[1], images.shape[2])
    for z in range(images.shape[2]):
        print("z: ", z)
        for x in range(images.shape[0]):
            for y in range(images.shape[1]):
                normalized_x = x / (images.shape[0] - 1)  # 归一化到 [0, 1]
                normalized_y = y / (images.shape[1] - 1)  # 归一化到 [0, 1]
                
                # 构建 input_vector
                input_vector = np.append(normalized_LEDCood_f[z], [normalized_x, normalized_y])
                
                # 对 input_vector 进行小波变换
                transformed_input_vector = apply_wavelet_transform(input_vector.reshape(1, -1)).flatten()

                # 获取像素值并将其附加到处理后的数据
                pixel_value = images[x, y, z]
                processed_data.append(np.append(transformed_input_vector, pixel_value))

    # 将 processed_data 转换为 numpy 数组
    processed_data = np.array(processed_data, dtype=np.float32)

    np.save('data/processed_data_total_normalized_wavelet_200.npy', processed_data)
    return processed_data


def split_data_gap_deprecated(processed_data, num_total_imgs=24, img_size=200*200):
    indices = np.arange(num_total_imgs)
    train_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    val_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    test_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    train_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in train_indices])
    val_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in val_indices])
    test_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in test_indices])

    np.save("data/test_data_200.npy", test_data)
    np.save("data/train_data_200.npy", train_data)
    np.save("data/val_data_200.npy", val_data)

    return train_data, val_data

def exp_decay(initial_lr, final_lr, epochs):
    decay_rate = -np.log(final_lr / initial_lr) / epochs
    return initial_lr * np.exp(-decay_rate * np.arange(epochs))

if __name__ == "__main__":
    net_kargs = {
        'skip_layers': range(2, 12, 2),
        'encoder_layer_num': 12,
        'decoder_layer_num': 1,
        'feature_num': 512,
        'ffm': 'none',
        'L': 4
    }

    data_file_path = 'data/Diatom_200.mat'
    processed_data_path = 'data/processed_data_total_normalized_wavelet_200.npy'

    if os.path.exists(processed_data_path):
        print("data loading")
        processed_data = np.load(processed_data_path)
    else:
        print("data processing")
        processed_data = process_mat_file_normalized(data_file_path)

    print("data loaded")

    train_data, val_data = split_data_gap_deprecated(processed_data)

    print("data splitted")

    train_provider = Provider.StrictEpochProvider(train_data[:, :4], train_data[:, 4:], is_shuffle=False)
    valid_provider = Provider.StrictEpochProvider(val_data[:, :4], val_data[:, 4:], is_shuffle=False)

    print("data provided")

    start = 1e-4
    end = 1e-6
    lr = exp_decay(start, end, 200)

    data_kargs = {'ic': 4, 'oc': 1}
    train_kargs = {
        'batch_size': 400,
        'valid_size': 'full',
        'epochs': 200,
        'learning_rate': lr,
        'is_restore': False,
        'prediction_path': 'data/predictions',
        'save_epoch': 5
    }

    net = MLP(data_kargs=data_kargs, net_kargs=net_kargs)

    output_path = 'save/cb_1612048_l2_001_wodrop_lr1416_bs256/'
    prediction_path = 'save/cb_162048_l2_001_wodrop_lr1416_bs256/'
    net.train(output_path, train_provider, valid_provider, **train_kargs)
