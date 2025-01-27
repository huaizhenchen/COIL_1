import os

from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

# Here indicating the GPU you want to use. If you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind  # 0,1,2,3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import scipy.io as spio
import numpy as np
import h5py
import tensorflow as tf
from NeuralNetwork.models.MLP_wavelet import MLP
from NeuralNetwork import Provider2 as Provider
import tensorflow as tf
import cv2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


###### Functions ######

def save_processed_data(processed_data, save_path='data/processed_data_2d.npy'):
    np.save(save_path, processed_data)



def process_mat_file_normalized(file_path, denoise_method=None):
    data = spio.loadmat(file_path, squeeze_me=True)
    LEDCood_f = data['LEDCoord_f']  # Assume the parameter vector is named LEDCood_f
    images = data['I_Raw']  # Assume image data is stored in the images field

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
    np.save('data/processed_data_total_normalized_600_5.npy', processed_data)
    return processed_data




def split_data_gap_deprecated(processed_data, num_total_imgs=24, img_size=700*700):
    """
    Split the processed data into training, validation, and test sets by alternating selection of images.

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
    train_indices = [0,2,4,6,8,10,12,14,16,18,20,22
       
    ]
    val_indices = [1,3,5,7,9,11,13,15,17,19,21,23
        
    ]
    test_indices = [1,3,5,7,9,11,13,15,17,19,21,23
        
    ]


    # Extract data based on calculated indices
    train_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in train_indices])
    val_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in val_indices])
    test_data = np.concatenate([processed_data[i*img_size:(i+1)*img_size] for i in test_indices])

    # Optionally save data to files
    np.save("data/test_data.npy", test_data)
    np.save("data/train_data.npy", train_data)
    np.save("data/val_data.npy", val_data)

    return train_data, val_data


data_kargs = {
    'ic': 4,  # Input channel size is 4 (original parameter vector plus pixel coordinates)
    'oc': 1  # Output channel size is 1 (predicted pixel value)
}




def exp_decay(initial_lr, final_lr, epochs):
    """
    Generates an array of exponentially decaying learning rates.

    Parameters:
    - initial_lr: float, initial learning rate
    - final_lr: float, final learning rate
    - epochs: int, number of epochs over which to decay the learning rate

    Returns:
    - An array of learning rates for each epoch.
    """
    decay_rate = -np.log(final_lr / initial_lr) / epochs
    return initial_lr * np.exp(-decay_rate * np.arange(epochs))

if __name__ == "__main__":
    net_kargs = {
    'skip_layers': range(2, 20, 2),
    'encoder_layer_num':20,
    'decoder_layer_num':1,
    'feature_num': 4096,
    'ffm': 'wavelet',
    'L':4
    }

    # Data file path
    data_file_path = 'data/Diatom.mat'

    processed_data_path = 'data/processed_data_total_normalized_600.npy'

    if os.path.exists(processed_data_path):
        print("data loading")
        processed_data = np.load(processed_data_path)
    else:
        print("data processing")
        processed_data = process_mat_file_normalized(data_file_path)
        # save_processed_data(processed_data, processed_data_path)

    print("data loaded")

    # Split the data into training and test sets
    train_data, val_data = split_data_gap_deprecated(processed_data)


    print("data splitted")

    # Initialize data providers
    # Initialize data providers
    train_provider = Provider.StrictEpochProvider(train_data[:, :4], train_data[:, 4:], is_shuffle=False)
    valid_provider = Provider.StrictEpochProvider(val_data[:, :4], val_data[:, 4:], is_shuffle=False)

    print("data provided")
    start = 5e-4
    end = 5e-6
    lr = exp_decay(start, end, 200)
    # Network initialization parameters
    data_kargs = {'ic': 4, 'oc': 1}
    train_kargs = {
        'batch_size': 4900,
        'valid_size': 'full',
        'epochs': 200,
        'learning_rate': lr,
        'is_restore': False,
        'prediction_path': 'data/predictions',
        'save_epoch': 5}

    # Initialize the network
    net = MLP(data_kargs=data_kargs, net_kargs=net_kargs)

    # Train the network
    output_path = 'save/cb_1612048_l2_001_wodrop_lr1416_bs256/'
    prediction_path = 'save/cb_162048_l2_001_wodrop_lr1416_bs256/'
    net.train(output_path, train_provider, valid_provider, **train_kargs)