import scipy.io as spio
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

# Load the data
data_file_path = 'data/Diatom.mat'
data = spio.loadmat(data_file_path)
I_Raw = data['I_Raw']  # Assuming shape is (500, 500, 24)

# Define the Sobel edge detection function
def sobel_edges(image):
    sobel_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=tf.float32)
    sobel_y = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)

    image = tf.reshape(image, [1, image.shape[0], image.shape[1], 1])  # Add batch and channel dimensions
    gx = tf.nn.conv2d(image, tf.reshape(sobel_x, [3, 3, 1, 1]), strides=[1, 1, 1, 1], padding='SAME')
    gy = tf.nn.conv2d(image, tf.reshape(sobel_y, [3, 3, 1, 1]), strides=[1, 1, 1, 1], padding='SAME')

    edges = tf.sqrt(tf.square(gx) + tf.square(gy))
    return tf.reshape(edges, [image.shape[1], image.shape[2]])

# Initialize an array to accumulate edge masks
accumulated_edges = np.zeros((700, 700), dtype=np.float32)

# Define the size of the border to ignore (e.g., 10 pixels)
border_size = 10

# Create a mask to ignore the borders
border_mask = np.ones((700, 700), dtype=np.float32)
border_mask[:border_size, :] = 0  # Top border
border_mask[-border_size:, :] = 0  # Bottom border
border_mask[:, :border_size] = 0  # Left border
border_mask[:, -border_size:] = 0  # Right border

# Disable XLA and set the session to use CPU
config = tf.ConfigProto(device_count={'GPU': 0})  # Disable GPU
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF  # Disable XLA

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(I_Raw.shape[-1]):
        image = I_Raw[:, :, i]
        
        # Apply Gaussian smoothing to reduce noise
        smoothed_image = gaussian_filter(image, sigma=2.0)
        
        # Perform Sobel edge detection
        edges = sess.run(sobel_edges(smoothed_image))
        
        # Accumulate the edge mask
        accumulated_edges += edges

# Apply the border mask to ignore borders during normalization
masked_edges = accumulated_edges * border_mask

# Normalize the accumulated edge map to the range [0, 1] (excluding borders)
min_val = np.min(masked_edges[border_mask > 0])
max_val = np.max(masked_edges[border_mask > 0])
normalized_weight_map = (masked_edges - min_val) / (max_val - min_val)

# Set values below 0.1 to 0
normalized_weight_map[normalized_weight_map < 0.0001] = 0.0001

# Restore the ignored borders to 0
normalized_weight_map[border_mask == 0] = 0

# Save the normalized weight map to a .mat file
output_file_path = 'normalized_weight_map_sigma2_diatom.mat'
spio.savemat(output_file_path, {'weight': normalized_weight_map})
