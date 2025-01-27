import scipy.io as spio
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def crop_images_and_save(mat_file_path, save_dir, mat_save_path, npy_save_path, crop_top=370, crop_bottom=210):
    # Load .mat file
    data = spio.loadmat(mat_file_path)
    images = data['I_Raw']
    labels = data['LEDCood_f']  # Assuming the labels are stored in 'LEDCood_f'
    
    # Check the shape of the images
    num_images = images.shape[2]
    height, width = images.shape[:2]
    
    if height != 1200 or width != 1200:
        raise ValueError("Images are not 1200x1200 pixels")
    
    # Crop images
    cropped_images = images[crop_top:height-crop_bottom, :, :]
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save each cropped image as a separate PNG file
    for i in range(num_images):
        plt.imsave(os.path.join(save_dir, f'image_{i}.png'), cropped_images[:, :, i], cmap='gray')
    print(f"Cropped images saved to {save_dir}")

    # Save all cropped images and labels to a MAT file
    spio.savemat(mat_save_path, {'cropped_images': cropped_images, 'LEDCood_f': labels})
    print(f"Cropped images and labels saved to {mat_save_path}")

    # Randomly select and display 3 images
    selected_indices = random.sample(range(num_images), 3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, idx in enumerate(selected_indices):
        axes[i].imshow(cropped_images[:, :, idx], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Image {idx}')
    plt.show()

    # Select images with indices ending in 3, 7, ..., 119
    selected_indices = list(range(0, num_images, 1))
    selected_images = cropped_images[:, :, selected_indices]
    selected_labels = labels[selected_indices]
    
    # Save the selected images and labels to an npy file
    np.save(npy_save_path, {'images': selected_images, 'labels': selected_labels})
    print(f"Selected images and labels saved to {npy_save_path}")

def process_mat_file_normalized(npy_save_path, processed_data_path):
    data = np.load(npy_save_path, allow_pickle=True).item()
    images = data['images']
    LEDCood_f = data['labels']

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
    np.save(processed_data_path, processed_data)
    return processed_data

def split_data_gap(processed_data, num_total_imgs=120, img_size=1200*620):
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

    return train_data, val_data, test_data

if __name__ == "__main__":

    # Paths
    mat_file_path = 'data/IRaw_CElegans_Shw.mat'
    save_dir = 'data/cropped_images'
    mat_save_path = 'data/cropped_images.mat'
    npy_save_path = 'data/selected_images_labels.npy'
    processed_data_path = 'data/processed_data.npy'

    # Crop images, save them as PNG files and a MAT file, save selected images and labels to an npy file, and display 3 randomly selected images
    crop_images_and_save(mat_file_path, save_dir, mat_save_path, npy_save_path)

    # Process cropped images and save them as a numpy file
    processed_data = process_mat_file_normalized(npy_save_path, processed_data_path)

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split_data_gap(processed_data)

    print("Data processing and splitting completed.")
