import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
from NeuralNetwork.models.MLP15 import MLP  
import scipy.io as spio
from skimage.metrics import structural_similarity as ssim

gpu_ind = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 鍒濆鍖栨ā鍨嬪弬鏁�
data_kargs = {'ic': 4, 'oc': 1}
net_kargs = {
    'skip_layers': range(2, 16, 2),
    'encoder_layer_num':16,
    'decoder_layer_num':1,
    'feature_num': 1024,
    'ffm': 'loglinear',
    'L': 10
    }

# 鍒涘缓MLP妯″瀷瀹炰緥
mlp_model = MLP(data_kargs=data_kargs, net_kargs=net_kargs)

# 鍔犺浇娴嬭瘯鏁版嵁
test_data_path = 'data/test_data.npy'
test_data = np.load(test_data_path)
0
# 鍒嗘壒閲忚繘琛屾帹鐞嗭紝璁剧疆閫傚綋鐨刡atch_size锛屼互閬垮厤鍐呭瓨涓嶈冻
output_dir = 'inference/cb_1612048_l2_001_wodrop_lr1416_bs4096/sim1_loglinear_lr_0.0003_enc_16_feat_1024_L_10_100_1000_dropout0.01_1535_50_single/1000_MODEL_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# TensorFlow 浼氳瘽閰嶇疆
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 鍒濆鍖栨寚鏍囩粺璁″彉閲�
total_mse = 0
total_ssim = 0
total_snr = 0
num_images=2
predicted_images = np.zeros((339, 339, num_images), dtype=np.float32)
# 鍒涘缓TensorFlow浼氳瘽
with tf.Session(config=config) as sess:
    # 鍒濆鍖栧彉閲�
    sess.run(tf.global_variables_initializer())

    # 鎭㈠妯″瀷
    saver = tf.train.Saver()
    model_path = '/projectnb/cislbu/huaizhen/TRY19/COIL/inference/sim1_loglinear_lr_0.0003_enc_16_feat_1024_L_10_100_1000_dropout0.01_1535_50_single/model/1000_model/model'
    saver.restore(sess, model_path)

    # 鍒嗘壒杩涜棰勬祴
    batch_size = 12769  # 鏍规嵁绯荤粺鍐呭瓨璁剧疆

    for i in range(2):
        print(i)
        input_data = test_data[i * 339 * 339:(i + 1) * 339 * 339, :-1]
        ground_truth = test_data[i * 339 * 339:(i + 1) * 339 * 339, -1].reshape(339, 339)

        predictions = []

        for j in range(0, input_data.shape[0], batch_size):
            batch_data = input_data[j:j + batch_size]
            batch_pred = sess.run(mlp_model.xhat, feed_dict={mlp_model.x: batch_data})
            predictions.append(batch_pred)

        predictions = np.concatenate(predictions).reshape(339, 339)
        predicted_images[:, :, i] = predictions

#         k=4*(i+1)-1
#         # 淇濆瓨鎺ㄧ悊鍥惧儚
#         plt.imsave(os.path.join(output_dir, f"image_{k}.png"), predictions, cmap='gray')
#         plt.imsave(os.path.join(output_dir, f"ground_truth_{i}.png"), ground_truth, cmap='gray')
        # Plot predictions and ground truth for visual comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 获取 ground truth 的最小值和最大值
        vmin = ground_truth.min()
        vmax1 = ground_truth.max()
        vmax=vmax1*0.6

        # 显示预测图像，确保 colorbar 的范围与 ground truth 一致
        pred_display = axes[0].imshow(predictions, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Predicted Image {i}")
        axes[0].axis('off')
        fig.colorbar(pred_display, ax=axes[0], orientation='vertical')  # Add color bar

        # 显示 ground truth 图像
        gt_display = axes[1].imshow(ground_truth, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Ground Truth {i}")
        axes[1].axis('off')
        fig.colorbar(gt_display, ax=axes[1], orientation='vertical')  # Add color bar

        plt.savefig(os.path.join(output_dir, f"image_{i}.png"))
        plt.close()  # Close the plot to free up memory

        # Calculate metrics
        mse_value = np.mean((predictions - ground_truth) ** 2)
        ssim_value = ssim(predictions, ground_truth, data_range=ground_truth.max() - ground_truth.min())
        signal_power = np.mean(ground_truth ** 2)
        noise_power = np.mean((predictions - ground_truth) ** 2)
        snr_value = 10 * np.log10(signal_power / noise_power)

        total_mse += mse_value
        total_ssim += ssim_value
        total_snr += snr_value

    # Calculate averages
    avg_mse = total_mse / num_images
    avg_ssim = total_ssim / num_images
    avg_snr = total_snr / num_images
    output_mat_file = 'sim1_loglinear_lr_0.0003_enc_16_feat_1024_L_10_100_1000_dropout0.01_1535_50_single.mat'
    spio.savemat(output_mat_file, {'I_Raw': predicted_images})

    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"Average MSE: {avg_mse}\n")
        f.write(f"Average SSIM: {avg_ssim}\n")
        f.write(f"Average SNR: {avg_snr}\n")

    print("Metrics summary written to", os.path.join(output_dir, 'metrics_summary.txt'))