import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
from NeuralNetwork.models.MLP15 import MLP  
import scipy.io as spio
from skimage.metrics import structural_similarity as ssim

gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 初始化模型参数
data_kargs = {'ic': 4, 'oc': 1}
net_kargs = {
    'skip_layers': range(2, 10, 2),
    'encoder_layer_num':10,
    'decoder_layer_num':1,
    'feature_num': 4096,
    'ffm': 'exp_diag',
    'L': 2
    }

# 创建MLP模型实例
mlp_model = MLP(data_kargs=data_kargs, net_kargs=net_kargs)

# 加载测试数据
test_data_path = 'data/test_data.npy'
test_data = np.load(test_data_path)

# 推理时设置的batch_size，避免内存不足
output_dir = 'inference/cb_1612048_l2_001_wodrop_lr1416_bs256/IRaw_CElegans_Shw_360_42_radial_encoding_xylxly_lr_0.0001_enc_10_feat_4096_L_2_100'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# TensorFlow 配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 初始化统计变量
total_mse = 0
total_ssim = 0
total_snr = 0
num_images=2
predicted_images = np.zeros((360, 360, num_images), dtype=np.float32)

# 创建TensorFlow会话
with tf.Session(config=config) as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 恢复模型
    saver = tf.train.Saver()
    model_path = '/projectnb/cislbu/huaizhen/TRY19/COIL/inference/IRaw_CElegans_Shw_360_42_radial_encoding_xylxly_lr_0.0001_enc_10_feat_4096_L_2_100/model/50_model/model'
    saver.restore(sess, model_path)

    # 执行预测
    batch_size = 1296  # 根据系统内存设置

    for i in range(2):
        print(i)
        input_data = test_data[i * 360 * 360:(i + 1) * 360 * 360, :-1]
        ground_truth = test_data[i * 360 * 360:(i + 1) * 360 * 360, -1].reshape(360, 360)

        predictions = []

        for j in range(0, input_data.shape[0], batch_size):
            batch_data = input_data[j:j + batch_size]
            batch_pred = sess.run(mlp_model.xhat, feed_dict={mlp_model.x: batch_data})
            predictions.append(batch_pred)

        predictions = np.concatenate(predictions).reshape(360, 360)
        predicted_images[:, :, i] = predictions

        # 保存预测图片和ground truth图片 # 修改部分
        plt.imsave(os.path.join(output_dir, f"predicted_image_{i}.png"), predictions, cmap='gray')
        plt.imsave(os.path.join(output_dir, f"ground_truth_{i}.png"), ground_truth, cmap='gray')

        # Plot predictions and ground truth for visual comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 获取 ground truth 的最小值和最大值
        vmin = ground_truth.min()
        vmax1 = ground_truth.max()
        vmax = vmax1 * 0.6

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

        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
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
    output_mat_file = 'IRaw_CElegans_Shw_360_42_radial_encoding_xylxly_lr_0.0001_enc_10_feat_4096_L_2_100.mat'
    spio.savemat(output_mat_file, {'I_Raw': predicted_images})

    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"Average MSE: {avg_mse}\n")
        f.write(f"Average SSIM: {avg_ssim}\n")
        f.write(f"Average SNR: {avg_snr}\n")

    print("Metrics summary written to", os.path.join(output_dir, 'metrics_summary.txt'))
