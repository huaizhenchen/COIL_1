# inputs provider class

# Yu Sun, CIG, WUSTL, 2019


from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
from PIL import Image

####### Epoch Provider #######

class BaseEpochProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
    
    def __call__(self, n, iter, fix=False):
        if type(n) == int and not fix:
            # X and Y are the images and truths
            train_inputs, truths = self._next_batch(n, iter)
        elif type(n) == int and fix:
            train_inputs, truths = self._fix_batch(n)
        elif type(n) == str and n == 'full':
            train_inputs, truths = self._full_batch() 
        else:
            raise ValueError("Invalid batch_size: "%n)
        
        # ensure dimensionality is correct
        if train_inputs.ndim is 1:
            train_inputs = np.expand_dims(train_inputs,-1)
        if truths.ndim is 1:
            truths= np.expand_dims(truths,-1)

        return train_inputs, truths

    def _next_batch(self, n, iter):
        pass

    def _full_batch(self):
        pass

class StrictEpochProvider(BaseEpochProvider):
    
    def __init__(self, inputs, truths, is_shuffle=True):
        super(BaseEpochProvider, self).__init__()
        self.inputs = inputs
        self.truths = truths
        self.file_count = inputs.shape[0]
        if is_shuffle:
            self.reset()
        else:
            image_size = 360
            block_size = 36

            # 计算图像数量
            num_images = self.file_count // (image_size * image_size)

            # 创建初始列表
            initial_list = [i for i in range(self.file_count)]

            # 将列表重塑为 (num_images, 360, 360) 的数组
            reshaped_array = np.reshape(initial_list, (num_images, image_size, image_size))

            # 创建一个空列表来存储重新排序后的索引
            reordered_list = []

            for img_index in range(reshaped_array.shape[0]):
                for block_row in range(image_size // block_size):
                    for block_col in range(image_size // block_size):
                        block = reshaped_array[img_index, 
                                               block_row * block_size:(block_row + 1) * block_size,
                                               block_col * block_size:(block_col + 1) * block_size]
                        reordered_list.extend(block.flatten())

            # 转换为numpy数组
            self.batch_index_set = np.array(reordered_list)
            

    def reset(self):
        self.batch_index_set = np.random.permutation(self.file_count)

    def _next_batch(self, n, iter):
        idx = self.batch_index_set[iter*n:iter*n+n]
        X = self.inputs[idx,...]
        Y = self.truths[idx,...]
        return X, Y

    def _fix_batch(self, n):
        # first n inputs
        X = self.inputs[0:n]
        Y = self.truths[0:n]
        return X, Y

    def _full_batch(self):
        return self.inputs, self.truths

    def _process_truths(self, truth):
        # normalization by channels
        truth = np.clip(np.fabs(truth), self.a_min, self.a_max)
        for channel in range(self.truth_channels):
            truth[:,:,channel] -= np.amin(truth[:,:,channel])
            truth[:,:,channel] /= np.amax(truth[:,:,channel])
        truth[:,:,channel] = truth[:,:,channel]
        return truth

    def _process_inputs(self, inputs):
        # normalization by channels
        inputs = np.clip(np.fabs(inputs), self.a_min, self.a_max)
        for channel in range(self.img_channels):
            inputs[:,:,channel] -= np.amin(inputs[:,:,channel])
            inputs[:,:,channel] /= np.amax(inputs[:,:,channel])
        inputs[:,:,channel] = inputs[:,:,channel]
        return inputs