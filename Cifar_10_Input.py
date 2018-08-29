from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import tensorflow as tf
import matplotlib.pyplot as plt
 
# %matplotlib inline

IMAGE_SIZE = 32

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 #训练集的样本总数
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 #验证集的样本总数

cifar_label_bytes = 1  # 2 for CIFAR-100 第一个字节为label
cifar_height = 32
cifar_width = 32
cifar_depth = 3 #通道数

#生产批量输入
def generate_batch_inputs(eval_data, shuffle, data_dir, batch_size):
    """
    参数:
    eval_data: bool值,指定训练或者验证.
    shuffle: bool值,是否将数据顺序打乱.
    data_dir: CIFAR-10数据集所在目录.
    batch_size: 批量大小.

    返回值:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
 
    if not eval_data:
        filepath = os.path.join(data_dir, 'data_batch_*') 
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filepath = os.path.join(data_dir, 'test_batch*')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    files = tf.train.match_filenames_once(filepath)

    with tf.name_scope('input'):
        # tf.train.string_input_producer会使用初始化时提供的文件列表创建一个输入队列，
        # 创建好的输入队列可以作为文件读取函数的参数.
        # shuffle参数为True时，文件在加入队列之前会被打乱顺序
        # tf.train.string_input_producer生成的输入队列可以同时被多个文件读取线程操作，
        # 而且输入队列会将队列中的文件均匀地分配给不同的线程，不会出现有些文件被处理过多次而有些文件还没有被处理过的情况
        # 当一个输入队列中的所有文件都被处理完后，它会将初始化时提供的文件类表中的文件全部重新加入队列，
        # 通过num_epochs参数来限制加载初始化文件列表的最大轮数。当所有文件都已经被使用了设定的轮数后，
        # 如果继续尝试读取新的文件，输入队列会报OutOfRange的错误。这里我们取None不做限制
        filename_queue = tf.train.string_input_producer(files, shuffle=False, num_epochs=None)
    
        # 从文件队列读取样本
        image_bytes = cifar_height * cifar_width * cifar_depth
        #每条数据的长度
        record_bytes = cifar_label_bytes + image_bytes
        # 读取固定长度的一条数据
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)

        # 格式转换
        record_bytes = tf.decode_raw(value, tf.uint8)

        # 第一个字节为分类标签
        label = tf.cast(
            tf.strided_slice(record_bytes, [0], [cifar_label_bytes]), tf.int32)

        # 标签字节后面的字节表示图片信息 
        # reshape from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [cifar_label_bytes],
                           [cifar_label_bytes + image_bytes]),
            [cifar_depth, cifar_height, cifar_width])
        # Convert from [depth, height, width] to [height, width, depth].
        uint8image = tf.transpose(depth_major, [1, 2, 0])
    
        reshaped_image = tf.cast(uint8image, tf.float32)
        #plt.imshow(reshaped_image)
        '''
        if not eval_data:
            # 数据增强用于训练
            # 随机的对图片进行一些处理，原来的一张图片在多次epoch中就会生成多张不同的图片，这样就增加了样本数量
            #由于数据增强会耗费大量的CPU时间，因此我们用16个线程来处理

            # Randomly crop a [IMAGE_SIZE, IMAGE_SIZE] section of the image.
            resized_image = tf.random_crop(reshaped_image, [IMAGE_SIZE, IMAGE_SIZE, 3])

            # Randomly flip the image horizontally.
            resized_image = tf.image.random_flip_left_right(resized_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458.
            resized_image = tf.image.random_brightness(resized_image,
                                                     max_delta=63)
            resized_image = tf.image.random_contrast(resized_image,
                                                   lower=0.2, upper=1.8)

        else:
            # 裁剪中间部分用于验证
            resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               IMAGE_SIZE, IMAGE_SIZE)
        
        # 减去均值并除以像素的方差
        float_image = tf.image.per_image_standardization(resized_image)
        '''
        
        # 这里我们不对图片进行任何处理，得到更大的图像，以便后面训练得到更好的精度
        float_image = reshaped_image 
        float_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    
    # TensorFlow提供了tf.train.shuffle_batch和tf.train.batch函数来将单个的样例组织成batch的形式输出，
    # 这两个函数都会生成一个队列，队列的入队操作是生成单个样例的方法，而每次出队得到的是一个batch的样例，它们唯一的区别在于是否会将数据顺序打乱
    # 参数capacity表示最多可以存储的样例个数，太大就会占用很多内存，太小则会因为出队操作没有数据而阻塞，影响效率
    # 参数min_after_dequeue限制了出队时队列中元素的最少个数，当队列中元素太少时，随机打乱样例顺序的作用就不大了。
    # tf.train.shuffle_batch和tf.train.batch函数除了可以将单个训练数据整理成输入batch，也提供了并行化处理数据的方法。
    # 参数num_threads，可以指定多个线程同时执行入队操作。

    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [float_image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [float_image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  #tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

