from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  
import tensorflow as tf
import Cifar_10_Input
import Cifar_10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './models_cifar10',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")
# 每台机器的 GPU 个数，这里单机上运行，因此为 0
tf.app.flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")

# 定义分布式参数
# 参数服务器parameter server节点
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
# 两个worker节点
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")

# 设置job name参数
tf.app.flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
tf.app.flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
tf.app.flags.DEFINE_bool('sync_replicas', True,'Whether or not to synchronize the replicas during training.')
# 如果服务器已经存在，采用 gRPC 协议通信;如果不存在，采用进程间通信
tf.app.flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
# 在同步训练模式下，设置收集的工作节点的数量。默认就是工作节点的总数
tf.app.flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update "
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = Cifar_10_Input.IMAGE_SIZE
NUM_CLASSES = Cifar_10_Input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = Cifar_10_Input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = Cifar_10_Input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# 使用多个服务器的多个gpu训练
def train():
    """Train CIFAR-10 for a number of steps."""

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    #如果是同步模式，则先获取同步更新模型参数所需要的副本数
    num_workers = len(worker_spec)
    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    
    if not FLAGS.existing_servers:
        # Not using existing servers. Create an in-process server.
        # 创建当前机器的server，用以连接到cluster
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        # 如果当前节点是parameter server，则不再进行后续的操作，而是使用server.join等待worker工作
        if FLAGS.job_name == "ps":
            server.join()   
        
    is_chief = (FLAGS.task_index == 0)
    '''
    设置 CUDA_VISIBLE_DEVICES 环境变量，限制各个工作节点只可见一个 GPU，启动进程时添加环境变量即可。
    例如，每个工作节点只能访问一个 GPU，在代码中不需要额外指定
    这里是通过代码来分配
    '''
    if FLAGS.num_gpus > 0:
        # 避免gpu分配冲突：现在为相应机器中的每个worker分配task_num - > #gpu
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
        
    # 我们使用 tf.train.replica_device_setter 将涉及变量的操作分配到参数服务器上，并使用 CPU;
    # 将涉及非变量的操作分配到工作节点上，使用 worker_device 的值。
    # 在这个 with 语句之下定义的参数，会自动分配到参数服务器上去定义 如果有多个参数服务器，就轮流循环分配
    # 在深度学习训练中，一般图的计算，对于每个worker task来说，都是相同的，所以我们会把所有图计算、变量定义等代码，都写到下面这个语句下
    with tf.device(
        tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device='/job:ps/cpu:0',
        cluster=cluster
    )):
        '''-------------------------------分割线---------------------------------'''
        # 全局训练step数 
        global_step = tf.Variable(0, name="global_step", trainable=False)
        training = tf.placeholder_with_default(False, shape=(), name='training')
        # Get images and labels for CIFAR-10.
        images, labels = Cifar_10_Input.generate_batch_inputs(eval_data=(training==False), shuffle=True, 
                                                              data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
        # Reuse variables for the next tower.
        # tf.get_variable_scope().reuse_variables()
        
        logits = Cifar_10.inference(images, is_training=training)
        
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        
        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        '''-------------------------------分割线---------------------------------'''

        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate
            
            # 创建同步训练的优化器，tf.train.SyncReplicasOptimizer实质上是对原有优化器的一个扩展，
            # 我们传入原有优化器及其他参数，它会将原有优化器改造为同步分布式训练版本
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="cifar10_sync_replicas")
            
        # 记得传入global_step以同步
        train_step = opt.minimize(loss, global_step=global_step)

        #如果是同步并且是chief task
        if FLAGS.sync_replicas and is_chief:
            # 创建队列执行器
            chief_queue_runner = opt.get_chief_queue_runner()
            # 创建全局参数初始化器
            sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        
        # logdir地址不要写错了
        sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=FLAGS.train_dir, # 保存和加载模型的路径 通过sv.saver访问
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        # sv中已经创建，不能重复创建
        # saver = tf.train.Saver() 
        
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, # 软放置 如果该操作函数没有 GPU 实现时，会本动使用 CPU 设备
            log_device_placement=FLAGS.log_device_placement, # 告诉放置器在放置节点时记录消息
            device_filters=["/job:ps",
                        "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)
    
        if FLAGS.existing_servers:
            server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
            print("Using existing server at: %s" % server_grpc_url)

            sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)
    
        if FLAGS.sync_replicas and is_chief:
            # chief task要执行下面两个操作：1.全局变量初始化 2.启动队列 此时其他task都处于等待状态
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])
       
        '''-------------------------------执行训练---------------------------------'''
        # 这里没用到tf.train.Coordinator
        
        local_step = 0
        while True:
            _, loss_value ,step ,accuracy_value = sess.run([train_step, loss, global_step, accuracy],feed_dict={training: True})
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            local_step += 1
            
            format_str = ('%s: step %d, loss = %.2f ,Train accuracy = %.2f')
            print (format_str % (datetime.now(), step, loss_value, accuracy_value))
            
            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                sv.saver.save(sess, checkpoint_path, global_step=step)

                #  添加测试集验证部分
                sv.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
                _, loss_test  ,accuracy_test = sess.run([train_step, loss, accuracy],feed_dict={training: False})
                format_str = ('%s: loss = %.2f ,Test accuracy = %.2f')
                print (format_str % (datetime.now(), loss_test, accuracy_test))
            
            if step >= FLAGS.max_steps:
                break
    

def main(argv=None):  
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
