# cifar10_distributed_train
cifar10 multi server multi gpu train

准备工作：
数据集下载地址：
http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

打开三个终端，首先cd到代码和数据集所在目录，分别运行以下命令：

终端一：
python3 Cifar_10_Multi_Server_Multi_gpu_Train.py --job_name=ps --task_index=0 

终端二：
python3 Cifar_10_Multi_Server_Multi_gpu_Train.py --job_name=worker --task_index=1  

终端三：
python3 Cifar_10_Multi_Server_Multi_gpu_Train.py  --job_name=worker --task_index=0 

