import torch
import numpy as np
import tensorflow as tf

# 定义一个函数，用于将PyTorch模型的权重文件转换为TensorFlow的检查点文件
def rename_var(pth_path, new_ckpt_path, num_classes):
    # 加载PyTorch模型的权重文件
    pytorch_dict = torch.load(pth_path)

    # 创建TensorFlow图和会话
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        # 初始化一个列表，用于存储新的TensorFlow变量
        new_var_list = []

        # 遍历PyTorch权重字典中的每个项
        for key, value in pytorch_dict.items():
            # 如果键在排除列表中，则跳过
            if key in except_list:
                continue

            # 设置新变量的名称
            new_name = key
            # 将PyTorch张量转换为NumPy数组
            value = value.detach().numpy()

            # 根据PyTorch权重的命名规则，替换为TensorFlow的命名规则
            if 'features.0' in new_name:
                new_name = new_name.replace("features.0.weight", "conv2d/kernel")
                new_name = new_name.replace("features.0.bias", "conv2d/bias")

            if 'features.3' in new_name:
                new_name = new_name.replace("features.3.weight", "conv2d_1/kernel")
                new_name = new_name.replace("features.3.bias", "conv2d_1/bias")

            if 'features.6' in new_name:
                new_name = new_name.replace("features.6.weight", "conv2d_2/kernel")
                new_name = new_name.replace("features.6.bias", "conv2d_2/bias")

            if 'features.8' in new_name:
                new_name = new_name.replace("features.8.weight", "conv2d_3/kernel")
                new_name = new_name.replace("features.8.bias", "conv2d_3/bias")

            if 'features.10' in new_name:
                new_name = new_name.replace("features.10.weight", "conv2d_4/kernel")
                new_name = new_name.replace("features.10.bias", "conv2d_4/bias")

            if 'classifier.1' in new_name:
                new_name = new_name.replace("classifier.1.weight", "dense/kernel")
                new_name = new_name.replace("classifier.1.bias", "dense/bias")

            if 'classifier.4' in new_name:
                new_name = new_name.replace("classifier.4.weight", "dense_1/kernel")
                new_name = new_name.replace("classifier.4.bias", "dense_1/bias")

            # 对于卷积层权重，进行转置以匹配TensorFlow的权重格式
            if 'conv2d' in new_name and 'kernel' in new_name:
                value = np.transpose(value, (2, 3, 1, 0)).astype(np.float32)
            # 对于其他权重，也进行转置并转换为float32类型
            else:
                value = np.transpose(value).astype(np.float32)

            # 创建新的TensorFlow变量，并将其添加到列表中
            re_var = tf.Variable(value, name=new_name)
            new_var_list.append(re_var)

        # 为新的全连接层创建TensorFlow变量，并使用He初始化方法初始化权重
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([4096, num_classes]), name="dense_2/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="dense_2/bias")
        new_var_list.append(re_var)

        # 创建TensorFlow保存器，用于保存新的变量
        saver = tf.compat.v1.train.Saver(new_var_list)
        # 初始化所有新创建的变量
        sess.run(tf.compat.v1.global_variables_initializer())
        # 保存新的变量为检查点文件
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)

# 定义排除列表，包含不需要转换的权重键名
except_list = ['classifier.6.weight', 'classifier.6.bias']
# PyTorch预训练模型的路径
pth_path = './alexnet-owt-4df8aa71.pth'
# 新的TensorFlow检查点文件的路径
new_ckpt_path = './pretrain_weights.ckpt'
# 目标分类任务的类别数
num_classes = 5
# 调用函数执行转换
rename_var(pth_path, new_ckpt_path, num_classes)