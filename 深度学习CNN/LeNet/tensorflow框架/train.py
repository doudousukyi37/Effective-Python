from __future__ import absolute_import,division,print_function,unicode_literals # 兼容 Python 2 和 Python 3 的代码
import tensorflow as tf
from model import MyModel

def main():
    mnist = tf.keras.datasets.mnist
    
    # 下载以及加载数据集
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train,x_test = x_train/255.0,x_test/255.0 # 归一化
    
    # 灰度图像增加通道维度
    x_train = x_train[...,tf.newaxis]
    x_test = x_test[...,tf.newaxis]
    
    # 创建数据生成器
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train,y_train).shuffle(10000).batch(32)
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test,y_test).batch(32)
    )
    
    # 创建模型
    model = MyModel()
    
    # 定义损失函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam()
    
    # 定义训练损失和训练准确率
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # 定义测试损失和测试准确率
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    # 定义训练函数，包括评估损失，应用数据生成器以及评估准确率
    @tf.function
    def train_step(images,labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels,predictions)
        gradients = tape.gradient(loss,model.trainale_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variable))
        
        train_loss(loss)
        train_accuracy(labels,predictions)
        
    # 定义测试函数
    @tf.function
    def test_step(images,labels):
        predictions = model(images)
        t_loss = loss_object(labels,predictions)
        
        test_loss(t_loss)
        test_accuracy(labels,predictions)
        
    EPOCHS = 5
    
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        for images,labels in train_ds:
            train_step(images,labels)
            
        for test_images,test_labels in test_ds:
            test_step(test_images,test_labels)
            
        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))


if __name__ == '__main__':
    main()    
