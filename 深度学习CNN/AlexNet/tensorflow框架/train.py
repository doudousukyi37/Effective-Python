from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from model import AlexNet_v1,AlexNet_v2
import tensorflow as tf 
import json
import os 

def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))  # os.getcwd()：获取当前工作目录的路径
    image_path = os.path.join(data_root,"data_set","flower_data")
    train_dir = os.path.join(image_path,"train")
    validation_dir = os.path.join(image_path,"val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")
        
    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    
    # 数据生成器和数据处理
    train_image_generator = ImageDataGenerator(
        rescale = 1./255,   # 归一化
        horizontal_flip=True
    )
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    # class_mode='categorical' 将类别转换为ong-hot编码
    
    # 获取图像总数
    total_train = train_data_gen.n 
    
    # 获取类别字典
    class_indices = train_data_gen.class_indices
    
    # 转换键值对
    inverse_dict = dict((val,key) for key,val in class_indices.items())
    json_str = json.dumps(inverse_dict,indent=4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)
        
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))
    
    # sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding
    #
    # # This function will plot images in the form of a grid with 1 row
    # # and 5 columns where images are placed in each column.
    # def plotImages(images_arr):
    #     fig, axes = plt.subplots(1,5,figsize=(20,20))
    #     axes = axes.flatten()
    #     for img,ax in zip(images_arr,axes):
    #         ax.imshow(img)
    #         ax.axis('off')      # 关闭子图的坐标轴
    #     plt.tight_layout()      # 调整子图布局，确保子图之间不会重叠
    #     plt.show()
    
    model = AlexNet_v1(im_height=im_height,im_width=im_width,num_classes=5)
    # model = AlexNet_v2(num_classes=5)
    # model.build((batch_size,224,224,3))
    model.summary()
    
    # using keras high level api for training
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 不经过softmax激活的输出
                  metrics = ["accuracy"]
                  )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           monitor='val_loss'
                                           )
    ]
    
    # 训练，一次性将数据集都加载到内存中
    history = model.fit(
        x = train_data_gen,
        steps_per_epoch = total_train // batch_size,    # 取整
        epochs = epochs,
        validation_data = val_data_gen,
        validation_steps = total_val // batch_size,
        callbacks = callbacks
    )
    
    # 绘制损失和准确率
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]
    
    # 图1
    plt.figure()
    plt.plot(range(epochs),train_loss,label='train_loss')
    plt.plot(range(epochs),val_loss,label='val_loss')
    plt.legend()    # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    # 图2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    
    # 大批量数据集的训练方式
    # history = model.fit_generator(
    #     generator = train_data_gen,
    #     steps_per_epoch = total_train // batch_size,
    #     epochs = epochs,
    #     validation_data = val_data_gen,
    #     validation_steps = total_val // batch_size,
    #     callbacks = callbacks
    # )
    
    # 使用keras低级的api训练
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    #
    #
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    #
    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)
    #
    #
    # best_test_loss = float('inf')
    # for epoch in range(1, epochs+1):
    #     train_loss.reset_states()        # clear history info
    #     train_accuracy.reset_states()    # clear history info
    #     test_loss.reset_states()         # clear history info
    #     test_accuracy.reset_states()     # clear history info
    #     for step in range(total_train // batch_size):
    #         images, labels = next(train_data_gen)
    #         train_step(images, labels)
    #
    #     for step in range(total_val // batch_size):
    #         test_images, test_labels = next(val_data_gen)
    #         test_step(test_images, test_labels)
    #
    #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #     print(template.format(epoch,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           test_loss.result(),
    #                           test_accuracy.result() * 100))
    #     if test_loss.result() < best_test_loss:
    #        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')


if __name__ == '__main__':
    main()