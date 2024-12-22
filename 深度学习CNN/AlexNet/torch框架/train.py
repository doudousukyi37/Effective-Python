import os 
import sys 
import json
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import transforms,datasets,utils
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using{device} device")
    
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    image_path = os.path.join(data_root,"data_set","flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)     # 断言，出错程序会立即停止
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),
                                         transform=data_transform["train"])
    
    train_num = len(train_dataset)
    
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key,val in flower_list.items())
    # 将dict写入json文件
    json_str = json.dumps(cla_dict,indent=4)    # 将该字典转换为 JSON 字符串 缩进 4 个空格
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)
        
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8]) # 线程数
    print('Using {} dataloader workers every process'.format(nw))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,shuffle=True,
                                               num_workers=nw)
    
    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path,"val"),
        transform=data_transform['val']
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=4,shuffle=True,
        num_workers=nw
    )
    
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    test_data_iter = iter(validate_loader)
    # iter() 函数将 validate_loader 转换为一个迭代器。这意味着你可以使用 next() 函数或 for 循环来逐个获取 DataLoader 中的批次（batch）
    test_image,test_label = test_data_iter.next()
    
    # def imshow(img):
    #     img = img/2 + 0.5 # 去掉标准化
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg,(1,2,0)))
    #     plt.show()
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))   
    '''
    np.transpose(npimg, (1, 2, 0)) 是用来调整图像数据的维度顺序的。这是在使用 Matplotlib 的 plt.imshow() 函数显示图像时常用的一个步骤，因为 plt.imshow() 默认期望图像数据的维度顺序是 (高度, 宽度, 通道数)，即 (H, W, C)。

    解释一下各个部分：

    npimg：

    这是一个 NumPy 数组，代表图像数据。在深度学习和图像处理中，图像数据通常以 (通道数, 高度, 宽度) 的顺序存储，即 (C, H, W)。例如，对于 RGB 图像，npimg 的形状可能是 (3, H, W)。
    np.transpose()：

    这是 NumPy 提供的一个函数，用于交换数组的轴。在这里，它被用来重新排列 npimg 数组的维度。
    (1, 2, 0)：

    这是传递给 np.transpose() 函数的参数，指定了新的轴顺序。具体来说，这意味着：
    原始数组的第一个轴（通道数 C）将变成第三个轴。
    原始数组的第二个轴（高度 H）将变成第一个轴。
    原始数组的第三个轴（宽度 W）将变成第二个轴。
    
    '''
    
    net = AlexNet(num_classes=5,init_weights=True)
    
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimzer = optim.Adam(net.parameters(),lr=0.0002)
    
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,file=sys.stdout)
        # file=sys.stdout 意味着进度条会输出到标准输出流，也就是你的终端或命令行界面
        for step,data in enumerate(train_bar):
            images, labels = data
            optimzer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs,labels.to(device))
            loss.backward()
            optimzer.step()
            
            running_loss += loss.item()
            
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        
        # 验证
        net.eval() # dropout层不生效
        acc = 0.0 
        with torch.no_grad():
            val_bar = tqdm(validate_loader,file=sys.stdout)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()
                
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()        