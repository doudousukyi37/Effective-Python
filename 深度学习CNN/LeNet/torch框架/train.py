import torch
import torch.utils
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import LeNet

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 5000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data',train=True,
                                             download=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=36,
                                               shuffle=True,num_workers=0)
    
    # 1000张验证图片
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image,val_label = next(val_data_iter) # 获取下一个批次的数据
    
    net = LeNet()
    loss_function = nn.CrossEntropyLoss() # 交叉熵损失
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    
    for epoch in range(5):
        
        runing_loss = 0.0
        for step,data in enumerate(train_loader,start=0):
            inputs,labels = data
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            
            runing_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs,dim=1)[1]
                    accuracy = torch.eq(predict_y,val_label).sum().item()/val_label.size(0)
                    
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    
    
    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()    