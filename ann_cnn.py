# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:03:27 2024

@author: Priyanshu singh
"""
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


import torch
import torch.nn as nn
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
transform = transforms.Compose([
    transforms.Resize((64,64),interpolation=InterpolationMode.BOX),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

trainset = datasets.MNIST(root="./data",train=True,transform=transform,download=True)

trainloader = DataLoader(dataset=trainset,batch_size=60,shuffle=True,num_workers=0)

testset = datasets.MNIST(root = "./data",train = False,transform=transform,download=True)

testloader = DataLoader(dataset=testset,batch_size =50,shuffle = True,num_workers = 0)

#print(trainset.classes)
def show_images():
    batch = next(iter(testloader))
    images,labels = batch
    #print(images.size())

    grid = make_grid(images,nrow=8,padding=1,pad_value=1)
    grid = grid.numpy()
    plt.figure(figsize=(12,12))
    plt.axis("off")
    plt.title("IMAGES")
    plt.imshow(np.transpose(grid,(1,2,0)),cmap='gray')
    #print(labels)
    plt.show()
show_images()

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1)  
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  


#%%
class Neural_net(nn.Module):
    
    def __init__(self):
        super(Neural_net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=4096,out_features=2048*6,bias=False),
            nn.BatchNorm1d(2048*6),
            nn.ReLU(True),
            nn.Dropout(0.185),
            
            
            nn.Linear(in_features=2048*6,out_features=1024*6,bias=False),
            nn.BatchNorm1d(1024*6),
            nn.ReLU(True),
            nn.Dropout(0.400),
            
            nn.Linear(in_features=1024*6,out_features=1024*6,bias=False),
            nn.BatchNorm1d(1024*6),
            nn.ReLU(True),
            nn.Dropout(0.305),
            
            nn.Linear(1024*6, 512*6,bias=False),
            nn.BatchNorm1d(512*6),
            nn.ReLU(True),
            nn.Dropout(0.300),
            
            nn.Linear(512*6,10,bias =False),
            
            )
    def forward(self,input):
        return self.main(input)

model = Neural_net().to(device)
initialize_weights(model)
print(model)


#%%
num_epoch = 15
initial_lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=initial_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.1)
scaler = torch.cuda.amp.GradScaler()
loss_list= [100]
acc_list = [0]
torch.cuda.empty_cache()
for epoch in range(num_epoch):
    epoch_loss = 0
  
    correct = 0
    total = 0
    torch.cuda.empty_cache()
    for i,(images,labels) in enumerate(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.size(0),-1)
        #print(images.size())
        
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output,labels)
            
        scaler.scale(loss).backward()
        batch_loss = loss.item()
        _,predicted = torch.max(output.data,1)
        
        # MAKING BOOLEAN TENSORS
        batch_correct = (predicted==labels).sum().item()
        batch_total = labels.size(0)
        
        batch_accuracy = 100*(batch_correct/batch_total)
        scaler.step(optimizer)
        scaler.update()
        
        print(f" EPOCH:[{epoch+1}/{num_epoch}]---BATCH:[{i+1}/{len(trainloader)}]---Loss:{batch_loss:.4f},Acc:{batch_accuracy:.4f}")
        epoch_loss+=batch_loss
        correct+=batch_correct
        total+=batch_total
        
    scheduler.step()
    avg_loss = epoch_loss/len(trainloader)
    avg_acc = 100*(correct/total)
    loss_list.append(avg_loss)
    acc_list.append(avg_acc)
    print(f"Epoch [{epoch + 1}/{num_epoch}] complete. Average Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")
print("TRAINING COMPLETED!")

    
   
        
#%%

plt.figure(figsize=(10,5))
plt.plot(loss_list,label = "loss")
plt.plot(acc_list,label = "accuracy")

plt.xlabel("iterations")
plt.ylabel("Loss and accuracy")
plt.legend()
plt.show()

torch.save(model.state_dict(),"D:/pythonProject/ml_dl/MNIST_ANN_CNN/ann_digit_class_final.pth")

#%%
class CNN_net(nn.Module):
        def __init__(self):
            super(CNN_net, self).__init__()
            
            self.main = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=5,padding=0,stride=1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=1,padding=0),
                
                nn.Conv2d(in_channels = 64, out_channels = 64*2, kernel_size=5,padding=0,stride=1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=1,padding=0),
                
                
                nn.Conv2d(in_channels = 64*2, out_channels = 64*4, kernel_size=5,padding=0,stride=1,bias=False),
                nn.BatchNorm2d(64*4),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Dropout2d(0.3),
                nn.Conv2d(in_channels=64*4, out_channels=64*4, kernel_size=5,padding = 0,stride = 1,bias=False),
                nn.BatchNorm2d(64*4),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.3),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                
                nn.Conv2d(in_channels = 64*4, out_channels = 64*8, kernel_size=3,padding=1,stride=1,bias=False),
                nn.BatchNorm2d(64*8),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Dropout2d(0.35),
                nn.Conv2d(in_channels=64*8, out_channels=64*8, kernel_size=3,padding = 1,stride = 1,bias=False),
                nn.BatchNorm2d(64*8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.35),
                nn.MaxPool2d(kernel_size=2,stride=4,padding=1),
                 
                 
                nn.Conv2d(in_channels = 64*8, out_channels = 64*16, kernel_size=3,padding=1,stride=1,bias=False),
                nn.BatchNorm2d(64*16),
                
                nn.LeakyReLU(0.2,inplace = True),
                nn.Dropout2d(0.425),
                  
                 
                nn.Conv2d(in_channels=64*16, out_channels=64*16, kernel_size=3,padding = 1,stride = 1,bias=False),
                nn.BatchNorm2d(64*16),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.45),
                nn.MaxPool2d(kernel_size=2,stride=4,padding=1),
                nn.Flatten(),
                nn.Linear(1024*2*2,1000,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Linear(1000,10,bias=True),
                nn.LeakyReLU(0.2,inplace=True),
                
                
                
                )
        def forward(self,input):
            return self.main(input)
model2 = CNN_net().to(device)
print(model2)



num_epoch = 5
initial_lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model2.parameters(),lr=initial_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.2)
scaler = torch.cuda.amp.GradScaler()
loss_list= [100]
acc_list = [0]
torch.cuda.empty_cache()

for epoch in range(num_epoch):
    epoch_loss = 0
  
    correct = 0
    total = 0
    torch.cuda.empty_cache()
    for i,(images,labels) in enumerate(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        print("imagesize",images.size())
        
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model2(images)
            loss = criterion(output,labels)
            
        scaler.scale(loss).backward()
        batch_loss = loss.item()
        _,predicted = torch.max(output.data,1)
        
        # MAKING BOOLEAN TENSORS
        batch_correct = (predicted==labels).sum().item()
        batch_total = labels.size(0)
        
        batch_accuracy = 100*(batch_correct/batch_total)
        scaler.step(optimizer)
        scaler.update()
        
        print(f" EPOCH:[{epoch+1}/{num_epoch}]---BATCH:[{i+1}/{len(trainloader)}]---Loss:{batch_loss:.4f},Acc:{batch_accuracy:.4f}")
        epoch_loss+=batch_loss
        correct+=batch_correct
        total+=batch_total
        
    scheduler.step()
    avg_loss = epoch_loss/len(trainloader)
    avg_acc = 100*(correct/total)
    loss_list.append(avg_loss)
    acc_list.append(avg_acc)
    print(f"Epoch [{epoch + 1}/{num_epoch}] complete. Average Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")
print("TRAINING COMPLETED!")

#%%
plt.figure(figsize=(10,5))
plt.plot(loss_list,label = "loss")
plt.plot(acc_list,label = "accuracy")

plt.xlabel("iterations")
plt.ylabel("Loss and accuracy")
plt.legend()
plt.show()

torch.save(model2.state_dict(),"D:/pythonProject/ml_dl/MNIST_ANN_CNN/cnn_digit_class_final.pth")
    


#%%
'''
tens = torch.rand((64,64))
tens = tens.unsqueeze(0).unsqueeze(0)
# tensor.repeat(3,1,1)   REPEAT WILL TAKE ARGUMENTS Q,W,E AND MULTIPLY RESULTS and return tensor with shape Q*C,W*WIDTH,E*Height
tens=tens.to(device)
#print(tens.size())
output2 = model2(tens).to(device)

print(output2.size())
nn.Linear(in_features=256*45*45, out_features=10000,bias = False),
nn.LeakyReLU(0.2,inplace = True),
nn.BatchNorm1d(10000),
nn.Dropout(0.15),

nn.Linear(10000, 10,bias = False),
 
 '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    