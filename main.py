#using https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
#with custom dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#image manipuliation (size)
import cv2
import os

data_root = "imagedata"
root = data_root + "_preprocessed"
classes = ("red", "blue")
IMG_WIDTH = 32
IMG_HEIGHT = 32

#TODO: fix image distortion when resizing (?)
def resizeImages():
    #make new directory for preprocessed images
    try:
        os.mkdir(root)
        for i in range(len(classes)):
            os.makedirs(root+"/"+classes[i])
    except:
        print("Directory already exists (and presumably the modified images do, too).")
    #resize the images in data_root and save them in the preprocessed folder
    for r, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".png"):
                fullpath = os.path.join(r,file)
                print(fullpath)
                subdir = fullpath.split("\\")[1]
                image = cv2.imread(fullpath, cv2.IMREAD_UNCHANGED)
                newimage = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                cv2.imwrite(root+"/"+subdir+"/"+file[:-4]+"_resized.png", newimage)
                print("Resized "+fullpath)


#neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    resizeImages()
    net = Net()
    #load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
    testset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)

    #train the network
    for epoch in range(10):  # loop over the dataset multiple times
        print("Epoch: "+str(epoch+1))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training.')

    #save the model
    torch.save(net.sate_dict(), "./trained_net.pth")

    #test the model
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    print("Program ran successfully.")
if __name__ == "__main__":
    main()