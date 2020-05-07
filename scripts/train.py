from torchvision import datasets, models, transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import json
import copy
import PIL
import os

def main():
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Path to config file
    config_path = os.path.join(root_dir,"config", "config.json")
    cfg_file = open(config_path)
    cfg = json.load(cfg_file)

    # Load the hyperparameters
    bs = cfg["batch_size"]
    
    train_dir = os.path.join(root_dir,"data","train")
    val_dir = os.path.join(root_dir,"data","val")
    output_dir = os.path.join(root_dir,"output")

    num_classes = cfg["num_classes"]
    epochs = cfg["epochs"]
    use_pretrained = bool(cfg["use_pretrained"])
    image_size = cfg["image_size"]
    lr = cfg["lr"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.squeezenet1_0(pretrained=True)
    for name,param in model.named_parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data = {
    'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
}

    train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
    train_data_size = len(data['train'])

    val_data = DataLoader(data['val'], batch_size=bs, shuffle=True)
    val_data_size = len(data['val'])
    
    model = model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    optimizer = optim.Adam(params_to_update, lr)
    criterion = nn.CrossEntropyLoss()

    train(epochs, train_data, val_data, optimizer, model, device, criterion, train_data_size, val_data_size)
    torch.save(model, os.path.join(output_dir, "model.pth"))

def train(epochs, train_data, val_data, optimizer, model, device, criterion, train_data_size, val_data_size):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        
        print('[INFO] Epoch {}/{}'.format(epoch, epochs - 1))
        running_loss = 0.0
        running_corrects = 0
        model.train()
        
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_data_size
        epoch_acc = running_corrects.double() / train_data_size
        print('[INFO] Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if ((epoch % 5) == 0):
            val_acc, val_loss = validation(model, val_data, device, criterion, optimizer, val_data_size)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
        
    return model

def validation(model, val_data, device, criterion, optimizer, test_data_size):
    
    with torch.no_grad():
        model.eval()
        running_loss = 0
        running_corrects = 0
        
        for inputs, labels in val_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / test_data_size
        val_acc = running_corrects.double() / test_data_size
        print('[INFO] Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

        return val_acc, val_loss

if __name__ == "__main__":
    main()

    










