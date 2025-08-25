import os
import sys
import warnings
import random
import numpy as np

import pandas as pd
import matplotlib as mpl
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torch.optim import Adam

from elu_approx import approx_elu

SEED = 110

mpl.rcParams['figure.dpi']= 300
mpl.rcParams["savefig.dpi"] = 300
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_PATH = "./Dataset/utkface_aligned_cropped/crop_part1/"
SAVE_PATH = "./"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
ppa_dir = os.path.join(parent_dir, "piecewise_polynomial_approximation")
sys.path.insert(0, ppa_dir)

ALPHA1 = 1.0

activation_flag = 0

if len(sys.argv) > 1:
    try:
        activation_flag = int(sys.argv[1])
    except ValueError:
        print("activation_flag must be 0 or 1")
        sys.exit(1)

if len(sys.argv) > 2:
    try:
        SEED = int(sys.argv[2])
    except ValueError:
        print("seed must be an integer, e.g., 110")
        sys.exit(1)        

def set_seed(seed: int = 42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# ELU
def elu(x):
    pos_flag = (x > 0).to(x.dtype)
    res = x * pos_flag + (1 - pos_flag) * ALPHA1 * (torch.exp(x) - 1)
    return res

class Activation_ApproxELU(nn.Module):
    def __init__(self, bound=30.0):
        super(Activation_ApproxELU, self).__init__()
        self.bound = bound

    def forward(self, x):      
        x_clipped = linear_clip(x, bound=self.bound)
        return approx_elu(x_clipped)                                       
    
class Activation_ELU(nn.Module):
    def __init__(self, bound=30.0):
        super(Activation_ELU, self).__init__()
        self.bound = bound
    
    def forward(self, x):
        x_clipped = linear_clip(x, bound=self.bound)
        return elu(x_clipped)

def linear_clip(x, bound=5.0):
    return torch.clamp(x, min=-bound, max=bound)

def reload_data():
    age_list = []
    gender_list = []
    race_list = []
    datetime_list = []
    filename_list = []

    for filename in sorted(os.listdir(DATA_PATH)):
        args = filename.split("_")

        if len(args) < 4:
            age = int(args[0])
            gender = int(args[1])
            race = 4
            datetime = args[2].split(".")[0]
        else:
            age = int(args[0])
            gender = int(args[1])
            race = int(args[2])
            datetime = args[3].split(".")[0]

        age_list.append(age)
        gender_list.append(gender)
        race_list.append(race)
        datetime_list.append(datetime)
        filename_list.append(filename)

    d = {'age': age_list, 'gender': gender_list, 'race': race_list, 'datetime': datetime_list, 'filename': filename_list}
    return pd.DataFrame(data=d)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.act1  = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=2, padding=2)  
        self.act2  = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.pool2 = nn.MaxPool2d(2, 2)     

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.act3  = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.pool3 = nn.MaxPool2d(2, 2)         

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.act4  = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.pool4 = nn.MaxPool2d(2, 2)         

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act5  = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.pool5 = nn.MaxPool2d(2, 2)      

        flat_dim = 128 * 3 * 3                   
        self.fc1 = nn.Linear(flat_dim, 120)
        self.act6 = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.dropout1 = nn.Dropout(0.0)

        self.fc2 = nn.Linear(120, 84)
        self.act7 = Activation_ELU() if activation_flag == 0 else Activation_ApproxELU()
        self.dropout2 = nn.Dropout(0.0)

        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        x = self.pool5(self.act5(self.conv5(x)))

        x = x.view(x.size(0), -1)        
        x = self.dropout1(self.act6(self.fc1(x)))
        x = self.dropout2(self.act7(self.fc2(x)))
        x = self.fc3(x)
        return x

class CustomImageDataset(Dataset):
    def __init__(self, df_, img_transform=None, target_transform=None):
        if img_transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                img_transform
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])        
    
        self.target_transform = target_transform
        self.df_ = df_

    def __len__(self):
        return len(self.df_['age'])

    def __getitem__(self, idx):
        image = Image.open(DATA_PATH + self.df_['filename'].iloc[idx])
        label = self.df_['age'].iloc[idx]

        image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def testMeanAbsoluteError(model, test_, testmode, bits):
    if testmode == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        was_training = model.training
        model = model.to(device).eval()
        
        running_mae = 0.0
        test_loader = CustomImageDataset(test_)

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device, dtype=torch.float64)
                if len(images.size()) == 3:
                    images = images.unsqueeze(0)
                outputs = model(images)
                predicted = torch.max(outputs.data)
                predicted = predicted.cpu()
                running_mae += torch.abs(predicted - labels).item()

        if was_training:
            model.train()
        mae = running_mae / len(test_loader)
    
    if testmode == 2:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        was_training = model.training
        model = model.to(device).eval()
        
        running_mae = 0.0
        test_loader = CustomImageDataset(test_)

        convertModelParametersToFixedPoint(model, bits)

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                
                fixed_point_images = realNumbersToFixedPointRepresentation(images, bits)
                fixed_point_images = fixed_point_images.to(torch.double)
                fixed_point_images = fixed_point_images.to(device)

                if len(fixed_point_images.size()) == 3:
                    fixed_point_images = fixed_point_images.unsqueeze(0)

                outputs = model(fixed_point_images)

                S = 2 ** (2 * bits + 3)
                outputs = outputs / S

                predicted = torch.max(outputs.data)
                predicted = predicted.cpu()                
                running_mae += torch.abs(predicted - labels).item()

        if was_training:
            model.train()
        mae = running_mae / len(test_loader)
    return(mae)

def convertModelParametersToFixedPoint(model, bits):
    scale_factor = 2 ** bits
    for param in model.parameters():
        param.data = torch.round(param.data * scale_factor).to(torch.double)

def realNumbersToFixedPointRepresentation(image, bits):
    scale_factor = 2 ** bits
    scaled_image = image * scale_factor
    rounded_image = torch.round(scaled_image)
    fixed_point_image = rounded_image.to(torch.int)
    return fixed_point_image  

def train(model, batch_size, optimizer, loss_fn, model_type,  train_, test_, num_epochs, trans=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_trans = len(trans) if trans else 1
     
    if trans:
        t_ = CustomImageDataset(train_)
        for t in trans:
            ds_ = CustomImageDataset(train_, img_transform=t)
            t_ = ConcatDataset([t_, ds_])
        g = torch.Generator().manual_seed(SEED)                 
        train_loader = DataLoader(t_, batch_size=batch_size, shuffle=True, generator=g)
    else:
        g = torch.Generator().manual_seed(SEED)                
        train_loader = DataLoader(CustomImageDataset(train_), batch_size=batch_size, shuffle=True, generator=g)
    
    print(f'Num Transforms: {num_trans}')
    
    for epoch in range(num_epochs):
        running_loss, run_count = 0.0, 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            images = images.to(device, dtype=torch.float64)

            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.float().unsqueeze(1)
            labels = labels.to(device)

            loss = loss_fn(outputs, labels.float())   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            run_count += 1

            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
                
        mae = testMeanAbsoluteError(model, test_, 1, 0)
        print(f"For epoch', {epoch + 1},'train MAE {running_loss / run_count} | test MAE: {mae}")
        
    saveModel(model, model_type) 
    return model

def saveModel(model, model_type='CNN'):
    path = f'{SAVE_PATH}{model_type}_ELU_{SEED}.pth'
    torch.save(model.state_dict(), path)

def loadModel(model_type='CNN'):
    model_ = run_model(model_type)[0]
    state = torch.load(f'CNN_ELU_{SEED}.pth', map_location='cpu')
    model_.load_state_dict(state)
    model_.double()           
    return model_.eval()

def run_model(model_type='CNN'):
    if model_type == 'CNN':
        bs = 128
        lr = 0.001159826
        wd = 5.02696e-05 
        epoch_ct = 24
        return Network(), bs, lr, wd, epoch_ct

def train_model():
    set_seed(SEED)

    split_size = .8
    model_type = "CNN"

    df = reload_data()

    model, batch_size, learning_rate, weight_decay, epoch_ct = run_model(model_type)
    model = model.double()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'There are {n_params} trainable parameters')
    print("Batch Size: " + str(batch_size))
    print("Learning Rate: " + str(learning_rate))
    print("Weight Decay: " + str(weight_decay))
    print("Epochs: " + str(epoch_ct))

    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_df = df.sample(frac=split_size, random_state=SEED)
    test_df = df.drop(train_df.index)

    print(f'Training Model: {model_type}')

    trained_model = train(model, batch_size, optimizer, loss_fn, model_type, train_=train_df, test_=test_df, num_epochs=epoch_ct)
    
    if activation_flag == 0:
        print("\n\nTest MAE with native ELU activation:")
    else: 
        print("\n\nTest MAE with piecewise polynomial ELU activation:")

    mae_double = testMeanAbsoluteError(trained_model, test_df, 1, 0)
    print(f"\nZIP (IEEE-754 double-precision): {mae_double}")

    final_mae_fixedpoint = testMeanAbsoluteError(trained_model, test_df, 2, 16)
    if final_mae_fixedpoint > 20.0:
        print(f"Baseline (w/FP) (fixed-point): no convergence")
    else:
        print(f"Baseline (w/FP) (fixed-point): {final_mae_fixedpoint}")

    return model

if __name__ == '__main__':
    model = train_model()
