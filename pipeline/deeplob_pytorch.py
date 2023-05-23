import sys
import os
import argparse
from torchsummary import summary

## Path adjustment
sys.path.append("..")
cwd = os.getcwd()
from pathlib import Path
home = str(Path.home())
sys.path.append(home + "/ProdigyAI")

import h5py

import wandb

import torch

import numpy as np
import yaml

### Ipython and Argparse setup STARTED
try:
    resuming = "NA"
    get_ipython()
    check_if_ipython = True
    path_adjust = "../"
    import getpass

    user = getpass.getuser()
except Exception as e:  ## If not using Ipython kernel deal with any argparse's
    check_if_ipython = False
    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)

    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("-s",
                        "--stage",
                        type=str,
                        help="Stage of Preprocesssing")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        help="one_model or two_model")
    parser.add_argument(
        "-f",
        "--resuming",
        type=str,
        help="Is this a continuation of preempted instance?",
    )
    parser.add_argument("-u",
                        "--user",
                        type=str,
                        help="Stage of Preprocesssing")
    args = parser.parse_args()
    if args.user != None:
        user = args.user
    else:
        import getpass
        user = getpass.getuser()
    if args.resuming != None:
        resuming = args.resuming
    else:
        resuming = "NA"
    if args.stage != None:
        arg_parse_stage = 1
        if int(args.stage) == 1:
            if os.path.exists(path_adjust + "temp/data_name_gpu.txt"):
                os.remove(path_adjust + "temp/data_name_gpu.txt")
                print("removed temp/data_name_gpu.txt")
            else:
                print("The file does not exist")

    if args.model != None:
        model = args.model
    path_adjust = ""

if cwd == home + "/":
    cwd = cwd + "/ProdigyAI"
    path_adjust = cwd
### Ipython and Argparse setup FINISHED

### WANDB SETUP STARTED
yaml_path = path_adjust + "yaml/deeplob.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

config_dictionary = dict(yaml=yaml_path, params=yaml_dict)

## Check if there is a run in progress
try:
    with open(path_adjust + "temp/deeplob_run_in_progress.txt",
              "r") as text_file:
        stored_id = text_file.read()
    resume = True
    if resuming == "resuming":
        resume = "allow"
        wandb_id = stored_id
    else:
        wandb_id = wandb.util.generate_id()
        resume = "allow"
except:
    resume = False

wandb.init(
    dir="/home/" + user + "/ProdigyAI/",
    project="prodigyai",
    config=config_dictionary,
    resume=resume,
    entity=user,
    id=wandb_id,
)
wandb.save("*.pt")

### WANDB SETUP FINISHED

### CUDA for PyTorch STARTED
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
### CUDA for PyTorch FINISHED
# fi-2010-lob start
print("fi-2010 load started")
path = home + "/ProdigyAI/data/lob_2010/train_and_test_deeplob.h5"
h5f = h5py.File(path, "r")
trainX_CNN = h5f["trainX_CNN"][:]
trainY_CNN = h5f["trainY_CNN"][:]
testX_CNN = h5f["testX_CNN"][:]
testY_CNN = h5f["testY_CNN"][:]
print("fi-2010 load finished")

## fi-2010-lob finished

## Load the data STARTED
file_name = wandb.config["params"]["dataset"]["value"]
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
prices_for_window_index_array_train = h5f[
    "prices_for_window_index_array_train"][:]
prices_for_window_index_array_val = h5f["prices_for_window_index_array_val"][:]
prices_for_window_index_array_test = h5f[
    "prices_for_window_index_array_test"][:]
input_features_normalized_train = h5f["input_features_normalized_train"][:]
input_features_normalized_val = h5f["input_features_normalized_val"][:]
input_features_normalized_test = h5f["input_features_normalized_test"][:]
y_train = h5f["y_train"][:].astype(np.int8)
y_val = h5f["y_val"][:].astype(np.int8)
y_test = h5f["y_test"][:].astype(np.int8)
h5f.close()
### Load the data FINISHED

from numba import njit, prange


@njit(parallel=True)
def generate_x_numba(num_features, window_length, features, index):
    X = np.empty((num_features, window_length))
    for i in prange(num_features):
        for j in prange(window_length):
            X[i][j] = features[i][index - j]

    return X


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels, features):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.features = features
        self.window_length = len(self.features[0]) - len(self.labels)
        self.num_features = len(self.features)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # fi-lob-2010
        # y = self.labels[index]
        # X = self.features[index]  ## (100, 40, 1)
        # X = X.reshape((1, X.shape[0], X.shape[1]))

        ## crypto-data
        y = self.labels[index]
        X = generate_x_numba(self.num_features, self.window_length,
                             self.features, index)

        X = np.swapaxes(X, 0, 1)
        X = X.reshape((1, X.shape[0], X.shape[1]))  # (1, 100, 40)

        return X, y


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input_shape = (64, 100, 40, 1)
        ## build the convolutional block
        self.conv1 = nn.Conv2d(1, 32, (1, 2), stride=(1, 2))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(32, 32, (4, 1))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(32, 32, (4, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(32, 32, (1, 2), stride=(1, 2))
        nn.init.xavier_uniform_(self.conv4.weight)
        self.relu4 = nn.LeakyReLU(0.1)
        self.conv5 = nn.Conv2d(32, 32, (4, 1))
        nn.init.xavier_uniform_(self.conv5.weight)
        self.relu5 = nn.LeakyReLU(0.1)
        self.conv6 = nn.Conv2d(32, 32, (4, 1))
        nn.init.xavier_uniform_(self.conv6.weight)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(32, 32, (1, 10), stride=(1, 2))
        nn.init.xavier_uniform_(self.conv7.weight)
        self.relu7 = nn.LeakyReLU(0.1)
        self.conv8 = nn.Conv2d(32, 32, (4, 1))
        nn.init.xavier_uniform_(self.conv8.weight)
        self.relu8 = nn.LeakyReLU(0.1)
        self.conv9 = nn.Conv2d(32, 32, (4, 1))
        nn.init.xavier_uniform_(self.conv9.weight)
        self.relu9 = nn.LeakyReLU(0.1)

        # Build the inception module
        self.conv10 = nn.Conv2d(32, 64, (1, 1))
        nn.init.xavier_uniform_(self.conv10.weight)
        self.relu10 = nn.LeakyReLU(0.1)
        self.conv11 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        nn.init.xavier_uniform_(self.conv11.weight)
        self.relu11 = nn.LeakyReLU(0.1)

        self.conv12 = nn.Conv2d(32, 64, (1, 1))
        nn.init.xavier_uniform_(self.conv12.weight)
        self.relu12 = nn.LeakyReLU(0.1)
        self.conv13 = nn.Conv2d(64, 64, (5, 1), padding=(2, 0))
        nn.init.xavier_uniform_(self.conv13.weight)
        self.relu13 = nn.LeakyReLU(0.1)

        self.maxpool1 = nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0))
        self.conv14 = nn.Conv2d(32, 64, (1, 1))
        nn.init.xavier_uniform_(self.conv14.weight)
        self.rnn1 = nn.LSTM(100, 64)
        self.linear1 = nn.Linear(64, 3)
        self.softmax1 = nn.Softmax(dim=1)

        ## testing layers
        # self.linear1 = nn.Linear(3200, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # bias init code
                BIAS_INIT = 0
                m.bias.data.fill_(BIAS_INIT)

    def forward(self, x):

        ### build the convolutional block ###
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        # x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        ### SHOULD I USE 1,2 or 2,1 padding?
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        x = self.conv5(x)
        x = self.relu5(x)
        x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        x = self.conv8(x)
        x = self.relu8(x)
        x = F.pad(x, (0, 0, 1, 2))  # [left, right, top, bot]
        x = self.conv9(x)
        x = self.relu9(x)

        ## testing layers
        # x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        # x = self.linear1(x)
        # x = self.softmax1(x)

        # ### Build the inception module
        x1 = self.conv10(x)
        x1 = self.relu10(x1)
        x1 = self.conv11(x1)
        x1 = self.relu11(x1)

        x2 = self.conv12(x)
        x2 = self.relu12(x2)
        x2 = self.conv13(x2)
        x2 = self.relu13(x2)

        x3 = self.maxpool1(x)
        x3 = self.conv14(x3)

        x4 = torch.cat((x1, x2, x3), 1)
        x4 = x4.view(x4.shape[0], x4.shape[1], x4.shape[2])
        x4, (h, c) = self.rnn1(x4)
        x4 = x4[:, -1, :]
        x4 = self.linear1(x4)
        # x4 = self.softmax1(x4)

        return x4


### simple model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # input_shape = (64, 100, 40, 1)
#         self.linear1 = nn.Linear(4000, 3)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
#         x = self.linear1(x)
#         x_out = self.softmax(x)
#         return x_out

net = Net()
if torch.cuda.is_available():
    net.cuda()

# summary(net, (100, 40, 1))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01, eps=1)

if resuming == "resuming":
    try:
        # restore model
        checkpoint_path = wandb.restore('model.pt',
                                        run_path=user + '/prodigyai/' +
                                        wandb_id).name
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("no prior epochs completed")

# Parameters
params = {'batch_size': 64, 'shuffle': True, 'num_workers': 1}
max_epochs = 100

#crypto-data
list_IDs = list(range(len(y_train)))
training_set = Dataset(list_IDs, y_train, input_features_normalized_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

list_IDs = list(range(len(y_val)))
validation_set = Dataset(list_IDs, y_val, input_features_normalized_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

#fi-lob-2010
# list_IDs = list(range(len(trainY_CNN)))
# training_set = Dataset(list_IDs, trainY_CNN, trainX_CNN)
# training_generator = torch.utils.data.DataLoader(training_set, **params)

# list_IDs = list(range(len(testY_CNN)))
# testing_set = Dataset(list_IDs, testY_CNN, testX_CNN)
# testing_generator = torch.utils.data.DataLoader(testing_set, **params)

# Log metrics with wandb
wandb.watch(net)

# Save to in progress temp file
with open(path_adjust + "temp/deeplob_run_in_progress.txt", "w+") as text_file:
    text_file.write(wandb_id)

batch_count = 0
# Loop over epochs
for epoch in range(max_epochs):
    print("epoch num=" + str(epoch))
    # Training
    batch_count = 0
    for local_batch, local_labels in training_generator:
        if local_batch.shape[2] == 100:

            # if batch_count == 0:
            #     first_local_batch = local_batch[:]
            #     first_local_labels = local_labels[:]
            #     # first_local_batch = torch.unsqueeze(first_local_batch, 0)
            #     # first_local_labels = torch.unsqueeze(first_local_labels, 0)

            # local_batch = first_local_batch
            # local_labels = first_local_labels

            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            if epoch > 0 and batch_count == 0:
                outputs = net(local_batch.float())
                print(local_labels)
                print(outputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch.float())

            local_labels_integer_tensor = torch.argmax(local_labels, dim=1)

            loss = criterion(outputs, local_labels_integer_tensor)
            loss.backward()
            optimizer.step()
            wandb.log({"Train Loss": loss})
            batch_count = batch_count + 1
    del outputs
    del local_batch
    del local_labels
    del local_labels_integer_tensor
    del loss
    torch.cuda.empty_cache()

    total_test_loss_sum = 0
    batch_count = 0
    for local_batch, local_labels in validation_generator:
        if local_batch.shape[2] == 100:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(
                device), local_labels.to(device)

            outputs = net(local_batch.float())
            local_labels_integer_tensor = torch.argmax(local_labels, dim=1)
            loss = criterion(outputs, local_labels_integer_tensor)
            total_test_loss_sum = int(total_test_loss_sum) + loss
            batch_count = batch_count + 1

    total_test_loss = total_test_loss_sum / batch_count

    del outputs
    del local_batch
    del local_labels
    del local_labels_integer_tensor

    torch.cuda.empty_cache()

    total_test_loss = total_test_loss_sum / batch_count
    print(total_test_loss)
    wandb.log({"Test Loss": total_test_loss})

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(wandb.run.dir, 'model.pt'))
    wandb.save(os.path.join(wandb.run.dir, "model.pt"))
