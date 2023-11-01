import numpy as np
import pandas as pd
import os
from collections import OrderedDict
import time
import copy
import matplotlib.pyplot as plt
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Testing the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ------------------ HELPER FUNCTIONS ------------------
def numpy_loader(path):
    """"
    Reads a numpy array and transforms it into a Tensor
    Args:
        path (str): path of Numpy array
    """
    with open(path, 'rb') as f:
        np_array = np.load(f)
        tensor = torch.tensor(np_array)
        tensor = torch.reshape(tensor, (1, 1, -1))
        tensor = tensor.to(torch.float)
        return tensor


class ComplexReLU(nn.Module):
    """
    Applies the rectified linear unit function element-wise: ReLU(Re(x)) + j ReLU(Im(x))
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: (*), where * means any number of dimensions.
        - Output: (*), same shape as the input.
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(torch.real(input), inplace=self.inplace) + 1j * F.relu(torch.imag(input), inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class ComplexMSE(nn.Module):
    """
    Applies the Mean Squared Error loss for complex tensors
    Shape:
        - Input and Target should be of the same dimensions
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(target.size(), input.size()),
                stacklevel=2,
            )
        return torch.mean(torch.abs((input - target)**2))


# Takes in a module and applies the specified weight initialization -- from ashunigion in stackoverflow
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # For every Linear layer in a model
    if classname.find('Linear') != -1:
        # Get the number of the inputs
        n = m.in_features
        # General rule for applying uniform weights is 1 / sqrt(n)
        y = 0.1
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def visualize_data(data, targets, predictions=None, dz=0.930854):
    """
    Visualizes a batch of data from the NN
    Arguments:
        data (PyTorch Tensor): BxCxWxH tensor of A-line scans
        targets (PyTorch Tensor): BxCxWxH tensor of Electric Permittivity Profile
        predictions (Optional PyTorch Tensor): BxCxWxH tensor of the predicted Electric Permittivity profiles
    """
    # Useful constants
    rows = data.shape[0]
    if rows > 4:
        rows = 4
    len_targets = targets.size(dim=-1)
    len_data = len(data[0].squeeze().cpu().numpy())

    # Deciding how many columns the visualization grid should have based on the inputs to the function
    if predictions is not None:
        cols = 3
    else:
        cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))

    # Plots for every element in the batch
    for i in range(rows):
        # Plot the A-Line
        if rows > 1:
            axs[i, 0].plot(dz * np.arange(-int(len_data/2), int(len_data/2)+1), data[i].squeeze().cpu().numpy())
        else:
            axs[0].plot(dz * np.arange(-int(len_data/2), int(len_data/2)+1), data[i].squeeze().cpu().numpy())
        # Plot where the prediction is for a 1 dimensional classifier
        if predictions is not None and len_targets == 1:
            if rows > 1:
                axs[i, 0].axvline(x=targets[i].detach().squeeze().cpu().numpy(), color='r', label='Prediction')
                axs[i, 0].legend()
            else:
                axs[0].axvline(x=targets[i].detach().squeeze().cpu().numpy(), color='r', label='Prediction')
                axs[0].legend()

        # Plot the Target depending on what it is
        if len_targets == 3:
            if rows > 1:
                b1 = axs[i, 1].bar(["er1", "er2", "z"], targets[i].squeeze().cpu().numpy())
                axs[i, 1].bar_label(b1, padding=3)
                axs[i, 1].set_ylim(0, 45)
            else:
                b1 = axs[1].bar(["er1", "er2", "z"], targets[i].squeeze().cpu().numpy())
                axs[1].bar_label(b1, padding=3)
                axs[1].set_ylim(0, 45)
        elif len_targets == 1:
            if rows > 1:
                b1 = axs[i, 1].bar(["z"], targets[i].squeeze().cpu().numpy())
                axs[i, 1].bar_label(b1, padding=3)
                axs[i, 1].set_ylim(0, 45)
            else:
                b1 = axs[1].bar(["z"], targets[i].squeeze().cpu().numpy())
                axs[1].bar_label(b1, padding=3)
                axs[1].set_ylim(0, 45)
        else:
            if rows > 1:
                axs[i, 1].plot(targets[i].squeeze().cpu().numpy())
            else:
                axs[1].plot(targets[i].squeeze().cpu().numpy())

        # Plotting the predictions
        if predictions is not None:
            if len_targets == 3:
                if rows > 1:
                    b2 = axs[i, 2].bar(["er1", "er2", "z"], predictions[i].detach().squeeze().cpu().numpy())
                    axs[i, 2].bar_label(b2, padding=3)
                    axs[i, 2].set_ylim(0, 45)
                else:
                    b2 = axs[2].bar(["er1", "er2", "z"], predictions[i].detach().squeeze().cpu().numpy())
                    axs[2].bar_label(b2, padding=3)
                    axs[2].set_ylim(0, 45)
            elif len_targets == 1:
                if rows > 1:
                    b2 = axs[i, 2].bar(["z"], targets[i].squeeze().cpu().numpy())
                    axs[i, 2].bar_label(b2, padding=3)
                    axs[i, 2].set_ylim(0, 45)
                else:
                    b2 = axs[2].bar(["z"], targets[i].squeeze().cpu().numpy())
                    axs[2].bar_label(b2, padding=3)
                    axs[2].set_ylim(0, 45)
            else:
                if rows > 1:
                    axs[i, 2].plot(predictions[i].detach().squeeze().cpu().numpy())
                else:
                    axs[2].plot(predictions[i].detach().squeeze().cpu().numpy())

    # Add titles
    if rows > 1:
        fig.suptitle("Data Visualization")
        axs[0, 0].set_title("A-Line")
        if len_targets == 3:
            axs[0, 1].set_title("Parameters to be estimated")
        else:
            axs[0, 1].set_title("Electric permittivity")
        if predictions is not None:
            axs[0, 2].set_title("Predicted Parameter(s)")
        plt.show()
    else:
        fig.suptitle("Data Visualization")
        axs[0].set_title("A-Line")
        if len_targets == 3:
            axs[1].set_title("Parameters to be estimated")
        else:
            axs[1].set_title("Electric permittivity")
        if predictions is not None:
            axs[2].set_title("Predicted Parameter(s)")
        plt.show()


# ------------------ DATASET ------------------
class ALineDataset(Dataset):
    """
    Creates a dataset where the data is a one-dimensional OCT A-line and the target is the relative Electric
    Permittivity (ER) of the tissue being modelled
    Args:
        annotations_file (csv): contains a file with the id of the file
        data_dir (str): Directory where the A-line files are located. The file format should be of the form "DaD###.npy"
                        where ### is the data id in 3 digits
        target_dir (str): Directory where the A-line files are located. The file format should be of the form "###.npy"
                        where ### is the data id in 3 digits
        transform: A PyTorch transformation for the data
        target_transform: Pytorch transform for the targets
    """

    def __init__(self, annotations_file, data_dir, target_dir, transform=None, target_transform=None):
        self.annotations_file = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        data_filename = "{0}.npy".format(int(self.annotations_file.iloc[idx, 0]))
        data_path = os.path.join(self.data_dir, data_filename)
        data = numpy_loader(data_path)
        # data = nn.functional.pad(data, (28, 29), mode='replicate')
        target_filename = "{0}.npy".format(int(self.annotations_file.iloc[idx, 0]))
        target_path = os.path.join(self.target_dir, target_filename)
        target = numpy_loader(target_path)
        # target = torch.tensor([self.annotations_file.loc[idx, "er"], self.annotations_file.loc[idx, "er2"],
        #                        self.annotations_file.loc[idx, "z"]])
        # target = torch.Tensor([self.annotations_file.loc[idx, "z"]])
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target


# ------------------ MODEL ------------------
class MLP(nn.Module):
    """
    Multi-Layer Perceptron based on Pytorch's Module and Sequential classes. The MLP consists of a customizable number
    of hidden layers, each with a number of hidden nodes, followed by a ReLU after each dense layer except for the
    last dense layer
    Arguments:
        input_dim (int): number of features d in input data of shape 1xd (for batch b it would be bxd)
        output_dim (int): number of output features 1xn
        num_hidden_layers (int): Number of hidden layers to be appended into the model
        num_hidden_nodes (int): Number of hidden nodes per each layer
    """

    def __init__(self, input_dim, output_dim, num_hidden_layers, num_hidden_nodes, data_type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_hidden_layers
        self.num_hidden = num_hidden_nodes
        self.is_complex = torch.is_complex(torch.ones(1).to(data_type))
        layers = [('dense1', nn.Linear(self.input_dim, self.num_hidden).to(data_type))]
        layers.append(('batchnorm1', nn.BatchNorm2d(1).to(data_type)))
        if self.is_complex:
            layers.append(('relu1', ComplexReLU()))
        else:
            layers.append(('relu1', nn.ReLU()))
        for i in range(2, self.num_layers):
            layers.append((f'dense{i}', nn.Linear(self.num_hidden, self.num_hidden).to(data_type)))
            layers.append((f'batchnorm{i}', nn.BatchNorm2d(1).to(data_type)))
            if self.is_complex:
                layers.append((f'relu{i}', ComplexReLU()))
            else:
                layers.append((f'relu{i}', nn.ReLU()))
        layers.append((f'dense{self.num_layers}', nn.Linear(self.num_hidden, self.output_dim).to(data_type)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


# ------------------ TRAINING AND TESTING FUNCTIONS ------------------
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100):
    """
    Training loop for the model. It automatically shifts between training and testing stages
    Arguments:
        model: A PyTorch Neural Network
        criterion: PyTorch loss
        optimizer: PyTorch Optimizer
        scheduler: PyTorch Scheduler
        dataloaders: A dictionary {'test': [], 'train':[]} of dataloaders
        dataset_sizes: A dictionary {'test': [], 'train':[]} of dataset sizes
        num_epochs (int): The number of epochs to train the model
    """
    # Start training time
    since = time.time()

    # Retrieving best weights
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loss
    losses = {'train': [], 'test': []}
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'test' and best_loss > epoch_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    # Time elapsed
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Visualizing the loss
    plt.plot(range(num_epochs), np.log(losses["train"]), label="Train Loss")
    plt.legend()
    plt.title("Model Train Performance")
    plt.ylabel("Log loss")
    plt.xlabel("Epochs")
    plt.show()

    plt.plot(range(num_epochs), np.log(losses["test"]), label="Test Loss")
    plt.legend()
    plt.title("Model Test Performance")
    plt.ylabel("Log Loss")
    plt.xlabel("Epochs")
    plt.show()

    # load best model weights
    # model.load_state_dict(best_model_wts)

    return model, best_model_wts


# ------------------ MAIN ------------------

def main():
    # Creating the datasets and dataloaders
    dataset_dir = "Dataset_Split_1Layer"
    data_dir = "data"
    target_dir = "target"
    model_weight_file = 'Models/model_weights_ali_er_2l_512n_layers_BN_1Layer.pth'
    annotations_file = {"train": os.path.join(dataset_dir, "train", "annotations_train.csv"),
                        "test": os.path.join(dataset_dir, "test", "annotations_test.csv")}
    task_datasets = {x: ALineDataset(annotations_file[x], os.path.join(dataset_dir, x, data_dir),
                                     os.path.join(dataset_dir, x, target_dir)) for x in ["train", "test"]}

    dataloaders = {x: DataLoader(task_datasets[x], batch_size=16, shuffle=True) for x in ["train", "test"]}
    dataset_sizes = {x: len(task_datasets[x]) for x in ['train', 'test']}

    # Getting a batch of training data
    inputs, targets = next(iter(dataloaders['train']))

    # Visualizing the data
    visualize_data(inputs, targets)

    # Getting the sizes and datatypes for the model
    input_dim = inputs[0].shape[-1]
    output_dim = targets[0].shape[-1]
    data_type = inputs[0].dtype
    is_complex = torch.is_complex(inputs[0])
    num_hidden_layers = 2
    num_hidden_nodes = 512

    # Model
    model = MLP(input_dim, output_dim, num_hidden_layers, num_hidden_nodes, data_type)
    # model = UNet(up_mode='upconv')

    # Weight initalization
    model.apply(weights_init_uniform_rule)

    # Send model to device
    model.to(device)

    # Visualizing the best weights model if it already exists
    if os.path.isfile(model_weight_file):
        model.load_state_dict(torch.load(model_weight_file))
        model.eval()
        # Fit on training data
        inputs, targets = next(iter(dataloaders['train']))
        predictions = model(inputs.to(device))
        visualize_data(inputs, targets, predictions)
        # Fit on test data
        inputs, targets = next(iter(dataloaders['test']))
        predictions = model(inputs.to(device))
        visualize_data(inputs, targets, predictions)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Loss function
    criterion = ComplexMSE() if is_complex else nn.MSELoss()

    # Step Function
    step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # Training
    model.train()
    model, best_model_wts = train_model(model, criterion, optimizer, step_lr, dataloaders, dataset_sizes, 2000)

    # Save model weights that perform best on test dataset
    torch.save(best_model_wts, model_weight_file)

    # Visualize predictions on the train set
    model.eval()
    inputs, targets = next(iter(dataloaders['train']))
    predictions = model(inputs.to(device))
    visualize_data(inputs, targets, predictions)

    # Visualize predictions on the test set
    model.load_state_dict(best_model_wts)
    model.eval()
    inputs, targets = next(iter(dataloaders['test']))
    predictions = model(inputs.to(device))
    visualize_data(inputs, targets, predictions)




if __name__ == '__main__':
    main()
