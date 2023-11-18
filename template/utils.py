import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import os
from tqdm import tqdm

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def plot_loss(train_loss_list = [], val_loss_list = []):
    """Plots train and validation loss.

    Args:
        train_loss_list (list, optional): Training loss. Defaults to [].
        val_loss_list (list, optional): Validation loss. Defaults to [].
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    smooth_train_loss = smooth(train_loss_list, 31)
    smooth_val_loss = smooth(val_loss_list, 31)

    ax[0].plot(train_loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_train_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[0].plot(val_loss_list, c="green", label="Test Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_val_loss, c="yellow", label="Smoothed Test Loss", linewidth=3)


    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress (linearscale)")

    ax[1].plot(train_loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[1].plot(val_loss_list, c="green", label="Test Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_val_loss, c="yellow", label="Smoothed Test Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (logscale)")
    plt.show()


def plot_loss_epoch(train_loss, test_loss):
    """
    Plot training and test loss per epoch.

    Args:
        train_loss (list): List of training loss values per epoch.
        test_loss (list): List of test loss values per epoch.
    """

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, test_loss, 'r*-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_epoch(test_accuracy, train_accuracy = []):
    """
    Plot training and test accuracy per epoch.

    Args:
        train_accuracy (list): List of training accuracy values per epoch.
        test_accuracy (list): List of test accuracy values per epoch.
    """
    epochs = range(1, len(test_accuracy) + 1)

    plt.figure(figsize=(10, 5))
    
    if train_accuracy:
        plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
        plt.title('Training and Test Accuracy')

    
    plt.plot(epochs, test_accuracy, 'r*-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(train_accuracy = [], val_accuracy = []):
    """Plots accuracy for training and validation data

    Args:
        train_accuracy (list, optional): Accuracy for training dataset. Defaults to [].
        val_accuracy (list, optional): Accuarcy for validation dataset. Defaults to [].
    """
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    smooth_val_accuracy = smooth(val_accuracy, 31)
    smooth_train_accuracy = smooth(train_accuracy, 31)

    ax[0].plot(val_accuracy, c="blue", label="Testing Accuracy", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_val_accuracy, c="red", label="Smoothed Testing Accuracy", linewidth=3, alpha=0.5)



    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Testing Accuracy")
    ax[0].set_title("Testing Accuracy Change")


    ax[1].plot(train_accuracy, c="blue", label="Training Accuracy", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train_accuracy, c="red", label="Smoothed Training Accuracy", linewidth=3, alpha=0.5)

    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Training Accuracy")
    ax[1].set_title("Training Accuracy Change")

    plt.show()

def plot_gradients(norms, max_values, min_values):
    """PLots related to gradient

    Args:
        norms (list): Gradient norms
        max_values (list): Max Gradient Values
        min_values (list): Min Gradient values
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(18, 5)

    ax[0].plot(norms, c="blue", label="Gradient Norms", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Gradient Norms")
    ax[0].set_title("Gradient Norms Change")

    ax[1].plot(max_values, c="red", label="Max Gradient Values", linewidth=3, alpha=0.5)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Max Gradient Values")
    ax[1].set_title("Max Gradient Values Change")

    ax[2].plot(min_values, c="green", label="Min Gradient Values", linewidth=3, alpha=0.5)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Min Gradient Values")
    ax[2].set_title("Min Gradient Values Change")

    plt.show()



def set_random_seeds(seed_value=42, use_gpu=False):
    # Set the random seed for Python's random number generator
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed_value)

    # Set the seed for PyTorch's random number generator on the CPU
    torch.manual_seed(seed_value)

    if use_gpu and torch.cuda.is_available():
        # Set the seed for PyTorch's random number generator on the GPU
        torch.cuda.manual_seed_all(seed_value)
        
    # Ensure determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

@torch.no_grad()
def eval_model(model, eval_loader, criterion, device):
    """ 
    Evaluating the model for either validation or test.

    Parameters:
    - model (nn.Module): The trained neural network model to be evaluated.
    - eval_loader (torch.utils.data.DataLoader): DataLoader for the validation or test dataset.
    - criterion (nn.Module): The loss function.
    - device (torch.device): The device on which to perform evaluation (e.g., 'cuda' or 'cpu').

    Returns:
    - float: Accuracy on the validation or test dataset.
    - float: Mean loss on the validation or test dataset.
    """    
    correct = 0
    total = 0
    loss_list = []
    
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass only to get logits/output
        outputs = model(images)
                 
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
            
        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len( torch.where(preds==labels)[0] )
        total += len(labels)
                 
    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    
    return accuracy, loss

def display_images(dataset, NUM_images = 8):
    N_IMGS = 8
    fig, ax = plt.subplots(1,N_IMGS)
    fig.set_size_inches(3 * N_IMGS, 3)

    ids = np.random.randint(low=0, high=len(dataset), size=N_IMGS)

    for i, n in enumerate(ids):
        img = dataset[n][0].numpy().reshape(3,224, 224).transpose(1, 2, 0)
        ax[i].imshow(img)
        ax[i].set_title(f"Img #{n}  Label: {dataset[n][1]}")
        ax[i].axis("off")
    plt.show()

def train_epoch(model, train_loader, optimizer, criterion, device, mixup = None):

    """ 
    Training a model for one epoch.

    Parameters:
    - model (nn.Module): The neural network model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - criterion (nn.Module): The loss function.
    - epoch (int): Current epoch number.
    - device (torch.device): The device on which to perform training (e.g., 'cuda' or 'cpu').

    Returns:
    - float: Mean training loss for the epoch.
    - list: List of individual training losses for each batch.
    """
    
    loss_list = []
    for i, (images, labels) in enumerate(train_loader):
        if mixup:
            images, labels = mixup(images, labels)
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(images)
         
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list