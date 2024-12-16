"""
Brief example pf dataloader and dataset from appendix A.6
"""

import  torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index): #A
        one_x = self.features[index] #A
        one_y = self.labels[index] #A
        return one_x, one_y #A
    def __len__(self):
        return self.labels.shape[0] #B

X_train = torch.tensor([
[-1.2, 3.1],
[-0.9, 2.9],
[-0.5, 2.6],
[2.3, -1.1],
[2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
[-0.8, 2.8],
[2.6, -1.6],
])
y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)


'''
A fixed seed ensures that the shuffling order for the first epoch is reproducible, 
but it does not lock the order across epochs. 
PyTorch reuses the random number generator for each epoch, introducing fresh randomness for reshuffling.
'''
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_ds, #A
    batch_size=2,
    shuffle=True, #B
    num_workers=0, #C
    drop_last=True          # remove the batch with fewer data
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False, #D
    num_workers=0
)