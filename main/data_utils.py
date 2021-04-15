
""" 
Dataloaders and stuff
"""

import os 
import pickle
import torch
import torch.nn as nn 
import pandas as pd 
import numpy as np 
from tqdm import tqdm


class LSTMDataLoader:

    def __init__(self, data, labels, batch_size, shuffle=False):
        self.data = data 
        self.labels = labels
        self.batch_size = batch_size
        self.ptr = 0
        if shuffle:
            idx = np.random.permutation(np.arange(len(self.data)))
            self.data = self.data[idx]
            self.labels = self.labels[idx]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __len__(self):
        return len(self.data) // self.batch_size

    def flow(self):
        X, y = [], []
        for i in range(self.batch_size):
            X.append( torch.from_numpy(self.data[self.ptr]).float().unsqueeze(0) )
            y.append( self.labels[self.ptr] )
            self.ptr += 1
            if self.ptr >= len(self.data):
                self.ptr = 0

        X = torch.cat(X, dim=0).permute(0, 2, 1).to(self.device)
        y = torch.from_numpy(np.array(y)).long().to(self.device)
        return X, y


def get_dataloaders(file, labels, test_size, batch_size, augment=None):
    if (not os.path.exists("./data/data.pkl")) or (not os.path.exists("./data/labels.pkl")):
        file = pd.ExcelFile(file)
        data = []
        for name in tqdm(file.sheet_names, desc='Reading data'):
            df = pd.read_excel(file, name, header=None)
            data.append(df.values)
        
        if augment is not None:
            for i in range(len(data)):
                for _ in range(augment):
                    x = data[i]
                    noise = np.random.normal(loc=0.0, scale=0.01, size=x.shape)
                    data.append(x + noise)
                    labels.append(labels[i])

        data = np.asarray(data)
        labels = np.asarray(labels)
        assert len(data.shape) == 3, f'Data collection error, shape {data.shape}'

        pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
        idx = []
        for i in range(len(pos)):
            idx.append(pos[i])
            idx.append(neg[i])

        data = data[np.array(idx)]
        labels = labels[np.array(idx)]

        # Save as pkl
        os.makedirs("./data", exist_ok=True)
        with open("./data/data.pkl", "wb") as f:
            pickle.dump(data, f)
        with open("./data/labels.pkl", "wb") as f:
            pickle.dump(labels, f)
    else:
        with open("./data/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open("./data/labels.pkl", "rb") as f:
            labels = pickle.load(f)

    # Split into train and test
    train_idx = np.arange(len(labels))[:int( (1-test_size)*len(labels) )]
    test_idx = np.arange(len(labels))[int( (1-test_size)*len(labels) ):]
    X_train, y_train = data[train_idx], labels[train_idx]
    X_test, y_test = data[test_idx], labels[test_idx]

    print("\n[INFO] Data statistics")
    print("\tTraining: {} damaged, {} undamaged".format(sum(y_train == 1), sum(y_train == 0)))
    print("\tTesting: {} damaged, {} undamaged".format(sum(y_test == 1), sum(y_test == 0)))

    # Loaders
    train_loader = LSTMDataLoader(X_train, y_train, batch_size, shuffle=True)
    test_loader = LSTMDataLoader(X_test, y_test, batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":

    labels = [0]*50 + [1]*50
    train_loader, test_loader = get_dataloaders("../project.xlsx", labels, 0.2, 5)

    X_train, y_train = train_loader.flow()
    X_test, y_test = test_loader.flow() 

    print(f"Train batch shapes: {X_train.shape} {y_train.shape}")
    print(f"Test batch shapes: {X_test.shape} {y_test.shape}")
