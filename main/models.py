
""" 
Model definitions
"""

import os
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import data_utils
import common
import sklearn 
from sklearn import metrics


COLORS = {
    "yellow": "\x1b[33m", 
    "blue": "\x1b[94m", 
    "green": "\x1b[32m", 
    "end": "\033[0m"
}


def get_metrics(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    acc = metrics.accuracy_score(target, output)
    recall = metrics.recall_score(target, output)
    precision = metrics.precision_score(target, output, zero_division=0)
    f1 = metrics.f1_score(target, output, zero_division=0)
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


class LSTMNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, _ = self.lstm(x)                         # (batch_size, seq_length, hidden_size)
        out = x[:, -1, :]                           # (batch_size, hidden_size)
        out = self.fc_out(out)                      # (batch_size, 2)
        return F.log_softmax(out, dim=-1)


class RecurrentModel:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = LSTMNetwork(config['input_size'], config['hidden_size'], config['num_layers']).to(self.device)
        self.train_loader, self.val_loader = data_utils.get_dataloaders(
            file = config['data_root'], 
            labels = [0]*50 + [1]*50, 
            test_size = config['test_size'], 
            batch_size = config['batch_size'],
            augment = config['augment']
        )
        self.criterion = nn.NLLLoss()
        self.best_val_acc = 0
        self.optim = optim.Adam(self.network.parameters(), lr=config['optim_lr'])
        os.makedirs('./outputs/lstm', exist_ok=True)

        if torch.cuda.is_available():
            print("\n[INFO] Device found: {}".format(torch.cuda.get_device_name(0)))
        else:
            print("\n[INFO] Device found: {}".format('cpu'))

    def train_one_step(self, batch):
        inp, trg = batch 
        output = self.network(inp)
        loss = self.criterion(output, trg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        preds = output.argmax(dim=-1)
        metrics = get_metrics(preds, trg)
        return {"loss": loss.item(), **metrics}

    def validate_one_step(self, batch):
        inp, trg = batch 
        with torch.no_grad():
            output = self.network(inp)
        loss = self.criterion(output, trg)
        preds = output.argmax(dim=-1)
        metrics = get_metrics(preds, trg)
        return {"loss": loss.item(), **metrics}

    def save_state(self, epoch):
        state = {'epoch': epoch, 'model': self.network.state_dict(), 'optim': self.optim.state_dict()}
        torch.save(state, os.path.join('./outputs/lstm', 'last_state.ckpt'))

    def save_model(self):
        torch.save(self.network.state_dict(), os.path.join('./outputs/lstm', 'best_model.ckpt'))

    def train(self):
        print("[INFO] Beginning training!\n")
        for epoch in range(self.config['epochs']):
            train_meter = common.AverageMeter()
            val_meter = common.AverageMeter()

            for step in tqdm(range(len(self.train_loader)), desc='[TRAIN] Epoch {:2d}'.format(epoch+1), leave=False):
                batch = self.train_loader.flow()
                train_metrics = self.train_one_step(batch)
                train_meter.add(train_metrics)
            print("[TRAIN] Epoch {:2d}: {}".format(epoch+1, train_meter.return_msg()))
            self.save_state(epoch+1)
        
            if (epoch+1) % self.config['eval_every'] == 0:
                for step in tqdm(range(len(self.val_loader)), desc='[VALID] Epoch {:2d}'.format(epoch+1), leave=False):
                    batch = self.val_loader.flow()
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                print("{}[VALID] Epoch {:2d}: {}{}".format(COLORS['blue'], epoch+1, val_meter.return_msg(), COLORS['end']))

                # Save model 
                val_metrics = val_meter.return_metrics()
                if val_metrics['acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['acc']
                    self.save_model()

        print("\n[INFO] Training complete!")