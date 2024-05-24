import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def add_noise(p_grad, multiplier, secure_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = torch.norm(p_grad)
    summed_noise = 0
    num_noise = 4 if secure_mode else 1
    for i in range(num_noise):
        noise = torch.normal(mean=0, std=multiplier*norm, size=p_grad.shape)
        summed_noise += noise
    if secure_mode:
        summed_noise /= 2
    return summed_noise.to(device) + p_grad

def to_torch(X):
    if isinstance(X, torch.Tensor):
        return X
    if isinstance(X, pd.Series):
        return torch.from_numpy(X.values).float()
    return torch.from_numpy(X).float()

class CustomDataset(Dataset):
    def __init__(self, X, y, weights):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = to_torch(X).to(device)
        self.y = to_torch(y).to(device)
        self.weights = to_torch(weights).to(device)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=100, num_layers=3, output_logits=False):
        super(Model, self).__init__()
        activation = torch.nn.Sigmoid if output_logits else torch.nn.ReLU
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_dim))
        layers.append(activation())
        for _ in range(num_layers-2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())            
        layers.append(torch.nn.Linear(hidden_dim, 1))    
        if output_logits:
            layers.append(activation())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train(X_train, y_train, X_test=None, weights=None, lr=.001, epochs=200, return_model=False, clip_value=None, noise_multiplier=None, is_bce=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if weights is None:
        weights = np.ones(X_train.shape[0])

    dataset = CustomDataset(X_train, y_train, weights)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    loss = lambda y_true, y_pred, weights: ((y_true - y_pred) ** 2 * weights).mean()
    if is_bce:
        loss = lambda y_true, y_pred, weights: torch.nn.BCELoss()(y_pred, y_true)

    model = Model(X_train.shape[1], hidden_dim=100, output_logits=is_bce).to(device)

    for p in model.parameters():
        if clip_value is not None:
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        if noise_multiplier is not None:
            p.register_hook(lambda grad: add_noise(grad, noise_multiplier, secure_mode=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X, y, w in dataloader:
            y_pred = model(X).squeeze()
            loss_value = loss(y, y_pred, w)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

    easy_model = lambda X : model(to_torch(X).to(device)).detach().squeeze().cpu().numpy()
    if return_model:
        return easy_model
    return easy_model(X_test)

def estimate_propensity(X, treatment):
    propensity = train(X, treatment, X, is_bce=True)
    # Clip propensity to avoid numerical instability
    return propensity.clip(.01, .99)