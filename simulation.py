import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# Configuration & Hyperparameters (UPDATED)

NUM_CLIENTS = 100         
CLIENTS_PER_ROUND = 10    
TOTAL_ROUNDS = 30         
LOCAL_EPOCHS = 3          
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# ARA-Fed Specific Parameters
ALPHA = 0.5               
BETA = 1.0                

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Dataset Preparation (Non-IID Simulation)

def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def split_non_iid(dataset, num_clients):
    
    data_len = len(dataset)
    indices = np.arange(data_len)
    labels = dataset.targets.numpy()
    
    
    idxs_labels = np.vstack((indices, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    
    shards = np.array_split(idxs, num_clients)
    client_dict = {i: shards[i] for i in range(num_clients)}
    return client_dict

# Load Data
train_data, test_data = get_dataset()
client_data_indices = split_non_iid(train_data, NUM_CLIENTS)

# Simulate Bandwidth for 100 clients (Randomly between 1Mbps and 100Mbps)
client_bandwidths = np.random.uniform(1, 100, NUM_CLIENTS)
max_bandwidth = 100.0


# Model Definition (Simple CNN)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Helper Functions

def client_update(client_model, optimizer, train_loader, epochs):
    client_model.train()
    total_loss = 0.0
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return client_model.state_dict(), total_loss / len(train_loader)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


# FEDERATED LEARNING ALGORITHMS


# Standard FedAvg (Baseline) 

def run_fedavg():
    print(f"\n--- Running Standard FedAvg (Clients={NUM_CLIENTS}) ---")
    global_model = SimpleCNN().to(device)
    acc_history = []
    
    for round_idx in range(TOTAL_ROUNDS):
        global_weights = global_model.state_dict()
        local_updates = []
        
        
        selected_clients = np.random.choice(range(NUM_CLIENTS), CLIENTS_PER_ROUND, replace=False)
        
        for client_id in selected_clients:
            idxs = client_data_indices[client_id]
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, idxs), batch_size=BATCH_SIZE, shuffle=True)
            
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_weights)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            
            w_i, _ = client_update(local_model, optimizer, loader, LOCAL_EPOCHS)
            local_updates.append({'weights': w_i, 'size': len(idxs)})
            
        
        total_samples = sum([u['size'] for u in local_updates])
        new_weights = copy.deepcopy(local_updates[0]['weights'])
        
        for key in new_weights.keys():
            new_weights[key] = torch.zeros_like(new_weights[key])
            for update in local_updates:
                weight_factor = update['size'] / total_samples
                new_weights[key] += update['weights'][key] * weight_factor
                
        global_model.load_state_dict(new_weights)
        
        # Test
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
        acc = evaluate(global_model, test_loader)
        acc_history.append(acc)
        print(f"Round {round_idx+1}/{TOTAL_ROUNDS}: FedAvg Accuracy = {acc:.2f}%")
        
    return acc_history

# Proposed ARA-Fed (Ours) 
def run_arafed():
    print(f"\n--- Running ARA-Fed (Clients={NUM_CLIENTS}) ---")
    global_model = SimpleCNN().to(device)
    acc_history = []
    
    total_data_size = sum([len(client_data_indices[i]) for i in range(NUM_CLIENTS)])
    
    for round_idx in range(TOTAL_ROUNDS):
        global_weights = global_model.state_dict()
        local_updates = []
        
        # Strategic Scheduling
        client_scores = []
        for i in range(NUM_CLIENTS):
            data_size = len(client_data_indices[i])
            bw = client_bandwidths[i]
           
            u_score = ALPHA * (bw / max_bandwidth) + (1 - ALPHA) * (data_size / total_data_size)
            client_scores.append(u_score)
            
        
        selected_clients = np.argsort(client_scores)[-CLIENTS_PER_ROUND:]
        
        for client_id in selected_clients:
            idxs = client_data_indices[client_id]
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, idxs), batch_size=BATCH_SIZE, shuffle=True)
            
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_weights)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            
            w_i, loss_i = client_update(local_model, optimizer, loader, LOCAL_EPOCHS)
            local_updates.append({'weights': w_i, 'size': len(idxs), 'loss': loss_i})
            
        # Loss-Adaptive Aggregation
        scaling_factors = []
        for update in local_updates:
            
            psi = np.exp(-BETA * update['loss'])
            scaling_factors.append(psi)
            
        total_weighted_size = sum([u['size'] * s for u, s in zip(local_updates, scaling_factors)])
        
        new_weights = copy.deepcopy(local_updates[0]['weights'])
        for key in new_weights.keys():
            new_weights[key] = torch.zeros_like(new_weights[key])
            for idx, update in enumerate(local_updates):
                weight_factor = (update['size'] * scaling_factors[idx]) / total_weighted_size
                new_weights[key] += update['weights'][key] * weight_factor
                
        global_model.load_state_dict(new_weights)
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
        acc = evaluate(global_model, test_loader)
        acc_history.append(acc)
        print(f"Round {round_idx+1}/{TOTAL_ROUNDS}: ARA-Fed Accuracy = {acc:.2f}%")
        
    return acc_history


# Execution & Plotting

history_fedavg = run_fedavg()
history_arafed = run_arafed()

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(range(1, TOTAL_ROUNDS+1), history_arafed, 'r-o', label='ARA-Fed (Ours)', linewidth=2)
plt.plot(range(1, TOTAL_ROUNDS+1), history_fedavg, 'b--s', label='FedAvg (Baseline)', linewidth=2)
plt.title(f'Comparison on Non-IID MNIST (N={NUM_CLIENTS})', fontsize=14)
plt.xlabel('Communication Rounds', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('arafed_100_clients.png')
plt.show()


print("\n--- Final Results ---")
print(f"FedAvg Final Accuracy: {history_fedavg[-1]:.2f}%")

print(f"ARA-Fed Final Accuracy: {history_arafed[-1]:.2f}%")
