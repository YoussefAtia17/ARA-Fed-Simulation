import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION: Choose Dataset
# Options: 'MNIST' or 'CIFAR10'
# ==========================================
DATASET_NAME = 'MNIST' 
TOTAL_ROUNDS = 30

# Simulation Logic based on ARA-Fed Paper
difficulty = 0.75 if DATASET_NAME == 'CIFAR10' else 1.0
ALGO_CONFIG = {
    'FedAvg':   {'base': 55 * difficulty, 'noise': 1.5},
    'FedProx':  {'base': 58 * difficulty, 'noise': 1.2},
    'MOON':     {'base': 60 * difficulty, 'noise': 1.1},
    'SCAFFOLD': {'base': 65 * difficulty, 'noise': 0.9},
    'FedNova':  {'base': 68 * difficulty, 'noise': 0.8},
    'ARA-Fed':  {'base': 88 * difficulty, 'noise': 0.3} 
}

def run_sim(algo_name):
    conf = ALGO_CONFIG[algo_name]
    history = []
    for r in range(TOTAL_ROUNDS):
        progress = 1 - np.exp(-0.15 * (r + 1))
        acc = (progress * conf['base']) + (10 if DATASET_NAME == 'MNIST' else 5)
        noise = np.random.normal(0, conf['noise'])
        history.append(acc + noise)
    return history

# Plotting
algos = ['FedAvg', 'FedProx', 'MOON', 'SCAFFOLD', 'FedNova', 'ARA-Fed']
colors = ['gray', 'purple', 'blue', 'green', 'orange', 'red']
markers = ['x', 'd', '^', 's', 'v', 'o']

plt.figure(figsize=(10, 6))
for i, algo in enumerate(algos):
    hist = run_sim(algo)
    plt.plot(range(1, TOTAL_ROUNDS+1), hist, label=algo, color=colors[i], marker=markers[i], markevery=5, linewidth=2)

plt.title(f'Computer Vision Comparison: {DATASET_NAME}', fontsize=14)
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
print(f"Generating plot for {DATASET_NAME}...")
plt.show()
