# ARA-Fed: Adaptive Resource-Aware Federated Learning Framework 

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Status](https://img.shields.io/badge/Status-Research_Validated-success.svg)

This repository contains the official **PyTorch implementation** and simulation environment for the research paper:

> **"Federated Learning for Privacy-Preserving AI: An Optimized Client-Server Aggregation Protocol"**

**ARA-Fed** is a novel Federated Learning framework designed to address communication bottlenecks and statistical heterogeneity (Non-IID data) in edge networks. It introduces a dual-optimization strategy: **Strategic Client Scheduling** and **Loss-Adaptive Aggregation**.

---

## Key Features

* **Multi-Modal Evaluation:** Validated across **Computer Vision** (MNIST, CIFAR-10), **NLP** (AG News, IMDB), and **Audio** (SpeechCommands, UrbanSound8K).
* **Comprehensive Benchmarking:** Compared against **5 State-of-the-Art Baselines**:
    * FedAvg (Google)
    * FedProx (Heterogeneity-aware)
    * SCAFFOLD (Control Variates)
    * FedNova (Normalized Averaging)
    * MOON (Contrastive Learning)
* **Non-IID Robustness:** Tested under strict Dirichlet partitioning ($Dir(\alpha)=0.5$).
* **Scalability:** Simulated on a network of **100 Clients** with heterogeneous bandwidths.

---

## Performance Highlights

ARA-Fed consistently outperforms state-of-the-art baselines across all modalities, achieving up to **17.8% higher accuracy** in resource-constrained settings.

### 1. Computer Vision (CV)
| Algorithm | MNIST (Accuracy) | CIFAR-10 (Accuracy) |
| :--- | :--- | :--- |
| FedAvg | 64.90% | 51.82% |
| SCAFFOLD | 75.45% | 57.78% |
| **ARA-Fed (Ours)** | **90.62%**  | **71.05%**  |

### 2. Natural Language Processing (NLP)
| Algorithm | AG News (Topics) | IMDB (Sentiment) |
| :--- | :--- | :--- |
| FedAvg | 62.23% | 55.24% |
| MOON | 69.48% | 61.05% |
| **ARA-Fed (Ours)** | **88.94%**  | **76.81%**  |

### 3. Audio Processing (Voice)
| Algorithm | Speech Commands | UrbanSound8K |
| :--- | :--- | :--- |
| FedAvg | 53.54% | 45.64% |
| FedNova | 74.63% | 57.59% |
| **ARA-Fed (Ours)** | **92.78%**  | **74.49%**  |

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YoussefAtia17/ARA-Fed-Simulation.git](https://github.com/YoussefAtia17/ARA-Fed-Simulation.git)
    cd ARA-Fed-Simulation
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchtext torchaudio matplotlib numpy
    ```

---

## How to Run

The simulation is modular. You can select the target domain (Vision, Text, or Audio) by modifying the configuration in the script.

### Scenario A: Computer Vision
To run the simulation on **MNIST** or **CIFAR-10**:
```bash
# Open vision_simulation.py and set DATASET_NAME = 'CIFAR10'
python vision_simulation.py
