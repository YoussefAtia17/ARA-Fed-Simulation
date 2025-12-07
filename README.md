# ARA-Fed: Adaptive Resource-Aware Federated Learning Framework 

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/Status-Research_Validated-success.svg)
![Domain](https://img.shields.io/badge/Domain-Multi_Modal_AI-orange.svg)

This repository contains the official simulation environment for the research paper:

> **"Federated Learning for Privacy-Preserving AI: An Optimized Client-Server Aggregation Protocol"**

**ARA-Fed** is a novel Federated Learning framework designed to address communication bottlenecks and statistical heterogeneity (Non-IID data) in edge networks. It introduces a dual-optimization strategy: **Strategic Client Scheduling** and **Loss-Adaptive Aggregation**.

---

## Key Features

* **Multi-Modal Evaluation:** Validated across 3 domains:
    * **Computer Vision:** MNIST, CIFAR-10
    * **NLP:** AG News, IMDB
    * **Audio:** SpeechCommands, UrbanSound8K
* **Comprehensive Benchmarking:** Compared against **5 State-of-the-Art Baselines**:
    * FedAvg, FedProx, SCAFFOLD, FedNova, MOON
* **Non-IID Robustness:** Tested under strict Dirichlet partitioning ($Dir(\alpha)=0.5$).
* **Scalability:** Simulated on a network of **100 Clients**.

---

## Performance Highlights

ARA-Fed consistently outperforms baselines across all modalities:

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

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YoussefAtia17/ARA-Fed-Simulation.git](https://github.com/YoussefAtia17/ARA-Fed-Simulation.git)
    cd ARA-Fed-Simulation
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib
    ```

3.  **Run Simulations:**

    * **For Computer Vision (MNIST / CIFAR-10):**
        ```bash
        python vision_simulation.py
        ```
        *(Note: Edit the file to toggle between `MNIST` and `CIFAR10`)*

    * **For NLP (AG News / IMDB):**
        ```bash
        python text_simulation.py
        ```

    * **For Audio (Speech / UrbanSound):**
        ```bash
        python audio_simulation.py
        ```

---

## Acknowledgment
We would like to thank **Ms. ALZahraa Sherif** for her valuable technical assistance and contribution to the implementation of the multi-modal simulation experiments.

---

## Citation
If you use this code, please cite our paper:
```bibtex
@article{arafed2024,
  title={Federated Learning for Privacy-Preserving AI: An Optimized Client-Server Aggregation Protocol},
  author={Atia, Youssef and Sherif, ALZahraa et al.},
  journal={Submitted to Q1 Journal},
  year={2024}
}
