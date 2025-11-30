# ARA-Fed: Adaptive Resource-Aware Federated Learning Framework

This repository contains the official PyTorch implementation of the paper:
**"Federated Learning for Privacy-Preserving AI: An Optimized Client-Server Aggregation Protocol"**

**ARA-Fed** is a novel Federated Learning framework designed to address communication bottlenecks and statistical heterogeneity (Non-IID data) in edge networks. It introduces:
1.  **Strategic Client Scheduling:** Prioritizes clients based on bandwidth and data utility.
2.  **Loss-Adaptive Aggregation:** Dynamically weights updates based on local training loss to mitigate client drift.

---

=> Key Results
The simulation demonstrates that ARA-Fed achieves **~12.9% higher accuracy** and **faster convergence** compared to standard FedAvg on the MNIST Non-IID benchmark ($N=100$ clients).

---

=> Prerequisites
* Python 3.8+
* PyTorch
* Torchvision
* Matplotlib
* Numpy

To install dependencies:
```bash
pip install torch torchvision matplotlib numpy
