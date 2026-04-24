# 🧠 Self-Pruning Neural Network

This project implements a **self-pruning neural network** using PyTorch on the CIFAR-10 dataset.  
The model learns to **dynamically remove unnecessary weights during training** using a differentiable gating mechanism and L1 regularization.

---

## 🚀 Key Idea

Each weight is associated with a learnable gate:

W_eff = W * sigmoid(G)

- **W** → original weights  
- **G** → learnable gate scores  
- **sigmoid(G)** → values in [0,1]  

👉 If a gate approaches **0**, the corresponding weight is effectively **pruned**.

---

## 🏗️ Model Architecture

- Input: 32×32×3 image → flattened (3072)
- PrunableLinear (3072 → 512) → ReLU  
- PrunableLinear (512 → 256) → ReLU  
- PrunableLinear (256 → 10)

---

## ⚙️ Loss Function

Total Loss = CrossEntropy + λ * Sparsity Loss

- **Classification Loss** → accuracy  
- **Sparsity Loss (L1 on gates)** → encourages pruning  

👉 L1 penalty drives many gates toward **zero**, enabling sparsity.

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.001  | 54.47       | 5.54        |
| 0.01   | 53.46       | 5.59        |
| 0.05   | 53.17       | 5.86        |
| 0.1    | 34.50       | 81.21       |

---

## 📉 Sparsity Definition

A weight is considered **pruned** if:

sigmoid(gate_score) < 1e-2

---

## 📈 Key Insight

Achieved ~81% sparsity with learned gating while maintaining functional classification performance, demonstrating effective differentiable pruning.

---

## 📊 Observations

- Low λ → High accuracy, low sparsity  
- High λ → High sparsity, lower accuracy  
- Clear **trade-off between accuracy and sparsity**

---

## 📄 Report

[View Detailed Report](./SelfPruning_NN_Report.pdf)

---

## 📘 Notebook

[Open in Colab](https://colab.research.google.com/github/smaranikaduttapattanaik/self-pruning-neural-network/blob/main/self_pruning_neural_network.ipynb)

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- NumPy  
- Matplotlib  

---

## 🔮 Future Work

- Extend to **CNN architecture** for improved performance  
- Explore structured pruning (neuron/channel-level)  
- Apply to larger datasets  

---

## 🎯 Conclusion

This project demonstrates that neural networks can **learn to prune themselves during training**, eliminating unnecessary weights and forming a sparse, efficient model.  
The results validate the expected trade-off between **model performance and sparsity**.

---
