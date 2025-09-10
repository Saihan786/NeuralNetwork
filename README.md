# Custom Neural Network Library in Python

This project implements a **neural network library** in Python, enabling you to **define your own neural networks** with fully customizable layers and neurons. The library is packaged as a Python package (via `setup.py`) and can be used to build, train, and experiment with neural networks from scratch.

---

## 🚀 Key Features
- **Modular structure**:
  - `Neuron`: stores **weights**, **bias**, and **activation**.
  - `InitialNeuronLayer`, `InternalNeuronLayer`, `OutputNeuronLayer`: manage layer-specific forward and backward logic.
  - `Network`: orchestrates multiple layers, forward activation, and training.
- **Forward propagation** using the sigmoid activation function.
- **Backpropagation**:
  - Iterative, memoized calculation of **weight gradients**.
  - Bias updates calculated similarly to weights.
  - Targeted learning per neuron for faster convergence.
- Customizable **training loops** with **cost calculation** and **weight updates**.
- Debugging-friendly with `print()` methods for layers and networks.

---

## 📦 Installation

You can install the package locally:

```bash
git clone <repo_url>
cd <repo_folder>
pip install .
```

Then import and use the library in your Python code:

```python
from neural_network import neuron_class, neuron_layer_classes, network_class
```

---

## ⚡ Usage

To see the library in action, please refer to **`main.py`** in this repository.  

It demonstrates:
- Creating a **10-layer neural network**.
- Initializing weights and biases.
- Forward propagation for sample inputs.
- Training the network using backpropagation with both weight and bias derivatives.
- Printing training progress and final metrics.

---

## 🎊 Training Metrics

**Initial naive backprop (weights only):**  
**❌ ~25% cost reduction** over a 10-layer network  

**Final backprop (weights + bias derivatives, targeted learning):**  
**✅ ~60% cost reduction**  

**Hyperparameters:**
- **Weight learning rate:** 0.05  
- **Bias learning rate:** 0.025 (bias has stronger effect)  
- **Moderate repetition of training examples** to avoid hyperspecialization.

---

## 🎟️ Design Decisions / Milestones

1. **Initial network prototype**:
   - Simple cost halving approach to propagate errors backward.
   - Extremely limited learning beyond 1–2 iterations.

2. **Rough backpropagation**:
   - Learned the calculus behind backprop.
   - Implemented iterative, memoized weight derivative updates.

3. **Full backpropagation**:
   - Added bias derivatives.
   - Implemented targeted learning per neuron and layer.
   - Optimized learning rates to balance weight and bias effects.

4. **Modular library structure**:
   - Separated neurons, layers, and network orchestration.
   - Enabled flexible definition of networks of any depth and size.

---

## ⚡ Next Steps / TODO

- Optimize network initialization and randomization.
- Extend activation functions beyond sigmoid.
- Add support for mini-batch training.
- Implement evaluation metrics beyond simple cost reduction.

---

This package provides an **educational yet practical framework** for building neural networks from scratch while gaining an intuitive understanding of forward propagation, backpropagation, and training dynamics.

