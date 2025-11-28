# 🧠 Neural Sine Learner

An interactive neural network visualizer learning the sine function in real-time. Built from scratch with no external ML libraries. Watch how a feedforward neural network progressively learns to approximate the sine wave through gradient descent and backpropagation.

> 🤖 **Note on Development:** This project was designed and programmed with the assistance of Artificial Intelligence. It serves as a dual example: on one hand, it teaches the fundamentals of neural networks (forward propagation, backpropagation, gradient descent), and on the other, it demonstrates how modern generative AI can accelerate and enhance software development.

<p align="center">
  <img src="preview.png" alt="Neural Sine Learner Preview" width="800"/>
</p>

## ✨ Features

- **Neural Network from Scratch**: 100% implemented in TypeScript without external ML libraries
- **Real-time Visualization**: Watch the network adjust its prediction epoch by epoch
- **Sci-fi Aesthetics**: CRT oscilloscope style with glow effects
- **Interactive Control Panel**:
  - Play/Pause training
  - Adjustable speed (epochs per tick) with slow-motion support
  - Real-time learning rate adjustment
  - Momentum control
  - Reset to restart the experiment
- **Neuron Visualization**: Neurons glow based on their activation level
- **Forward Pass Animation**: Visualize signal flow through the network with cyan highlights
- **Hover Tooltips**: Detailed information about each neuron's weights, bias, and interpretation
- **i18n Support**: English/Spanish language toggle

## 🏗️ Network Architecture

```
Input (1) → Hidden (8, tanh) → Hidden (8, tanh) → Output (1, linear)
```

- **Input**: A single value x in range [-6, 6]
- **Hidden Layers**: 2 layers of 8 neurons each with tanh activation
- **Output**: A single value ŷ (sine prediction)
- **Optimization**: Stochastic Gradient Descent (SGD) with manual backpropagation
- **Momentum**: Optional momentum for faster convergence

## 📊 Dataset

- **200 points** generated as `sin(x) + noise`
- **Gaussian noise** with σ = 0.2
- **Range**: x ∈ [-6, 6]

## 🚀 Installation

### Requirements
- Node.js 18+
- npm or yarn

### Steps

```bash
# Clone the repository
git clone https://github.com/protos50/Neural-Sine-Learner.git
cd Neural-Sine-Learner

# Install dependencies
npm install

# Run development server
npm run dev

# Open in browser
# http://localhost:3000
```

## 🎮 Usage

### Controls
| Control | Description |
|---------|-------------|
| ▶ Play/Pause | Start/stop training |
| ⏹ Stop | Stop training, keep current state |
| ↺ Reset | Reset network with random weights |
| ⚡ Forward | Toggle forward pass animation |
| Speed Slider | Control epochs per tick (0.1-50) |
| Learning Rate | Adjust learning rate (0.001-0.1) |
| Momentum | Add inertia to weight updates (0-0.99) |

### Visualization

#### Main Canvas (Oscilloscope)
- **Bright green dots**: Noisy dataset (sin(x) + noise)
- **Green laser line**: Current network prediction
- **Faint dotted line**: True sine function

#### Neural Network View
- **Bright circles**: Highly activated neurons
- **Dim circles**: Low activation neurons
- **Green lines**: Positive weights
- **Red lines**: Negative weights
- **Cyan glow**: Forward pass animation (when enabled)

#### Metrics
- **Epoch**: Number of complete passes through the dataset
- **Loss (MSE)**: Mean Squared Error (lower is better)

## 📁 Project Structure

```
Neural-Sine-Learner/
├── src/
│   ├── app/
│   │   ├── globals.css           # Global styles (CRT effects)
│   │   ├── layout.tsx            # Main layout
│   │   └── page.tsx              # Main page with training loop
│   ├── components/
│   │   ├── OscilloscopeCanvas.tsx  # Graph canvas
│   │   ├── NeuralNetworkViz.tsx    # Neuron visualization
│   │   └── TrainingPanel.tsx       # Control panel
│   └── lib/
│       ├── neural-network.ts       # Neural network from scratch
│       └── i18n.tsx                # Internationalization
├── package.json
├── tailwind.config.ts
└── README.md
```

## 🧪 Concepts Demonstrated

| Concept | Description |
|---------|-------------|
| **Forward Propagation** | Signal flow from input to output |
| **Backpropagation** | Gradient calculation through chain rule |
| **Gradient Descent** | Weight updates to minimize loss |
| **Activation Functions** | tanh non-linearity for learning curves |
| **Loss Function (MSE)** | Error metric for regression |
| **Momentum** | Accelerated convergence, escaping local minima |
| **Learning Rate** | Step size trade-off (stability vs speed) |

## 🛠️ Tech Stack

- **Next.js 14** (React framework)
- **TypeScript** (Type safety)
- **Tailwind CSS** (Styling)
- **Canvas API** (Visualization)
- **Lucide React** (Icons)

## 🎓 Educational Purpose

This project visually demonstrates:

- How a neural network **learns to approximate a function**
- The effect of **hyperparameters** (learning rate, momentum, architecture)
- The **convergence process** of the loss function
- The importance of **non-linearity** (without tanh, it couldn't learn the curve)
- **Forward pass signal flow** through the network layers

## 👤 Author

**Franco Joaquín Zini**

- LinkedIn: [francojzini](https://linkedin.com/in/francojzini)
- GitHub: [protos50](https://github.com/protos50)

## 📝 License

MIT License - Educational use

---

<p align="center">
  Developed for <strong>Artificial Intelligence</strong> course - UNNE 2025
</p>
