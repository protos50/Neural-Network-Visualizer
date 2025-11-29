# 🧠 Neural Network Visualizer

An interactive neural network visualizer for learning mathematical functions and classifying datasets in real-time. Features a sci-fi CRT oscilloscope aesthetic with TensorFlow.js acceleration.

<p align="center">
  <img src="preview.png" alt="Neural Network Visualizer Preview" width="800"/>
</p>

## ✨ Features

### Two Application Modes

**📈 Regression Mode**
- Approximate mathematical functions (sin, cos, tan, custom formulas)
- Real-time oscilloscope visualization
- Test mode to evaluate generalization

**📊 Classification Mode**
- Load CSV datasets for binary classification
- Train/Test split with configurable ratio
- Live accuracy metrics (train & test)
- Feature importance visualization per neuron

### Core Features

- **TensorFlow.js Backend**: GPU-accelerated training
- **Interactive Network Visualization**: Neurons glow based on activation
- **Forward Pass Animation**: Watch signal flow through layers
- **Hover Tooltips**: Detailed neuron info with feature importance bars
- **Configurable Architecture**: Adjust hidden layers, neurons, and activations
- **i18n Support**: English/Spanish

## 🚀 Quick Start

```bash
git clone https://github.com/protos50/Neural-Network-Visualizer.git
cd Neural-Network-Visualizer
npm install
npm run dev
# Open http://localhost:3000
```

## 📁 CSV Dataset Format

Place CSV files in `/public/datasets/`. Use metadata comments to define columns:

```csv
# INPUT_COLS: Pclass,Sex,Age,SibSp,Parch,Fare
# OUTPUT_COLS: Survived
Pclass,Sex,Age,SibSp,Parch,Fare,Survived
3,0,22,1,0,7.25,0
1,1,38,1,0,71.28,1
...
```

### Rules

| Rule | Description |
|------|-------------|
| `# INPUT_COLS:` | Comma-separated input column names |
| `# OUTPUT_COLS:` | Output column name (binary: 0 or 1) |
| First row after comments | Header row (column names) |
| All values | Must be numeric |

## 🎮 Controls

| Control | Description |
|---------|-------------|
| ▶ Play/Pause | Start/stop training |
| ⏹ Stop | Stop and keep current state |
| ↺ Reset | Reinitialize network weights |
| Train/Test slider | Adjust data split ratio |
| Learning Rate | Control step size |

## 🛠️ Tech Stack

- **Next.js 14** + **TypeScript**
- **TensorFlow.js** (GPU acceleration)
- **Tailwind CSS** (CRT styling)
- **Lucide React** (Icons)

## 👤 Author

**Franco Joaquín Zini** — [GitHub](https://github.com/protos50) · [LinkedIn](https://linkedin.com/in/francojzini)

## 📝 License

MIT License — Educational use

---

<p align="center">
  Developed for <strong>Artificial Intelligence</strong> course — UNNE 2025
</p>
