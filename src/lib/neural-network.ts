/**
 * Red Neuronal 100% JavaScript/TypeScript
 * Implementación desde cero para visualizar el aprendizaje
 * 
 * Arquitectura configurable con múltiples funciones de activación
 */

// ============================================
// FUNCIONES DE ACTIVACIÓN
// ============================================

export type ActivationFunction = 'tanh' | 'relu' | 'sigmoid' | 'linear';

function tanh(x: number): number {
  return Math.tanh(x);
}

function tanhDerivative(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

function relu(x: number): number {
  return Math.max(0, x);
}

function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
}

function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

function linear(x: number): number {
  return x;
}

function linearDerivative(_x: number): number {
  return 1;
}

// Obtener función de activación y su derivada
function getActivation(name: ActivationFunction): { fn: (x: number) => number; derivative: (x: number) => number } {
  switch (name) {
    case 'tanh':
      return { fn: tanh, derivative: tanhDerivative };
    case 'relu':
      return { fn: relu, derivative: reluDerivative };
    case 'sigmoid':
      return { fn: sigmoid, derivative: sigmoidDerivative };
    case 'linear':
      return { fn: linear, derivative: linearDerivative };
    default:
      return { fn: tanh, derivative: tanhDerivative };
  }
}

// Descripción de funciones de activación
export const activationDescriptions: Record<ActivationFunction, { formula: string; range: string; desc: string }> = {
  tanh: {
    formula: 'tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)',
    range: '[-1, 1]',
    desc: 'Centra los datos, buena para capas ocultas',
  },
  relu: {
    formula: 'ReLU(x) = max(0, x)',
    range: '[0, ∞)',
    desc: 'Simple y rápida, puede "morir" si x<0',
  },
  sigmoid: {
    formula: 'σ(x) = 1/(1 + e⁻ˣ)',
    range: '(0, 1)',
    desc: 'Clásica, problemas de gradiente en extremos',
  },
  linear: {
    formula: 'f(x) = x',
    range: '(-∞, ∞)',
    desc: 'Sin transformación, usada en capa de salida',
  },
};

// ============================================
// UTILIDADES MATEMÁTICAS
// ============================================

// Inicialización Xavier/Glorot para pesos
function initializeWeights(rows: number, cols: number): number[][] {
  const limit = Math.sqrt(6 / (rows + cols));
  const weights: number[][] = [];
  for (let i = 0; i < rows; i++) {
    weights[i] = [];
    for (let j = 0; j < cols; j++) {
      weights[i][j] = (Math.random() * 2 - 1) * limit;
    }
  }
  return weights;
}

function initializeBiases(size: number): number[] {
  return new Array(size).fill(0);
}

// Multiplicación matriz-vector
function matVecMul(matrix: number[][], vec: number[]): number[] {
  return matrix.map(row => 
    row.reduce((sum, w, i) => sum + w * vec[i], 0)
  );
}

// ============================================
// INTERFACES
// ============================================

export interface LayerState {
  preActivation: number[];  // Antes de activación (z)
  activation: number[];     // Después de activación (a)
}

// Layer type for i18n
export type LayerType = 'input' | 'hidden' | 'output';

// Pattern type for i18n
export interface PatternInfo {
  type: 'inputSensitivity' | 'combinesHidden' | 'maxWeight' | 'receivesX';
  sign?: string;
  value?: number;
  neuronNum?: number;
}

export interface NeuronInfo {
  layer: number;
  layerType: LayerType;      // For i18n translation
  layerNumber?: number;      // Hidden layer number (1, 2, etc.)
  index: number;
  weights: number[];
  bias: number;
  activation: ActivationFunction;
  currentInput: number;   // z = Σ(w*x) + b
  currentOutput: number;  // a = activation(z)
  patternInfo: PatternInfo;  // Structured pattern data for i18n
}

export interface NetworkConfig {
  hiddenLayers: number[];           // [8, 8] para 2 capas de 8 neuronas
  hiddenActivation: ActivationFunction;
  outputActivation: ActivationFunction;
  learningRate: number;
  momentum: number;                 // Momentum para SGD
}

export interface NetworkState {
  weights: number[][][];
  biases: number[][];
  layers: LayerState[];
  loss: number;
  epoch: number;
  config: NetworkConfig;
}

// ============================================
// RED NEURONAL CONFIGURABLE
// ============================================

export class NeuralNetwork {
  private weights: number[][][] = [];
  private biases: number[][] = [];
  
  // Velocidades para momentum
  private vWeights: number[][][] = [];
  private vBiases: number[][] = [];
  
  // Cache para backward pass
  private layerInputs: number[][] = [];
  private layerZ: number[][] = [];
  private layerA: number[][] = [];
  
  private config: NetworkConfig;
  public epoch: number = 0;

  constructor(config: Partial<NetworkConfig> = {}) {
    this.config = {
      hiddenLayers: config.hiddenLayers || [8, 8],
      hiddenActivation: config.hiddenActivation || 'tanh',
      outputActivation: config.outputActivation || 'linear',
      learningRate: config.learningRate || 0.01,
      momentum: config.momentum || 0,
    };
    
    this.initializeNetwork();
  }

  private initializeNetwork(): void {
    // Construir arquitectura: 1 → hidden1 → hidden2 → ... → 1
    const layers = [1, ...this.config.hiddenLayers, 1];
    
    this.weights = [];
    this.biases = [];
    this.vWeights = [];
    this.vBiases = [];
    
    for (let i = 0; i < layers.length - 1; i++) {
      const inputSize = layers[i];
      const outputSize = layers[i + 1];
      
      this.weights.push(initializeWeights(outputSize, inputSize));
      this.biases.push(initializeBiases(outputSize));
      
      // Inicializar velocidades para momentum en cero
      const zeroWeights: number[][] = [];
      for (let r = 0; r < outputSize; r++) {
        zeroWeights.push(new Array(inputSize).fill(0));
      }
      this.vWeights.push(zeroWeights);
      this.vBiases.push(new Array(outputSize).fill(0));
    }
  }

  // Forward pass para una sola muestra
  forward(x: number): number {
    this.layerInputs = [[x]];
    this.layerZ = [];
    this.layerA = [];
    
    let currentInput = [x];
    
    for (let i = 0; i < this.weights.length; i++) {
      // z = W * input + b
      const z = matVecMul(this.weights[i], currentInput).map((v, j) => v + this.biases[i][j]);
      this.layerZ.push(z);
      
      // Elegir activación
      const isOutputLayer = i === this.weights.length - 1;
      const activationName = isOutputLayer ? this.config.outputActivation : this.config.hiddenActivation;
      const { fn } = getActivation(activationName);
      
      // a = activation(z)
      const a = z.map(fn);
      this.layerA.push(a);
      
      // Guardar para la siguiente capa
      if (i < this.weights.length - 1) {
        this.layerInputs.push(a);
      }
      currentInput = a;
    }
    
    return this.layerA[this.layerA.length - 1][0];
  }

  // Backward pass para una sola muestra
  backward(yTrue: number): void {
    const numLayers = this.weights.length;
    const dL_dz: number[][] = [];
    
    // Empezar desde la última capa
    const yPred = this.layerA[numLayers - 1][0];
    const dL_da_last = 2 * (yPred - yTrue);
    
    // Derivada de la capa de salida
    const { derivative: outputDeriv } = getActivation(this.config.outputActivation);
    dL_dz[numLayers - 1] = [dL_da_last * outputDeriv(this.layerZ[numLayers - 1][0])];
    
    // Propagar hacia atrás
    const { derivative: hiddenDeriv } = getActivation(this.config.hiddenActivation);
    
    for (let i = numLayers - 2; i >= 0; i--) {
      // dL/da[i] = W[i+1]^T * dL/dz[i+1]
      const nextWeights = this.weights[i + 1];
      const nextDz = dL_dz[i + 1];
      
      const dL_da = new Array(this.layerA[i].length).fill(0);
      for (let j = 0; j < nextWeights.length; j++) {
        for (let k = 0; k < nextWeights[j].length; k++) {
          dL_da[k] += nextWeights[j][k] * nextDz[j];
        }
      }
      
      // dL/dz[i] = dL/da[i] * activation'(z[i])
      dL_dz[i] = dL_da.map((v, j) => v * hiddenDeriv(this.layerZ[i][j]));
    }
    
    // Actualizar pesos y biases con momentum
    for (let i = 0; i < numLayers; i++) {
      const input = this.layerInputs[i];
      const dz = dL_dz[i];
      
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          const grad = dz[j] * input[k];
          
          // Aplicar momentum: v = momentum * v + lr * grad
          this.vWeights[i][j][k] = this.config.momentum * this.vWeights[i][j][k] + this.config.learningRate * grad;
          this.weights[i][j][k] -= this.vWeights[i][j][k];
        }
        
        // Bias update con momentum
        const biasGrad = dz[j];
        this.vBiases[i][j] = this.config.momentum * this.vBiases[i][j] + this.config.learningRate * biasGrad;
        this.biases[i][j] -= this.vBiases[i][j];
      }
    }
  }

  // Entrenar una época completa
  trainEpoch(X: number[], Y: number[]): number {
    let totalLoss = 0;
    
    // Mezclar datos para SGD
    const indices = X.map((_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    for (const i of indices) {
      const pred = this.forward(X[i]);
      totalLoss += (pred - Y[i]) ** 2;
      this.backward(Y[i]);
    }
    
    this.epoch++;
    return totalLoss / X.length;  // MSE
  }

  // Predecir para array de inputs
  predict(X: number[]): number[] {
    return X.map(x => this.forward(x));
  }

  // Obtener información detallada de una neurona específica
  getNeuronInfo(layerIndex: number, neuronIndex: number): NeuronInfo {
    const layers = [1, ...this.config.hiddenLayers, 1];
    const isOutputLayer = layerIndex === layers.length - 1;
    const isInputLayer = layerIndex === 0;
    
    let layerType: LayerType;
    let layerNumber: number | undefined;
    let weights: number[] = [];
    let bias = 0;
    let activation: ActivationFunction;
    let currentInput = 0;
    let currentOutput = 0;
    
    if (isInputLayer) {
      layerType = 'input';
      activation = 'linear';
      if (this.layerInputs[0]) {
        currentInput = this.layerInputs[0][0];
        currentOutput = this.layerInputs[0][0];
      }
    } else {
      const weightLayerIdx = layerIndex - 1;
      
      if (isOutputLayer) {
        layerType = 'output';
        activation = this.config.outputActivation;
      } else {
        layerType = 'hidden';
        layerNumber = layerIndex;
        activation = this.config.hiddenActivation;
      }
      
      if (this.weights[weightLayerIdx] && this.weights[weightLayerIdx][neuronIndex]) {
        weights = [...this.weights[weightLayerIdx][neuronIndex]];
        bias = this.biases[weightLayerIdx][neuronIndex];
      }
      
      if (this.layerZ[weightLayerIdx]) {
        currentInput = this.layerZ[weightLayerIdx][neuronIndex] || 0;
      }
      if (this.layerA[weightLayerIdx]) {
        currentOutput = this.layerA[weightLayerIdx][neuronIndex] || 0;
      }
    }
    
    // Generar información del patrón (para traducción en UI)
    let patternInfo: PatternInfo;
    if (!isInputLayer && weights.length > 0) {
      const absWeights = weights.map(Math.abs);
      const maxWeight = Math.max(...absWeights);
      const dominantIdx = absWeights.indexOf(maxWeight);
      const sign = weights[dominantIdx] > 0 ? '+' : '-';
      
      if (layerIndex === 1) {
        patternInfo = { type: 'inputSensitivity', sign, value: maxWeight };
      } else if (isOutputLayer) {
        patternInfo = { type: 'combinesHidden' };
      } else {
        patternInfo = { type: 'maxWeight', sign, value: maxWeight, neuronNum: dominantIdx + 1 };
      }
    } else {
      patternInfo = { type: 'receivesX' };
    }
    
    return {
      layer: layerIndex,
      layerType,
      layerNumber,
      index: neuronIndex,
      weights,
      bias,
      activation,
      currentInput,
      currentOutput,
      patternInfo,
    };
  }

  // Obtener estado actual para visualización
  getState(X: number[], Y: number[]): NetworkState {
    // Hacer forward con el primer punto para tener activaciones
    if (X.length > 0) {
      this.forward(X[0]);
    }
    
    // Calcular loss actual
    const predictions = this.predict(X);
    const loss = predictions.reduce((sum, pred, i) => sum + (pred - Y[i]) ** 2, 0) / X.length;
    
    // Construir layers para visualización (incluyendo input)
    const layers: LayerState[] = [
      { preActivation: this.layerInputs[0] ? [...this.layerInputs[0]] : [0], activation: this.layerInputs[0] ? [...this.layerInputs[0]] : [0] },
      ...this.layerZ.map((z, i) => ({
        preActivation: [...z],
        activation: [...this.layerA[i]],
      })),
    ];
    
    return {
      weights: this.weights.map(w => w.map(row => [...row])),
      biases: this.biases.map(b => [...b]),
      layers,
      loss,
      epoch: this.epoch,
      config: { ...this.config },
    };
  }

  // Obtener estado inicial SIN ejecutar forward (para reset limpio)
  getInitialState(): NetworkState {
    const arch = this.getArchitecture();
    
    // Crear layers vacíos con ceros
    const layers: LayerState[] = arch.map(count => ({
      preActivation: new Array(count).fill(0),
      activation: new Array(count).fill(0),
    }));
    
    return {
      weights: this.weights.map(w => w.map(row => [...row])),
      biases: this.biases.map(b => [...b]),
      layers,
      loss: 1,
      epoch: 0,
      config: { ...this.config },
    };
  }

  // Calcular solo el loss sin modificar el estado
  calculateLoss(X: number[], Y: number[]): number {
    const predictions = X.map(x => this.forward(x));
    return predictions.reduce((sum, pred, i) => sum + (pred - Y[i]) ** 2, 0) / X.length;
  }

  // Generar puntos del patrón que esta neurona detecta
  // Esto muestra la "contribución" de la neurona a la salida
  getNeuronPattern(layerIndex: number, neuronIndex: number, numPoints: number = 50): { x: number; y: number }[] {
    const points: { x: number; y: number }[] = [];
    const arch = this.getArchitecture();
    
    // Para la capa de entrada, simplemente muestra la identidad
    if (layerIndex === 0) {
      for (let i = 0; i < numPoints; i++) {
        const x = -6 + (12 * i / (numPoints - 1));
        points.push({ x, y: x / 6 }); // Normalizado a [-1, 1]
      }
      return points;
    }
    
    // Para capas ocultas y salida, mostrar la activación de esa neurona
    for (let i = 0; i < numPoints; i++) {
      const x = -6 + (12 * i / (numPoints - 1));
      this.forward(x);
      
      let y = 0;
      if (layerIndex === arch.length - 1) {
        // Capa de salida
        y = this.layerA[this.layerA.length - 1][neuronIndex] || 0;
      } else {
        // Capa oculta
        const hiddenLayerIdx = layerIndex - 1;
        if (this.layerA[hiddenLayerIdx]) {
          y = this.layerA[hiddenLayerIdx][neuronIndex] || 0;
        }
      }
      
      points.push({ x, y });
    }
    
    return points;
  }

  // Getters y setters
  setLearningRate(lr: number): void {
    this.config.learningRate = lr;
  }

  setMomentum(m: number): void {
    this.config.momentum = m;
  }

  getConfig(): NetworkConfig {
    return { ...this.config };
  }

  getArchitecture(): number[] {
    return [1, ...this.config.hiddenLayers, 1];
  }

  // Reiniciar la red (misma config)
  reset(): void {
    this.initializeNetwork();
    this.epoch = 0;
    this.layerInputs = [];
    this.layerZ = [];
    this.layerA = [];
  }

  // Reiniciar con nueva configuración
  reconfigure(config: Partial<NetworkConfig>): void {
    this.config = {
      ...this.config,
      ...config,
    };
    this.initializeNetwork();
    this.epoch = 0;
    this.layerInputs = [];
    this.layerZ = [];
    this.layerA = [];
  }
}

// ============================================
// GENERADOR DE DATOS
// ============================================

export function generateSineDataset(
  numPoints: number = 200,
  noiseLevel: number = 0.2,
  xMin: number = -6,
  xMax: number = 6
): { X: number[], Y: number[], YTrue: number[] } {
  const X: number[] = [];
  const Y: number[] = [];
  const YTrue: number[] = [];
  
  for (let i = 0; i < numPoints; i++) {
    const x = xMin + (xMax - xMin) * (i / (numPoints - 1));
    const yTrue = Math.sin(x);
    const noise = (Math.random() * 2 - 1) * noiseLevel;
    
    X.push(x);
    YTrue.push(yTrue);
    Y.push(yTrue + noise);
  }
  
  return { X, Y, YTrue };
}
