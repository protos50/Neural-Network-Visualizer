/**
 * Red Neuronal con TensorFlow.js
 * Implementación avanzada con múltiples activaciones, optimizadores y funciones de pérdida
 */

import * as tf from '@tensorflow/tfjs';

// ============================================
// TIPOS Y CONFIGURACIONES
// ============================================

export const ALLOWED_ACTIVATIONS = [
  'linear',
  'relu',
  'relu6',
  'sigmoid',
  'tanh',
  'elu',
  'selu',
  'softsign',
  'softplus',
  'swish',
  'gelu',
  'mish',
  'hardSigmoid',
] as const;

export type TFActivation = typeof ALLOWED_ACTIVATIONS[number];

export const LOSS_FUNCTIONS = [
  'meanSquaredError',
  'meanAbsoluteError',
  'huber',
  'logcosh',
] as const;

export type LossFn = typeof LOSS_FUNCTIONS[number];

export const OPTIMIZERS = [
  'sgd',
  'momentum',
  'adam',
  'adamax',
  'rmsprop',
  'adagrad',
  'adadelta',
] as const;

export type OptimizerName = typeof OPTIMIZERS[number];

// Descripciones para UI
export const activationDescriptions: Record<TFActivation, { formula: string; range: string; desc: string }> = {
  linear: {
    formula: 'f(x) = x',
    range: '(-∞, ∞)',
    desc: 'Sin transformación, identidad',
  },
  relu: {
    formula: 'ReLU(x) = max(0, x)',
    range: '[0, ∞)',
    desc: 'Rápida, puede "morir" si x<0',
  },
  relu6: {
    formula: 'ReLU6(x) = min(max(0, x), 6)',
    range: '[0, 6]',
    desc: 'ReLU limitada a 6, evita explosión',
  },
  sigmoid: {
    formula: 'σ(x) = 1/(1 + e⁻ˣ)',
    range: '(0, 1)',
    desc: 'Clásica, problemas de gradiente',
  },
  tanh: {
    formula: 'tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)',
    range: '[-1, 1]',
    desc: 'Centra datos, buena para ocultas',
  },
  elu: {
    formula: 'ELU(x) = x if x>0 else α(eˣ-1)',
    range: '(-α, ∞)',
    desc: 'Suaviza negativos, evita muerte',
  },
  selu: {
    formula: 'SELU(x) = λ·ELU(x)',
    range: '(-λα, ∞)',
    desc: 'Auto-normalización, redes profundas',
  },
  softsign: {
    formula: 'f(x) = x/(1 + |x|)',
    range: '(-1, 1)',
    desc: 'Similar a tanh, más suave',
  },
  softplus: {
    formula: 'f(x) = log(1 + eˣ)',
    range: '(0, ∞)',
    desc: 'Versión suave de ReLU',
  },
  swish: {
    formula: 'Swish(x) = x·σ(x)',
    range: '(-∞, ∞)',
    desc: 'No monótona, mejor que ReLU',
  },
  gelu: {
    formula: 'GELU(x) ≈ x·Φ(x)',
    range: '(-∞, ∞)',
    desc: 'Usada en Transformers, suave',
  },
  mish: {
    formula: 'Mish(x) = x·tanh(softplus(x))',
    range: '(-∞, ∞)',
    desc: 'Suave, no monótona, moderna',
  },
  hardSigmoid: {
    formula: 'f(x) = max(0, min(1, 0.2x+0.5))',
    range: '[0, 1]',
    desc: 'Aproximación rápida de sigmoid',
  },
};

export const optimizerDescriptions: Record<OptimizerName, string> = {
  sgd: 'Descenso de gradiente estocástico básico',
  momentum: 'SGD con momentum (inercia)',
  adam: 'Adaptive Moment Estimation (recomendado)',
  adamax: 'Variante de Adam con norma infinita',
  rmsprop: 'Root Mean Square Propagation',
  adagrad: 'Adaptive Gradient (adapta LR por parámetro)',
  adadelta: 'Extensión de Adagrad sin LR fijo',
};

export const lossDescriptions: Record<LossFn, string> = {
  meanSquaredError: 'MSE: (y - ŷ)² - Penaliza errores grandes',
  meanAbsoluteError: 'MAE: |y - ŷ| - Robusto a outliers',
  huber: 'Huber: MSE para errores pequeños, MAE para grandes',
  logcosh: 'Log-cosh: Suave, similar a Huber',
};

// ============================================
// CONFIGURACIÓN DEL MODELO
// ============================================

export interface LayerConfig {
  units: number;
  activation: TFActivation;
}

export interface TFModelConfig {
  layers: LayerConfig[];
  loss: LossFn;
  optimizer: OptimizerName;
  learningRate: number;
  momentum?: number; // Solo para momentum optimizer
}

export interface TFNetworkState {
  epoch: number;
  loss: number;
  valLoss?: number;
  predictions: number[];
  config: TFModelConfig;
}

// ============================================
// CLASE WRAPPER PARA TENSORFLOW.JS
// ============================================

export class TensorFlowNetwork {
  private model: tf.Sequential | null = null;
  private config: TFModelConfig;
  public epoch: number = 0;
  private trainingHistory: { epoch: number; loss: number; valLoss?: number }[] = [];

  constructor(config: TFModelConfig) {
    this.config = config;
    this.buildModel();
  }

  private buildModel(): void {
    // Limpiar modelo anterior si existe
    if (this.model) {
      this.model.dispose();
    }

    this.model = tf.sequential();

    // Agregar capas
    this.config.layers.forEach((layer, i) => {
      this.model!.add(tf.layers.dense({
        units: layer.units,
        activation: layer.activation as any,
        inputShape: i === 0 ? [1] : undefined,
        kernelInitializer: 'glorotUniform',
      }));
    });

    // Construir optimizador
    const optimizer = this.buildOptimizer();

    // Compilar modelo
    this.model.compile({
      optimizer,
      loss: this.config.loss as any,
      metrics: ['mse'],
    });
  }

  private buildOptimizer(): tf.Optimizer {
    const lr = this.config.learningRate;
    
    switch (this.config.optimizer) {
      case 'sgd':
        return tf.train.sgd(lr);
      case 'momentum':
        return tf.train.momentum(lr, this.config.momentum || 0.9);
      case 'adam':
        return tf.train.adam(lr);
      case 'adamax':
        return tf.train.adamax(lr);
      case 'rmsprop':
        return tf.train.rmsprop(lr);
      case 'adagrad':
        return tf.train.adagrad(lr);
      case 'adadelta':
        return tf.train.adadelta(lr);
      default:
        return tf.train.adam(lr);
    }
  }

  // Entrenar una época
  async trainEpoch(X: number[], Y: number[]): Promise<number> {
    if (!this.model) throw new Error('Model not initialized');

    const xs = tf.tensor2d(X, [X.length, 1]);
    const ys = tf.tensor2d(Y, [Y.length, 1]);

    const history = await this.model.fit(xs, ys, {
      epochs: 1,
      batchSize: 32,
      shuffle: true,
      verbose: 0,
    });

    xs.dispose();
    ys.dispose();

    this.epoch++;
    const loss = history.history.loss[0] as number;
    
    this.trainingHistory.push({ epoch: this.epoch, loss });
    
    return loss;
  }

  // Entrenar múltiples épocas
  async trainEpochs(X: number[], Y: number[], epochs: number): Promise<number> {
    if (!this.model) throw new Error('Model not initialized');

    const xs = tf.tensor2d(X, [X.length, 1]);
    const ys = tf.tensor2d(Y, [Y.length, 1]);

    const history = await this.model.fit(xs, ys, {
      epochs,
      batchSize: 32,
      shuffle: true,
      verbose: 0,
    });

    xs.dispose();
    ys.dispose();

    this.epoch += epochs;
    const losses = history.history.loss as number[];
    const finalLoss = losses[losses.length - 1];
    
    // Guardar historial
    losses.forEach((loss, i) => {
      this.trainingHistory.push({ epoch: this.epoch - epochs + i + 1, loss });
    });
    
    return finalLoss;
  }

  // Predecir
  predict(X: number[]): number[] {
    if (!this.model) throw new Error('Model not initialized');

    const xs = tf.tensor2d(X, [X.length, 1]);
    const predictions = this.model.predict(xs) as tf.Tensor;
    const data = predictions.dataSync();
    
    xs.dispose();
    predictions.dispose();

    return Array.from(data);
  }

  // Calcular loss sin entrenar
  calculateLoss(X: number[], Y: number[]): number {
    if (!this.model) throw new Error('Model not initialized');

    const xs = tf.tensor2d(X, [X.length, 1]);
    const ys = tf.tensor2d(Y, [Y.length, 1]);
    
    const predictions = this.model.predict(xs) as tf.Tensor;
    
    // Calcular MSE manualmente
    const diff = tf.sub(predictions, ys);
    const squared = tf.square(diff);
    const mse = tf.mean(squared);
    const loss = mse.dataSync()[0];
    
    xs.dispose();
    ys.dispose();
    predictions.dispose();
    diff.dispose();
    squared.dispose();
    mse.dispose();

    return loss;
  }

  // Obtener estado actual
  getState(X: number[], Y: number[]): TFNetworkState {
    const predictions = this.predict(X);
    const loss = this.calculateLoss(X, Y);

    return {
      epoch: this.epoch,
      loss,
      predictions,
      config: { ...this.config },
    };
  }

  // Obtener arquitectura como array
  getArchitecture(): number[] {
    return [1, ...this.config.layers.map(l => l.units)];
  }

  // Obtener pesos y biases del modelo
  getWeightsAndBiases(): { weights: number[][][]; biases: number[][] } {
    if (!this.model) {
      return { weights: [], biases: [] };
    }

    const weights: number[][][] = [];
    const biases: number[][] = [];

    // Cada capa densa tiene 2 tensores: kernel (pesos) y bias
    const modelWeights = this.model.getWeights();
    
    for (let i = 0; i < modelWeights.length; i += 2) {
      const kernelTensor = modelWeights[i];
      const biasTensor = modelWeights[i + 1];
      
      // kernel shape: [inputSize, outputSize]
      const kernelData = kernelTensor.arraySync() as number[][];
      const biasData = biasTensor.arraySync() as number[];
      
      // Convertir al formato esperado: weights[layer][toNeuron][fromNeuron]
      const layerWeights: number[][] = [];
      const outputSize = kernelData[0].length;
      const inputSize = kernelData.length;
      
      for (let to = 0; to < outputSize; to++) {
        const toNeuronWeights: number[] = [];
        for (let from = 0; from < inputSize; from++) {
          toNeuronWeights.push(kernelData[from][to]);
        }
        layerWeights.push(toNeuronWeights);
      }
      
      weights.push(layerWeights);
      biases.push(biasData);
    }

    return { weights, biases };
  }

  // Obtener activaciones intermedias para un input
  getActivations(x: number): { preActivation: number[]; activation: number[] }[] {
    if (!this.model) {
      return [];
    }

    const layers: { preActivation: number[]; activation: number[] }[] = [];
    
    // Capa de entrada
    layers.push({
      preActivation: [x],
      activation: [x],
    });

    // Obtener activaciones de cada capa
    let currentInput: tf.Tensor = tf.tensor2d([[x]]);
    
    for (let i = 0; i < this.model.layers.length; i++) {
      const layer = this.model.layers[i];
      const output = layer.apply(currentInput) as tf.Tensor;
      const outputData = output.dataSync();
      
      // Pre-activación (z) - obtener antes de activación es complejo en TF.js
      // Aproximamos usando los mismos valores
      layers.push({
        preActivation: Array.from(outputData),
        activation: Array.from(outputData),
      });
      
      currentInput.dispose();
      currentInput = output;
    }
    
    currentInput.dispose();
    
    return layers;
  }

  // Obtener patrón de una neurona específica (respuesta sobre rango de x)
  getNeuronPattern(layerIndex: number, neuronIndex: number, xRange: number[] = []): { x: number; y: number }[] {
    if (!this.model || layerIndex === 0) {
      // Para capa de entrada, retornar identidad
      if (xRange.length === 0) {
        xRange = Array.from({ length: 50 }, (_, i) => -6 + (12 * i) / 49);
      }
      return xRange.map(x => ({ x, y: x }));
    }

    if (xRange.length === 0) {
      xRange = Array.from({ length: 50 }, (_, i) => -6 + (12 * i) / 49);
    }

    const pattern: { x: number; y: number }[] = [];

    for (const x of xRange) {
      const activations = this.getActivations(x);
      if (activations[layerIndex] && activations[layerIndex].activation[neuronIndex] !== undefined) {
        pattern.push({
          x,
          y: activations[layerIndex].activation[neuronIndex],
        });
      }
    }

    return pattern;
  }

  // Obtener historial de entrenamiento
  getHistory(): { epoch: number; loss: number; valLoss?: number }[] {
    return [...this.trainingHistory];
  }

  // Reset
  reset(): void {
    this.epoch = 0;
    this.trainingHistory = [];
    this.buildModel();
  }

  // Reconfigurar
  reconfigure(newConfig: Partial<TFModelConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.epoch = 0;
    this.trainingHistory = [];
    this.buildModel();
  }

  // Actualizar solo learning rate (sin rebuild)
  setLearningRate(lr: number): void {
    this.config.learningRate = lr;
    if (this.model) {
      const optimizer = this.buildOptimizer();
      this.model.compile({
        optimizer,
        loss: this.config.loss as any,
        metrics: ['mse'],
      });
    }
  }

  // Actualizar solo momentum (sin rebuild)
  setMomentum(momentum: number): void {
    if (this.config.optimizer === 'momentum') {
      this.config.momentum = momentum;
      this.setLearningRate(this.config.learningRate); // Recompila
    }
  }

  // Limpiar recursos
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  // Obtener configuración actual
  getConfig(): TFModelConfig {
    return { ...this.config };
  }
}

// ============================================
// UTILIDADES
// ============================================

// Generar dataset de seno (compatible con el existente)
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

// Configuración por defecto
export const DEFAULT_TF_CONFIG: TFModelConfig = {
  layers: [
    { units: 16, activation: 'swish' },
    { units: 16, activation: 'tanh' },
    { units: 1, activation: 'linear' },
  ],
  loss: 'meanSquaredError',
  optimizer: 'adam',
  learningRate: 0.01,
};
