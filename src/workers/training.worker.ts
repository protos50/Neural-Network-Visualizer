/**
 * Web Worker para entrenamiento de redes neuronales
 * Ejecuta el entrenamiento en un hilo separado para no bloquear la UI
 */

// Tipos de mensajes
interface TrainMessage {
  type: 'train';
  mode: 'manual' | 'tensorflow';
  epochs: number;
  dataX: number[];
  dataY: number[];
  config: any;
}

interface InitMessage {
  type: 'init';
  mode: 'manual' | 'tensorflow';
  config: any;
}

interface StopMessage {
  type: 'stop';
}

interface ResetMessage {
  type: 'reset';
  config: any;
}

type WorkerMessage = TrainMessage | InitMessage | StopMessage | ResetMessage;

interface TrainingResult {
  type: 'result';
  loss: number;
  epoch: number;
  predictions: number[];
  weights: number[][][];
  biases: number[][];
  activations: { preActivation: number[]; activation: number[] }[];
}

interface ErrorResult {
  type: 'error';
  message: string;
}

// ============================================
// Implementación de Red Neuronal Manual (copia simplificada)
// ============================================

type ActivationFn = (x: number) => number;

const activations: Record<string, { fn: ActivationFn; derivative: ActivationFn }> = {
  tanh: {
    fn: (x) => Math.tanh(x),
    derivative: (x) => 1 - Math.tanh(x) ** 2,
  },
  relu: {
    fn: (x) => Math.max(0, x),
    derivative: (x) => (x > 0 ? 1 : 0),
  },
  sigmoid: {
    fn: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
  },
  linear: {
    fn: (x) => x,
    derivative: () => 1,
  },
};

class ManualNetwork {
  private weights: number[][][] = [];
  private biases: number[][] = [];
  private velocityW: number[][][] = [];
  private velocityB: number[][] = [];
  private config: any;
  private epoch = 0;

  constructor(config: any) {
    this.config = config;
    this.initializeWeights();
  }

  private initializeWeights() {
    const layers = [1, ...this.config.hiddenLayers, 1];
    this.weights = [];
    this.biases = [];
    this.velocityW = [];
    this.velocityB = [];

    for (let l = 1; l < layers.length; l++) {
      const prevSize = layers[l - 1];
      const currSize = layers[l];
      const scale = Math.sqrt(2 / (prevSize + currSize));

      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      const layerVelW: number[][] = [];
      const layerVelB: number[] = [];

      for (let j = 0; j < currSize; j++) {
        const neuronWeights: number[] = [];
        const neuronVelW: number[] = [];
        for (let i = 0; i < prevSize; i++) {
          neuronWeights.push((Math.random() * 2 - 1) * scale);
          neuronVelW.push(0);
        }
        layerWeights.push(neuronWeights);
        layerVelW.push(neuronVelW);
        layerBiases.push(0);
        layerVelB.push(0);
      }

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
      this.velocityW.push(layerVelW);
      this.velocityB.push(layerVelB);
    }
  }

  private forward(input: number[]): { preActivations: number[][]; activationOutputs: number[][] } {
    const preActivations: number[][] = [[...input]];
    const activationOutputs: number[][] = [[...input]];

    let current = input;

    for (let l = 0; l < this.weights.length; l++) {
      const isOutput = l === this.weights.length - 1;
      const actName = isOutput ? this.config.outputActivation : this.config.hiddenActivation;
      const actFn = activations[actName]?.fn || activations.tanh.fn;

      const pre: number[] = [];
      const post: number[] = [];

      for (let j = 0; j < this.weights[l].length; j++) {
        let sum = this.biases[l][j];
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * this.weights[l][j][i];
        }
        pre.push(sum);
        post.push(actFn(sum));
      }

      preActivations.push(pre);
      activationOutputs.push(post);
      current = post;
    }

    return { preActivations, activationOutputs };
  }

  trainEpoch(X: number[], Y: number[]): number {
    let totalLoss = 0;
    const lr = this.config.learningRate;
    const momentum = this.config.momentum || 0;

    for (let s = 0; s < X.length; s++) {
      const { preActivations, activationOutputs } = this.forward([X[s]]);
      const output = activationOutputs[activationOutputs.length - 1][0];
      const error = output - Y[s];
      totalLoss += error ** 2;

      // Backprop
      const deltas: number[][] = [];
      const outputDelta = [error * (activations[this.config.outputActivation]?.derivative || activations.linear.derivative)(preActivations[preActivations.length - 1][0])];
      deltas.unshift(outputDelta);

      for (let l = this.weights.length - 2; l >= 0; l--) {
        const actName = this.config.hiddenActivation;
        const derivative = activations[actName]?.derivative || activations.tanh.derivative;
        const layerDelta: number[] = [];

        for (let j = 0; j < this.weights[l].length; j++) {
          let sum = 0;
          for (let k = 0; k < this.weights[l + 1].length; k++) {
            sum += deltas[0][k] * this.weights[l + 1][k][j];
          }
          layerDelta.push(sum * derivative(preActivations[l + 1][j]));
        }
        deltas.unshift(layerDelta);
      }

      // Update weights
      for (let l = 0; l < this.weights.length; l++) {
        for (let j = 0; j < this.weights[l].length; j++) {
          for (let i = 0; i < this.weights[l][j].length; i++) {
            const grad = deltas[l][j] * activationOutputs[l][i];
            this.velocityW[l][j][i] = momentum * this.velocityW[l][j][i] - lr * grad;
            this.weights[l][j][i] += this.velocityW[l][j][i];
          }
          const biasGrad = deltas[l][j];
          this.velocityB[l][j] = momentum * this.velocityB[l][j] - lr * biasGrad;
          this.biases[l][j] += this.velocityB[l][j];
        }
      }
    }

    this.epoch++;
    return totalLoss / X.length;
  }

  predict(X: number[]): number[] {
    return X.map(x => {
      const { activationOutputs } = this.forward([x]);
      return activationOutputs[activationOutputs.length - 1][0];
    });
  }

  getState() {
    const layers = [1, ...this.config.hiddenLayers, 1];
    const layerStates = layers.map(size => ({
      preActivation: new Array(size).fill(0),
      activation: new Array(size).fill(0.5),
    }));

    return {
      weights: this.weights,
      biases: this.biases,
      activations: layerStates,
      epoch: this.epoch,
    };
  }

  reset() {
    this.epoch = 0;
    this.initializeWeights();
  }
}

// ============================================
// Estado del Worker
// ============================================

let manualNetwork: ManualNetwork | null = null;
let isTraining = false;
let currentEpoch = 0;

// ============================================
// Message Handler
// ============================================

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const { data } = e;

  switch (data.type) {
    case 'init':
      if (data.mode === 'manual') {
        manualNetwork = new ManualNetwork(data.config);
        currentEpoch = 0;
        self.postMessage({ type: 'ready', mode: 'manual' });
      }
      break;

    case 'train':
      if (data.mode === 'manual' && manualNetwork) {
        isTraining = true;
        let loss = 0;

        for (let i = 0; i < data.epochs && isTraining; i++) {
          loss = manualNetwork.trainEpoch(data.dataX, data.dataY);
          currentEpoch++;
        }

        const predictions = manualNetwork.predict(data.dataX);
        const state = manualNetwork.getState();

        const result: TrainingResult = {
          type: 'result',
          loss,
          epoch: currentEpoch,
          predictions,
          weights: state.weights,
          biases: state.biases,
          activations: state.activations,
        };

        self.postMessage(result);
      }
      break;

    case 'stop':
      isTraining = false;
      break;

    case 'reset':
      if (manualNetwork) {
        manualNetwork = new ManualNetwork(data.config);
        currentEpoch = 0;
      }
      self.postMessage({ type: 'reset_done' });
      break;
  }
};

export {};
