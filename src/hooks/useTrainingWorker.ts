'use client';

import { useRef, useCallback, useEffect, useState } from 'react';
import type { NetworkConfig } from '@/lib/neural-network';

interface TrainingResult {
  loss: number;
  epoch: number;
  predictions: number[];
  weights: number[][][];
  biases: number[][];
  activations: { preActivation: number[]; activation: number[] }[];
}

interface UseTrainingWorkerOptions {
  onResult: (result: TrainingResult) => void;
  onReady?: () => void;
  onError?: (error: string) => void;
}

export function useTrainingWorker(options: UseTrainingWorkerOptions) {
  const workerRef = useRef<Worker | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isSupported, setIsSupported] = useState(true);

  // Inicializar worker
  useEffect(() => {
    // Verificar si Web Workers están soportados
    if (typeof window === 'undefined' || !window.Worker) {
      setIsSupported(false);
      return;
    }

    try {
      // Crear worker inline para evitar problemas con Next.js
      const workerCode = `
        // Implementación simplificada de la red neuronal en el worker
        const activations = {
          tanh: { fn: x => Math.tanh(x), derivative: x => 1 - Math.tanh(x) ** 2 },
          relu: { fn: x => Math.max(0, x), derivative: x => x > 0 ? 1 : 0 },
          sigmoid: { fn: x => { const s = 1/(1+Math.exp(-x)); return s; }, derivative: x => { const s = 1/(1+Math.exp(-x)); return s*(1-s); } },
          linear: { fn: x => x, derivative: () => 1 },
        };

        class ManualNetwork {
          constructor(config) {
            this.config = config;
            this.epoch = 0;
            this.initializeWeights();
          }

          initializeWeights() {
            const layers = [1, ...this.config.hiddenLayers, 1];
            this.weights = [];
            this.biases = [];
            this.velocityW = [];
            this.velocityB = [];

            for (let l = 1; l < layers.length; l++) {
              const prevSize = layers[l - 1];
              const currSize = layers[l];
              const scale = Math.sqrt(2 / (prevSize + currSize));

              const layerWeights = [];
              const layerBiases = [];
              const layerVelW = [];
              const layerVelB = [];

              for (let j = 0; j < currSize; j++) {
                const neuronWeights = [];
                const neuronVelW = [];
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

          forward(input) {
            const preActivations = [[...input]];
            const activationOutputs = [[...input]];
            let current = input;

            for (let l = 0; l < this.weights.length; l++) {
              const isOutput = l === this.weights.length - 1;
              const actName = isOutput ? this.config.outputActivation : this.config.hiddenActivation;
              const actFn = activations[actName]?.fn || activations.tanh.fn;

              const pre = [];
              const post = [];

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

          trainEpoch(X, Y) {
            let totalLoss = 0;
            const lr = this.config.learningRate;
            const momentum = this.config.momentum || 0;

            for (let s = 0; s < X.length; s++) {
              const { preActivations, activationOutputs } = this.forward([X[s]]);
              const output = activationOutputs[activationOutputs.length - 1][0];
              const error = output - Y[s];
              totalLoss += error ** 2;

              const deltas = [];
              const outputDerivative = activations[this.config.outputActivation]?.derivative || activations.linear.derivative;
              const outputDelta = [error * outputDerivative(preActivations[preActivations.length - 1][0])];
              deltas.unshift(outputDelta);

              for (let l = this.weights.length - 2; l >= 0; l--) {
                const derivative = activations[this.config.hiddenActivation]?.derivative || activations.tanh.derivative;
                const layerDelta = [];

                for (let j = 0; j < this.weights[l].length; j++) {
                  let sum = 0;
                  for (let k = 0; k < this.weights[l + 1].length; k++) {
                    sum += deltas[0][k] * this.weights[l + 1][k][j];
                  }
                  layerDelta.push(sum * derivative(preActivations[l + 1][j]));
                }
                deltas.unshift(layerDelta);
              }

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

          predict(X) {
            return X.map(x => {
              const { activationOutputs } = this.forward([x]);
              return activationOutputs[activationOutputs.length - 1][0];
            });
          }

          getActivations(testX) {
            const { preActivations, activationOutputs } = this.forward([testX]);
            return preActivations.map((pre, i) => ({
              preActivation: pre,
              activation: activationOutputs[i],
            }));
          }

          getState() {
            return {
              weights: this.weights,
              biases: this.biases,
              epoch: this.epoch,
            };
          }
        }

        let network = null;
        let isTraining = false;

        self.onmessage = function(e) {
          const { data } = e;

          switch (data.type) {
            case 'init':
              network = new ManualNetwork(data.config);
              self.postMessage({ type: 'ready' });
              break;

            case 'train':
              if (!network) return;
              isTraining = true;
              
              let loss = 0;
              for (let i = 0; i < data.epochs && isTraining; i++) {
                loss = network.trainEpoch(data.dataX, data.dataY);
              }

              const predictions = network.predict(data.dataX);
              const state = network.getState();
              const activations = network.getActivations(0);

              self.postMessage({
                type: 'result',
                loss,
                epoch: state.epoch,
                predictions,
                weights: state.weights,
                biases: state.biases,
                activations,
              });
              break;

            case 'stop':
              isTraining = false;
              break;

            case 'reset':
              network = new ManualNetwork(data.config);
              self.postMessage({ type: 'reset_done' });
              break;

            case 'predict':
              if (!network) return;
              const preds = network.predict(data.dataX);
              self.postMessage({ type: 'predictions', predictions: preds });
              break;
          }
        };
      `;

      const blob = new Blob([workerCode], { type: 'application/javascript' });
      const workerUrl = URL.createObjectURL(blob);
      workerRef.current = new Worker(workerUrl);

      workerRef.current.onmessage = (e) => {
        const { data } = e;
        
        if (data.type === 'ready' || data.type === 'reset_done') {
          setIsReady(true);
          options.onReady?.();
        } else if (data.type === 'result') {
          options.onResult(data as TrainingResult);
        } else if (data.type === 'error') {
          options.onError?.(data.message);
        }
      };

      workerRef.current.onerror = (error) => {
        console.error('Worker error:', error);
        options.onError?.(error.message);
        setIsSupported(false);
      };

      return () => {
        workerRef.current?.terminate();
        URL.revokeObjectURL(workerUrl);
      };
    } catch (error) {
      console.error('Failed to create worker:', error);
      setIsSupported(false);
    }
  }, []);

  // Inicializar red en el worker
  const initNetwork = useCallback((config: NetworkConfig) => {
    if (!workerRef.current || !isSupported) return false;
    
    setIsReady(false);
    workerRef.current.postMessage({
      type: 'init',
      config,
    });
    return true;
  }, [isSupported]);

  // Entrenar épocas
  const train = useCallback((dataX: number[], dataY: number[], epochs: number) => {
    if (!workerRef.current || !isReady || !isSupported) return false;
    
    workerRef.current.postMessage({
      type: 'train',
      dataX,
      dataY,
      epochs,
    });
    return true;
  }, [isReady, isSupported]);

  // Detener entrenamiento
  const stop = useCallback(() => {
    workerRef.current?.postMessage({ type: 'stop' });
  }, []);

  // Resetear red
  const reset = useCallback((config: NetworkConfig) => {
    if (!workerRef.current || !isSupported) return false;
    
    setIsReady(false);
    workerRef.current.postMessage({
      type: 'reset',
      config,
    });
    return true;
  }, [isSupported]);

  return {
    isReady,
    isSupported,
    initNetwork,
    train,
    stop,
    reset,
  };
}
