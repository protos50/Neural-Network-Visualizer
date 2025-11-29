'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { NeuralNetwork, type NetworkState, type NetworkConfig, type NeuronInfo } from '@/lib/neural-network';
import { TensorFlowNetwork, type TFModelConfig, DEFAULT_TF_CONFIG } from '@/lib/neural-network-tfjs';
import { 
  generateDataset, 
  createDatasetFromCSV,
  generateTestPoints,
  functionDescriptions,
  type DatasetConfig, 
  DEFAULT_DATASET_CONFIG 
} from '@/lib/dataset-generator';
import OscilloscopeCanvas from '@/components/OscilloscopeCanvas';
import NeuralNetworkViz from '@/components/NeuralNetworkViz';
import TrainingPanel from '@/components/TrainingPanel';
import TensorFlowPanel from '@/components/TensorFlowPanel';
import DatasetPanel from '@/components/DatasetPanel';
import ModeSelector, { type NetworkMode } from '@/components/ModeSelector';
import { useI18n, LanguageSelector } from '@/lib/i18n';
import { Brain, Waves, Github, GraduationCap, FlaskConical } from 'lucide-react';

const DEFAULT_CONFIG: NetworkConfig = {
  hiddenLayers: [8, 8],
  hiddenActivation: 'tanh',
  outputActivation: 'linear',
  learningRate: 0.01,
  momentum: 0,
};

export default function Home() {
  const { t } = useI18n();
  
  // ==========================================
  // MODO DE RED (Manual vs TensorFlow.js)
  // ==========================================
  const [networkMode, setNetworkMode] = useState<NetworkMode>('manual');
  
  // Red neuronal manual
  const networkRef = useRef<NeuralNetwork | null>(null);
  
  // Red neuronal TensorFlow.js
  const tfNetworkRef = useRef<TensorFlowNetwork | null>(null);
  const [tfConfig, setTFConfig] = useState<TFModelConfig>(DEFAULT_TF_CONFIG);
  
  // Dataset (compartido)
  const [dataset, setDataset] = useState<{
    X: number[];
    Y: number[];
    YTrue: number[];
  }>({ X: [], Y: [], YTrue: [] });
  
  // Estado del entrenamiento (compartido)
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [networkState, setNetworkState] = useState<NetworkState>({
    weights: [],
    biases: [],
    layers: [],
    loss: 1,
    epoch: 0,
    config: DEFAULT_CONFIG,
  });
  
  // Controles
  const [isTraining, setIsTraining] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [config, setConfig] = useState<NetworkConfig>(DEFAULT_CONFIG);
  
  // Dataset configuration
  const [datasetConfig, setDatasetConfig] = useState<DatasetConfig>(DEFAULT_DATASET_CONFIG);
  
  // Modo de test (animación cíclica)
  const [isTestMode, setIsTestMode] = useState(false);
  const [testPoints, setTestPoints] = useState<{ x: number; y: number }[]>([]);
  const [testIndex, setTestIndex] = useState(0);
  const [testPredictions, setTestPredictions] = useState<{ x: number; yTrue: number; yPred: number }[]>([]);
  
  // ==========================================
  // INICIALIZACIÓN
  // ==========================================
  useEffect(() => {
    // Crear red neuronal manual por defecto
    networkRef.current = new NeuralNetwork(config);
    
    // Generar dataset usando el nuevo generador
    const data = generateDataset(datasetConfig);
    setDataset(data);
    
    // Estado inicial
    if (networkRef.current) {
      const preds = new Array(data.X.length).fill(0);
      setPredictions(preds);
      const state = networkRef.current.getInitialState();
      setNetworkState(state);
      setEpoch(0);
      setLoss(1);
    }
    
    // Cleanup TensorFlow al desmontar
    return () => {
      if (tfNetworkRef.current) {
        tfNetworkRef.current.dispose();
      }
    };
  }, []);
  
  // ==========================================
  // CAMBIO DE MODO
  // ==========================================
  const handleModeChange = useCallback((mode: NetworkMode) => {
    setIsTraining(false);
    setNetworkMode(mode);
    
    // Nuevo dataset usando config actual
    const data = generateDataset(datasetConfig);
    setDataset(data);
    
    if (mode === 'tensorflow') {
      // Limpiar red TF anterior si existe
      if (tfNetworkRef.current) {
        tfNetworkRef.current.dispose();
      }
      // Crear nueva red TensorFlow.js
      tfNetworkRef.current = new TensorFlowNetwork(tfConfig);
      
      const preds = new Array(data.X.length).fill(0);
      setPredictions(preds);
      setEpoch(0);
      setLoss(1);
      
      // Obtener pesos y biases iniciales
      const { weights, biases } = tfNetworkRef.current.getWeightsAndBiases();
      const activations = tfNetworkRef.current.getActivations(0);
      const tfArch = tfNetworkRef.current.getArchitecture();
      const hiddenAct = tfConfig.layers[0]?.activation || 'swish';
      const outputAct = tfConfig.layers[tfConfig.layers.length - 1]?.activation || 'linear';
      
      setNetworkState({
        weights,
        biases,
        layers: activations.length > 0 
          ? activations.map(act => ({
              preActivation: act.preActivation,
              activation: act.activation.map(a => Math.abs(a)),
            }))
          : tfArch.map(count => ({
              preActivation: new Array(count).fill(0.1),
              activation: new Array(count).fill(0.1),
            })),
        loss: 1,
        epoch: 0,
        config: {
          hiddenLayers: tfConfig.layers.slice(0, -1).map(l => l.units),
          hiddenActivation: hiddenAct as any,
          outputActivation: outputAct as any,
          learningRate: tfConfig.learningRate,
          momentum: 0,
        },
      });
    } else {
      // Volver a modo manual
      networkRef.current = new NeuralNetwork(config);
      
      if (networkRef.current) {
        const preds = new Array(data.X.length).fill(0);
        setPredictions(preds);
        const state = networkRef.current.getInitialState();
        setNetworkState(state);
        setEpoch(0);
        setLoss(1);
      }
    }
  }, [config, tfConfig, datasetConfig]);
  
  // ==========================================
  // LOOP DE ENTRENAMIENTO (DUAL) - Optimizado con requestAnimationFrame
  // ==========================================
  useEffect(() => {
    if (!isTraining || dataset.X.length === 0) return;
    
    let isRunning = true;
    let lastTime = 0;
    let accumulatedTime = 0;
    let animationFrameId: number;
    
    // Intervalo base: más lento a bajas velocidades, más rápido a altas
    const getTargetInterval = () => speed < 1 ? 200 / speed : 16; // ~60fps max
    
    const runTraining = async (timestamp: number) => {
      if (!isRunning) return;
      
      // Calcular delta time para animación suave
      const deltaTime = lastTime ? timestamp - lastTime : 16;
      lastTime = timestamp;
      accumulatedTime += deltaTime;
      
      const targetInterval = getTargetInterval();
      
      // Solo entrenar si pasó suficiente tiempo
      if (accumulatedTime < targetInterval) {
        animationFrameId = requestAnimationFrame(runTraining);
        return;
      }
      
      // Reset acumulador
      accumulatedTime = 0;
      
      // Calcular épocas: limitar a 10 por frame para mantener fluidez
      const epochsThisTick = Math.min(10, Math.max(1, Math.floor(speed)));
      
      // ========== MODO MANUAL ==========
      if (networkMode === 'manual' && networkRef.current) {
        let currentLoss = 0;
        for (let i = 0; i < epochsThisTick; i++) {
          currentLoss = networkRef.current.trainEpoch(dataset.X, dataset.Y);
        }
        
        const preds = networkRef.current.predict(dataset.X);
        const state = networkRef.current.getState(dataset.X, dataset.Y);
        
        setPredictions(preds);
        setNetworkState(state);
        setEpoch(state.epoch);
        setLoss(currentLoss);
        
        if (currentLoss < 0.0005) {
          setIsTraining(false);
          return;
        }
      }
      // ========== MODO TENSORFLOW.JS ==========
      else if (networkMode === 'tensorflow' && tfNetworkRef.current) {
        try {
          const currentLoss = await tfNetworkRef.current.trainEpochs(
            dataset.X,
            dataset.Y,
            epochsThisTick
          );
          
          if (!isRunning) return;
          
          const preds = tfNetworkRef.current.predict(dataset.X);
          const currentEpoch = tfNetworkRef.current.epoch;
          
          setPredictions(preds);
          setEpoch(currentEpoch);
          setLoss(currentLoss);
          
          // Obtener pesos y biases reales del modelo TF.js
          const { weights, biases } = tfNetworkRef.current.getWeightsAndBiases();
          
          // Obtener activaciones reales para un punto de prueba (x=0)
          const activations = tfNetworkRef.current.getActivations(0);
          
          const tfArch = tfNetworkRef.current.getArchitecture();
          const hiddenAct = tfConfig.layers[0]?.activation || 'swish';
          const outputAct = tfConfig.layers[tfConfig.layers.length - 1]?.activation || 'linear';
          
          setNetworkState({
            weights,
            biases,
            layers: activations.length > 0 
              ? activations.map(act => ({
                  preActivation: act.preActivation,
                  activation: act.activation.map(a => Math.abs(a)),
                }))
              : tfArch.map(count => ({
                  preActivation: new Array(count).fill(0.5),
                  activation: new Array(count).fill(0.5),
                })),
            loss: currentLoss,
            epoch: currentEpoch,
            config: {
              hiddenLayers: tfConfig.layers.slice(0, -1).map(l => l.units),
              hiddenActivation: hiddenAct as any,
              outputActivation: outputAct as any,
              learningRate: tfConfig.learningRate,
              momentum: 0,
            },
          });
          
          if (currentLoss < 0.0005) {
            setIsTraining(false);
            return;
          }
        } catch (error) {
          console.error('TensorFlow training error:', error);
        }
      }
      
      // Continuar el loop
      if (isRunning) {
        animationFrameId = requestAnimationFrame(runTraining);
      }
    };
    
    // Iniciar loop
    animationFrameId = requestAnimationFrame(runTraining);
    
    return () => {
      isRunning = false;
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isTraining, speed, dataset, networkMode, tfConfig]);
  
  // ==========================================
  // HANDLERS MODO MANUAL
  // ==========================================
  const handleConfigChange = useCallback((newConfig: Partial<NetworkConfig>) => {
    if (networkMode !== 'manual') return;
    
    setIsTraining(false);
    
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);
    
    const structuralChange = 
      newConfig.hiddenLayers !== undefined ||
      newConfig.hiddenActivation !== undefined ||
      newConfig.outputActivation !== undefined;
    
    if (structuralChange) {
      networkRef.current = new NeuralNetwork(updatedConfig);
      const data = generateDataset(datasetConfig);
      setDataset(data);
      
      if (networkRef.current) {
        const preds = new Array(data.X.length).fill(0);
        setPredictions(preds);
        const state = networkRef.current.getInitialState();
        setNetworkState(state);
        setEpoch(0);
        setLoss(1);
      }
    } else {
      if (networkRef.current) {
        if (newConfig.learningRate !== undefined) {
          networkRef.current.setLearningRate(newConfig.learningRate);
        }
        if (newConfig.momentum !== undefined) {
          networkRef.current.setMomentum(newConfig.momentum);
        }
      }
    }
  }, [config, networkMode, datasetConfig]);
  
  // ==========================================
  // HANDLERS MODO TENSORFLOW
  // ==========================================
  const handleTFConfigChange = useCallback((newConfig: Partial<TFModelConfig>) => {
    if (networkMode !== 'tensorflow') return;
    
    setIsTraining(false);
    
    const updatedConfig = { ...tfConfig, ...newConfig };
    setTFConfig(updatedConfig);
    
    // Recrear red TensorFlow
    if (tfNetworkRef.current) {
      tfNetworkRef.current.dispose();
    }
    tfNetworkRef.current = new TensorFlowNetwork(updatedConfig);
    
    const data = generateDataset(datasetConfig);
    setDataset(data);
    
    const preds = new Array(data.X.length).fill(0);
    setPredictions(preds);
    setEpoch(0);
    setLoss(1);
    
    // Obtener pesos y biases iniciales
    const { weights, biases } = tfNetworkRef.current.getWeightsAndBiases();
    const activations = tfNetworkRef.current.getActivations(0);
    const tfArch = tfNetworkRef.current.getArchitecture();
    const hiddenAct = updatedConfig.layers[0]?.activation || 'swish';
    const outputAct = updatedConfig.layers[updatedConfig.layers.length - 1]?.activation || 'linear';
    
    setNetworkState({
      weights,
      biases,
      layers: activations.length > 0 
        ? activations.map(act => ({
            preActivation: act.preActivation,
            activation: act.activation.map(a => Math.abs(a)),
          }))
        : tfArch.map(count => ({
            preActivation: new Array(count).fill(0.1),
            activation: new Array(count).fill(0.1),
          })),
      loss: 1,
      epoch: 0,
      config: {
        hiddenLayers: updatedConfig.layers.slice(0, -1).map(l => l.units),
        hiddenActivation: hiddenAct as any,
        outputActivation: outputAct as any,
        learningRate: updatedConfig.learningRate,
        momentum: 0,
      },
    });
  }, [tfConfig, networkMode, datasetConfig]);
  
  // ==========================================
  // RESET Y STOP (DUAL)
  // ==========================================
  const handleReset = useCallback(() => {
    setIsTraining(false);
    setIsTestMode(false);
    
    const data = generateDataset(datasetConfig);
    setDataset(data);
    
    if (networkMode === 'manual') {
      networkRef.current = new NeuralNetwork(config);
      if (networkRef.current) {
        const preds = new Array(data.X.length).fill(0);
        setPredictions(preds);
        const state = networkRef.current.getInitialState();
        setNetworkState(state);
        setEpoch(0);
        setLoss(1);
      }
    } else {
      if (tfNetworkRef.current) {
        tfNetworkRef.current.dispose();
      }
      tfNetworkRef.current = new TensorFlowNetwork(tfConfig);
      
      const preds = new Array(data.X.length).fill(0);
      setPredictions(preds);
      setEpoch(0);
      setLoss(1);
      
      // Obtener pesos y biases iniciales
      const { weights, biases } = tfNetworkRef.current.getWeightsAndBiases();
      const activations = tfNetworkRef.current.getActivations(0);
      const tfArch = tfNetworkRef.current.getArchitecture();
      const hiddenAct = tfConfig.layers[0]?.activation || 'swish';
      const outputAct = tfConfig.layers[tfConfig.layers.length - 1]?.activation || 'linear';
      
      setNetworkState({
        weights,
        biases,
        layers: activations.length > 0 
          ? activations.map(act => ({
              preActivation: act.preActivation,
              activation: act.activation.map(a => Math.abs(a)),
            }))
          : tfArch.map(count => ({
              preActivation: new Array(count).fill(0.1),
              activation: new Array(count).fill(0.1),
            })),
        loss: 1,
        epoch: 0,
        config: {
          hiddenLayers: tfConfig.layers.slice(0, -1).map(l => l.units),
          hiddenActivation: hiddenAct as any,
          outputActivation: outputAct as any,
          learningRate: tfConfig.learningRate,
          momentum: 0,
        },
      });
    }
  }, [config, tfConfig, networkMode, datasetConfig]);
  
  const handleStop = useCallback(() => {
    setIsTraining(false);
  }, []);
  
  // ==========================================
  // HANDLERS DE DATASET
  // ==========================================
  const handleDatasetConfigChange = useCallback((newConfig: DatasetConfig) => {
    setDatasetConfig(newConfig);
    setIsTraining(false);
    setIsTestMode(false);
    
    // Regenerar dataset
    const data = generateDataset(newConfig);
    setDataset(data);
    
    // Resetear predicciones
    const preds = new Array(data.X.length).fill(0);
    setPredictions(preds);
    setEpoch(0);
    setLoss(1);
    
    // Resetear red
    if (networkMode === 'manual' && networkRef.current) {
      networkRef.current.reset();
      setNetworkState(networkRef.current.getInitialState());
    } else if (networkMode === 'tensorflow' && tfNetworkRef.current) {
      tfNetworkRef.current.dispose();
      tfNetworkRef.current = new TensorFlowNetwork(tfConfig);
      const { weights, biases } = tfNetworkRef.current.getWeightsAndBiases();
      const activations = tfNetworkRef.current.getActivations(0);
      const tfArch = tfNetworkRef.current.getArchitecture();
      setNetworkState(prev => ({
        ...prev,
        weights,
        biases,
        layers: activations.map(act => ({
          preActivation: act.preActivation,
          activation: act.activation.map(a => Math.abs(a)),
        })),
      }));
    }
  }, [networkMode, tfConfig]);

  const handleImportCSV = useCallback((X: number[], Y: number[]) => {
    setIsTraining(false);
    setIsTestMode(false);
    
    const data = createDatasetFromCSV(X, Y);
    setDataset(data);
    setDatasetConfig(data.config);
    
    // Resetear predicciones
    const preds = new Array(data.X.length).fill(0);
    setPredictions(preds);
    setEpoch(0);
    setLoss(1);
  }, []);

  const handleToggleTestMode = useCallback(() => {
    if (!isTestMode) {
      // Activar modo test
      setIsTraining(false);
      setIsTestMode(true);
      
      // Generar puntos de test
      const points = generateTestPoints(datasetConfig, 5, 100);
      setTestPoints(points);
      setTestIndex(0);
      setTestPredictions([]);
    } else {
      // Desactivar modo test
      setIsTestMode(false);
      setTestPredictions([]);
    }
  }, [isTestMode, datasetConfig]);

  // ==========================================
  // ANIMACIÓN MODO TEST (onda cíclica)
  // ==========================================
  useEffect(() => {
    if (!isTestMode || testPoints.length === 0) return;

    const animationSpeed = 50; // ms entre puntos
    const windowSize = 100; // Puntos visibles en pantalla

    const interval = setInterval(() => {
      setTestIndex(prev => {
        const nextIndex = (prev + 1) % testPoints.length;
        
        // Predecir el punto actual
        const point = testPoints[nextIndex];
        let prediction = 0;
        
        if (networkMode === 'manual' && networkRef.current) {
          prediction = networkRef.current.predict([point.x])[0];
        } else if (networkMode === 'tensorflow' && tfNetworkRef.current) {
          prediction = tfNetworkRef.current.predict([point.x])[0];
        }
        
        // Agregar a predicciones del test (mantener ventana)
        setTestPredictions(prev => {
          const newPreds = [...prev, { x: point.x, yTrue: point.y, yPred: prediction }];
          // Mantener solo los últimos N puntos para la animación
          return newPreds.slice(-windowSize);
        });
        
        return nextIndex;
      });
    }, animationSpeed);

    return () => clearInterval(interval);
  }, [isTestMode, testPoints, networkMode]);
  
  // ==========================================
  // HELPERS PARA VISUALIZACIÓN
  // ==========================================
  const getNeuronInfo = useCallback((layer: number, index: number): NeuronInfo => {
    if (networkMode === 'manual' && networkRef.current) {
      return networkRef.current.getNeuronInfo(layer, index);
    }
    
    // Info para TensorFlow.js - usar datos reales del estado
    const numLayers = networkState.layers.length;
    const isInput = layer === 0;
    const isOutput = layer === numLayers - 1;
    
    // Determinar activación según la capa
    let activation: string = 'linear';
    if (!isInput && tfConfig.layers.length > 0) {
      const layerConfigIndex = layer - 1;
      if (layerConfigIndex >= 0 && layerConfigIndex < tfConfig.layers.length) {
        activation = tfConfig.layers[layerConfigIndex].activation;
      }
    }
    
    // Obtener valores de activación del estado
    const layerState = networkState.layers[layer];
    const currentOutput = layerState?.activation?.[index] ?? 0;
    const currentInput = layerState?.preActivation?.[index] ?? 0;
    
    // Obtener pesos reales de esta neurona (hacia ella desde la capa anterior)
    let weights: number[] = [];
    let bias = 0;
    
    if (!isInput && layer > 0 && networkState.weights.length > 0) {
      const weightsLayerIndex = layer - 1; // Los pesos están indexados desde 0
      if (networkState.weights[weightsLayerIndex] && networkState.weights[weightsLayerIndex][index]) {
        weights = networkState.weights[weightsLayerIndex][index];
      }
      if (networkState.biases[weightsLayerIndex]) {
        bias = networkState.biases[weightsLayerIndex][index] ?? 0;
      }
    }
    
    // Determinar interpretación
    let patternType: 'inputSensitivity' | 'combinesHidden' | 'maxWeight' | 'receivesX' = 'combinesHidden';
    let patternValue = 0;
    let patternSign = '';
    let patternNeuronNum = 0;
    
    if (isInput) {
      patternType = 'receivesX';
    } else if (weights.length > 0) {
      // Encontrar el peso máximo
      let maxWeight = 0;
      let maxIdx = 0;
      weights.forEach((w, i) => {
        if (Math.abs(w) > Math.abs(maxWeight)) {
          maxWeight = w;
          maxIdx = i;
        }
      });
      patternType = isOutput ? 'combinesHidden' : 'maxWeight';
      patternValue = Math.abs(maxWeight);
      patternSign = maxWeight > 0 ? '+' : '-';
      patternNeuronNum = maxIdx + 1;
    }
    
    return {
      layer,
      layerType: isInput ? 'input' : isOutput ? 'output' : 'hidden',
      layerNumber: !isInput && !isOutput ? layer : undefined,
      index,
      weights,
      bias,
      activation: activation as any,
      currentInput,
      currentOutput,
      patternInfo: { 
        type: patternType,
        value: patternValue,
        sign: patternSign,
        neuronNum: patternNeuronNum,
      },
      tfInfo: {
        optimizer: tfConfig.optimizer,
        loss: tfConfig.loss,
        learningRate: tfConfig.learningRate,
      },
    };
  }, [networkMode, networkState, tfConfig]);
  
  const getNeuronPattern = useCallback((layer: number, index: number): { x: number; y: number }[] => {
    if (networkMode === 'manual' && networkRef.current) {
      return networkRef.current.getNeuronPattern(layer, index);
    }
    // Para TensorFlow.js, usar el método del modelo
    if (networkMode === 'tensorflow' && tfNetworkRef.current) {
      return tfNetworkRef.current.getNeuronPattern(layer, index);
    }
    return [];
  }, [networkMode]);
  
  // Convertir datos para el canvas
  // En modo test, mostrar los datos de la animación cíclica
  const dataPoints = isTestMode && testPredictions.length > 0
    ? testPredictions.map(p => ({ x: p.x, y: p.yTrue }))
    : dataset.X.map((x, i) => ({ x, y: dataset.Y[i] }));
  
  const predictionPoints = isTestMode && testPredictions.length > 0
    ? testPredictions.map(p => ({ x: p.x, y: p.yPred }))
    : dataset.X.map((x, i) => ({ x, y: predictions[i] || 0 }));
  
  const trueSinePoints = isTestMode && testPredictions.length > 0
    ? testPredictions.map(p => ({ x: p.x, y: p.yTrue }))
    : dataset.X.map((x, i) => ({ x, y: dataset.YTrue[i] }));
  
  // Arquitectura actual como string
  const currentArchitecture = networkMode === 'manual'
    ? `1 → ${config.hiddenLayers.join(' → ')} → 1`
    : `1 → ${tfConfig.layers.map(l => l.units).join(' → ')}`;

  return (
    <main className="min-h-screen p-4 md:p-6 flex flex-col">
      {/* Header */}
      <div className="text-center mb-4">
        <div className="flex items-center justify-center gap-3 mb-1">
          <Brain className="w-7 h-7 text-crt-green animate-pulse" />
          <h1 className="text-2xl md:text-3xl font-bold glow-text tracking-wider">
            {t('title')}
          </h1>
          <Waves className="w-7 h-7 text-crt-green animate-pulse" />
        </div>
        <p className="text-crt-green/50 text-xs mb-2">
          {t('subtitle')}
        </p>
        {/* Language selector + Mode selector */}
        <div className="flex flex-col items-center gap-2">
          <LanguageSelector />
          <ModeSelector 
            mode={networkMode} 
            onModeChange={handleModeChange}
            disabled={isTraining}
          />
          {/* Indicador de modo activo */}
          <div className={`text-[10px] px-2 py-0.5 rounded ${
            networkMode === 'manual' 
              ? 'bg-crt-green/10 text-crt-green/60 border border-crt-green/30' 
              : 'bg-cyan-400/10 text-cyan-400/60 border border-cyan-400/30'
          }`}>
            {networkMode === 'manual' 
              ? '🧠 Backprop manual (educativo)' 
              : '⚡ TensorFlow.js (optimizado)'}
          </div>
        </div>
      </div>
      
      {/* Main content */}
      <div className="max-w-[1400px] mx-auto flex-1">
        {/* Fila superior: Osciloscopio + Panel de Control */}
        <div className="flex flex-col lg:flex-row gap-4 justify-center items-start mb-4">
          {/* Oscilloscope */}
          <div className="flex-shrink-0">
            <div className="text-xs text-crt-green/50 uppercase tracking-wider mb-1 text-center">
              📊 {t('functionApproximation')}
            </div>
            <OscilloscopeCanvas
              dataPoints={dataPoints}
              predictions={predictionPoints}
              trueSine={trueSinePoints}
              width={750}
              height={400}
            />
            
            {/* Legend */}
            <div className="flex justify-center gap-4 mt-2 text-[10px]">
              <div className="flex items-center gap-1">
                <div className="w-3 h-1 rounded" style={{ background: 'rgba(0, 255, 65, 0.8)', boxShadow: '0 0 5px rgba(0,255,65,0.8)' }} />
                <span className="text-crt-green/70">{t('dataWithNoise')}</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-1 rounded" style={{ background: 'linear-gradient(90deg, rgba(150,255,150,0.9), rgba(0,255,65,0.9))', boxShadow: '0 0 8px rgba(0,255,65,0.6)' }} />
                <span className="text-crt-green/70">{t('networkPrediction')}</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-1 rounded border border-dashed border-yellow-400/70" />
                <span className="text-crt-green/70">{t('realSine')}</span>
              </div>
            </div>
          </div>
          
          {/* Panel de controles - condicional según modo */}
          <div className="flex flex-col gap-3">
            {networkMode === 'manual' ? (
              <TrainingPanel
                epoch={epoch}
                loss={loss}
                isTraining={isTraining}
                speed={speed}
                config={config}
                onToggleTraining={() => !isTestMode && setIsTraining(!isTraining)}
                onReset={handleReset}
                onStop={handleStop}
                onSpeedChange={setSpeed}
                onConfigChange={handleConfigChange}
              />
            ) : (
              <TensorFlowPanel
                epoch={epoch}
                loss={loss}
                isTraining={isTraining}
                speed={speed}
                config={tfConfig}
                onToggleTraining={() => !isTestMode && setIsTraining(!isTraining)}
                onReset={handleReset}
                onStop={handleStop}
                onSpeedChange={setSpeed}
                onConfigChange={handleTFConfigChange}
                disabled={isTestMode}
              />
            )}
            
            {/* Panel de Dataset */}
            <DatasetPanel
              config={datasetConfig}
              onConfigChange={handleDatasetConfigChange}
              onImportCSV={handleImportCSV}
              disabled={isTraining}
              isTestMode={isTestMode}
              onToggleTestMode={handleToggleTestMode}
            />
            
            {/* Indicador de modo test */}
            {isTestMode && (
              <div className="terminal-panel p-3 bg-purple-900/20 border-purple-500/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <FlaskConical className="w-4 h-4 text-purple-400 animate-pulse" />
                    <span className="text-xs text-purple-400">Modo Test Activo</span>
                  </div>
                  <div className="text-[10px] text-purple-400/70">
                    Punto {testIndex + 1} / {testPoints.length}
                  </div>
                </div>
                <div className="text-[9px] text-purple-400/50 mt-1">
                  Visualizando predicciones en tiempo real sobre datos de prueba
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Fila inferior: Visualización de Red (ancho completo) */}
        <div className="flex justify-center">
          <NeuralNetworkViz
            networkState={networkState}
            getNeuronInfo={getNeuronInfo}
            getNeuronPattern={getNeuronPattern}
            width={1100}
            height={350}
            isTraining={isTraining}
            epoch={epoch}
            onToggleTraining={() => setIsTraining(!isTraining)}
            onStop={handleStop}
            onReset={handleReset}
          />
        </div>
      </div>
      
      {/* Footer */}
      <footer className="mt-8 pt-6 border-t border-crt-green/20">
        <div className="max-w-[1400px] mx-auto">
          {/* Info técnica */}
          <div className="text-center mb-4 text-[10px] text-crt-green/40 space-y-0.5">
            <p>{t('footerArchitecture')}: {currentArchitecture}</p>
            <p>
              Dataset: {datasetConfig.numPoints} puntos • f(x) = {functionDescriptions[datasetConfig.functionType].formula} • 
              Ruido: {(datasetConfig.noiseLevel * 100).toFixed(0)}% • x ∈ [{datasetConfig.xMin}, {datasetConfig.xMax}]
            </p>
            <p>
              {networkMode === 'manual' 
                ? t('footerImplementation')
                : 'TensorFlow.js • Optimized GPU/WebGL acceleration'}
            </p>
            {networkMode === 'tensorflow' && (
              <p className="text-cyan-400/50">
                Optimizer: {tfConfig.optimizer.toUpperCase()} • Loss: {tfConfig.loss} • LR: {tfConfig.learningRate}
              </p>
            )}
          </div>
          
          {/* Sección educativa */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 mb-4">
            <div className="flex items-center gap-2 text-crt-green/50">
              <GraduationCap className="w-5 h-5" />
              <span className="text-xs">{t('footerEducational')}</span>
            </div>
          </div>
          
          {/* Autor */}
          <div className="flex flex-col items-center justify-center gap-2 mb-4">
            <span className="text-sm text-crt-green/60 font-medium">Franco Joaquín Zini</span>
            <div className="flex items-center gap-4 text-xs text-crt-green/40">
              <a 
                href="https://linkedin.com/in/francojzini" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-crt-green/70 transition-colors"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                francojzini
              </a>
              <a 
                href="https://github.com/protos50" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-crt-green/70 transition-colors"
              >
                <Github className="w-4 h-4" />
                protos50
              </a>
            </div>
          </div>
          
          {/* Credits y año */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 text-[10px] text-crt-green/30">
            <span>{t('footerAuthor')} • {t('footerYear')}</span>
            <span className="hidden md:inline">|</span>
            <span className="flex items-center gap-1">
              <span>Neural Sine Learner v1.0</span>
            </span>
          </div>
          
          {/* Decoración CRT */}
          <div className="mt-6 flex justify-center">
            <div className="w-32 h-1 bg-gradient-to-r from-transparent via-crt-green/30 to-transparent rounded-full" />
          </div>
        </div>
      </footer>
    </main>
  );
}
