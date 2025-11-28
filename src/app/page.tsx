'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { NeuralNetwork, generateSineDataset, type NetworkState, type NetworkConfig, type NeuronInfo } from '@/lib/neural-network';
import OscilloscopeCanvas from '@/components/OscilloscopeCanvas';
import NeuralNetworkViz from '@/components/NeuralNetworkViz';
import TrainingPanel from '@/components/TrainingPanel';
import { useI18n, LanguageSelector } from '@/lib/i18n';
import { Brain, Waves, Github, GraduationCap } from 'lucide-react';

const DEFAULT_CONFIG: NetworkConfig = {
  hiddenLayers: [8, 8],
  hiddenActivation: 'tanh',
  outputActivation: 'linear',
  learningRate: 0.01,
  momentum: 0,
};

export default function Home() {
  const { t } = useI18n();
  
  // Red neuronal
  const networkRef = useRef<NeuralNetwork | null>(null);
  
  // Dataset
  const [dataset, setDataset] = useState<{
    X: number[];
    Y: number[];
    YTrue: number[];
  }>({ X: [], Y: [], YTrue: [] });
  
  // Estado del entrenamiento
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
  
  // Inicializar
  useEffect(() => {
    // Crear red neuronal
    networkRef.current = new NeuralNetwork(config);
    
    // Generar dataset
    const data = generateSineDataset(200, 0.2, -6, 6);
    setDataset(data);
    
    // Estado inicial SIN entrenar (epoch 0, activaciones en 0)
    if (networkRef.current) {
      // Predicciones iniciales (línea horizontal aleatoria)
      const preds = new Array(data.X.length).fill(0);
      setPredictions(preds);
      
      // Estado limpio sin forward pass
      const state = networkRef.current.getInitialState();
      setNetworkState(state);
      setEpoch(0);
      setLoss(1);
    }
  }, []);
  
  // Loop de entrenamiento
  useEffect(() => {
    if (!isTraining || !networkRef.current || dataset.X.length === 0) return;
    
    // Intervalo dinámico: más lento para velocidades bajas (slow motion)
    // speed < 1: intervalo más largo (hasta 500ms para speed=0.1)
    // speed >= 1: intervalo fijo de 50ms
    const intervalMs = speed < 1 
      ? Math.min(500, 50 / speed)  // e.g., speed=0.1 => 500ms, speed=0.5 => 100ms
      : 50;  // 20 FPS para velocidades normales
    
    const interval = setInterval(() => {
      if (!networkRef.current) return;
      
      // Para velocidades < 1, entrenar solo 1 época (ya hay intervalo más lento)
      // Para velocidades >= 1, entrenar speed épocas por tick
      const epochsThisTick = Math.max(1, Math.floor(speed));
      
      let currentLoss = 0;
      for (let i = 0; i < epochsThisTick; i++) {
        currentLoss = networkRef.current.trainEpoch(dataset.X, dataset.Y);
      }
      
      // Actualizar estado
      const preds = networkRef.current.predict(dataset.X);
      const state = networkRef.current.getState(dataset.X, dataset.Y);
      
      setPredictions(preds);
      setNetworkState(state);
      setEpoch(state.epoch);
      setLoss(currentLoss);
      
      // Auto-stop si el loss es muy bajo
      if (currentLoss < 0.0005) {
        setIsTraining(false);
      }
    }, intervalMs);
    
    return () => clearInterval(interval);
  }, [isTraining, speed, dataset]);
  
  // Manejar cambios de configuración
  const handleConfigChange = useCallback((newConfig: Partial<NetworkConfig>) => {
    setIsTraining(false);
    
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);
    
    // Si cambia la arquitectura o activaciones, recrear la red
    const structuralChange = 
      newConfig.hiddenLayers !== undefined ||
      newConfig.hiddenActivation !== undefined ||
      newConfig.outputActivation !== undefined;
    
    if (structuralChange) {
      // Nueva red con nueva configuración
      networkRef.current = new NeuralNetwork(updatedConfig);
      
      // Nuevo dataset
      const data = generateSineDataset(200, 0.2, -6, 6);
      setDataset(data);
      
      if (networkRef.current) {
        // Estado limpio sin forward pass
        const preds = new Array(data.X.length).fill(0);
        setPredictions(preds);
        
        const state = networkRef.current.getInitialState();
        setNetworkState(state);
        setEpoch(0);
        setLoss(1);
      }
    } else {
      // Solo actualizar learning rate o momentum
      if (networkRef.current) {
        if (newConfig.learningRate !== undefined) {
          networkRef.current.setLearningRate(newConfig.learningRate);
        }
        if (newConfig.momentum !== undefined) {
          networkRef.current.setMomentum(newConfig.momentum);
        }
      }
    }
  }, [config]);
  
  // Reset
  const handleReset = useCallback(() => {
    setIsTraining(false);
    
    // Nueva red
    networkRef.current = new NeuralNetwork(config);
    
    // Nuevo dataset
    const data = generateSineDataset(200, 0.2, -6, 6);
    setDataset(data);
    
    // Reset estado - limpio sin forward pass
    if (networkRef.current) {
      const preds = new Array(data.X.length).fill(0);
      setPredictions(preds);
      
      const state = networkRef.current.getInitialState();
      setNetworkState(state);
      setEpoch(0);
      setLoss(1);
    }
  }, [config]);
  
  // Stop - pausa sin resetear (mantiene datos y estado)
  const handleStop = useCallback(() => {
    setIsTraining(false);
  }, []);
  
  // Obtener info de neurona (para el hover)
  const getNeuronInfo = useCallback((layer: number, index: number): NeuronInfo => {
    if (networkRef.current) {
      return networkRef.current.getNeuronInfo(layer, index);
    }
    // Fallback
    return {
      layer,
      layerType: 'input',
      index,
      weights: [],
      bias: 0,
      activation: 'linear',
      currentInput: 0,
      currentOutput: 0,
      patternInfo: { type: 'receivesX' },
    };
  }, []);
  
  // Obtener patrón de neurona (mini gráfico)
  const getNeuronPattern = useCallback((layer: number, index: number): { x: number; y: number }[] => {
    if (networkRef.current) {
      return networkRef.current.getNeuronPattern(layer, index);
    }
    return [];
  }, []);
  
  // Convertir datos para el canvas
  const dataPoints = dataset.X.map((x, i) => ({ x, y: dataset.Y[i] }));
  const predictionPoints = dataset.X.map((x, i) => ({ x, y: predictions[i] || 0 }));
  const trueSinePoints = dataset.X.map((x, i) => ({ x, y: dataset.YTrue[i] }));

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
        {/* Language selector */}
        <div className="flex justify-center">
          <LanguageSelector />
        </div>
      </div>
      
      {/* Main content - nuevo layout */}
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
          
          {/* Training controls - más ancho */}
          <TrainingPanel
            epoch={epoch}
            loss={loss}
            isTraining={isTraining}
            speed={speed}
            config={config}
            onToggleTraining={() => setIsTraining(!isTraining)}
            onReset={handleReset}
            onStop={handleStop}
            onSpeedChange={setSpeed}
            onConfigChange={handleConfigChange}
          />
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
            <p>{t('footerArchitecture')}: {networkState.config?.hiddenLayers ? `1 → ${networkState.config.hiddenLayers.join(' → ')} → 1` : '1 → 8 → 8 → 1'}</p>
            <p>{t('footerDataset')}</p>
            <p>{t('footerImplementation')}</p>
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
