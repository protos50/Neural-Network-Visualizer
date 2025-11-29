'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type Language = 'es' | 'en';

// Traducciones
const translations = {
  es: {
    // Header
    title: 'NEURAL SINE LEARNER',
    subtitle: 'Visualizador de Red Neuronal Aprendiendo la Función Seno en Tiempo Real',
    
    // Oscilloscope
    functionApproximation: 'Aproximación de Función',
    dataWithNoise: 'Datos con ruido',
    networkPrediction: 'Predicción de la red',
    realSine: 'sin(x) real',
    
    // Training panel
    trainingControl: 'Control de Entrenamiento',
    epoch: 'Época',
    loss: 'Pérdida (MSE)',
    start: 'Iniciar',
    train: 'Entrenar',
    pause: 'Pausar',
    stop: 'Detener',
    stopTooltip: 'Detiene el entrenamiento pero mantiene los datos y el estado actual. Útil para cambiar parámetros sin regenerar datos.',
    reset: 'Reiniciar',
    resetTooltip: 'Reinicia todo: pesos, datos y época a cero.',
    speed: 'Velocidad',
    speedTooltip: 'Cantidad de épocas a ejecutar por cada actualización de la visualización.',
    epochsPerTick: 'épocas/tick',
    
    // Status
    statusTraining: 'Entrenando...',
    statusPaused: 'Pausado',
    statusReady: 'Listo para entrenar',
    
    // Architecture changes
    pendingChanges: 'cambios pendientes',
    apply: 'Aplicar',
    architectureWarning: 'Cambiar la arquitectura reiniciará los pesos de la red.',
    hiddenLayersHelp: 'Ej: 16,8 = una capa de 16 y otra de 8 neuronas. Presiona Enter o haz clic en Aplicar.',
    
    // Activation
    activation: 'Función de Activación',
    activationTooltip: 'Función no lineal aplicada a cada neurona oculta.',
    
    // Data points
    dataPoints: 'Puntos de Datos',
    dataPointsTooltip: 'Cantidad de puntos de entrenamiento generados.',
    
    // Hyperparameters
    hyperparameters: 'Hiperparámetros',
    learningRate: 'Tasa de Aprendizaje',
    learningRateTooltip: 'Controla qué tan grandes son los pasos de ajuste. Valores altos = aprendizaje rápido pero inestable.',
    momentum: 'Momentum',
    momentumTooltip: 'Acumula velocidad en dirección consistente. Ayuda a escapar mínimos locales.',
    
    // Architecture
    architecture: 'Arquitectura',
    hiddenLayers: 'Capas Ocultas',
    hiddenLayersTooltip: 'Neuronas en cada capa oculta separadas por coma. Ejemplo: 8,8 = dos capas de 8 neuronas.',
    hiddenActivation: 'Activación Oculta',
    hiddenActivationTooltip: 'Función de activación para las capas ocultas.',
    outputActivation: 'Activación Salida',
    outputActivationTooltip: 'Función de activación para la capa de salida. Linear es común para regresión.',
    
    // Network visualization
    networkArchitecture: 'Arquitectura de la Red Neuronal',
    hoverHint: 'Pasa el mouse sobre una neurona para ver su patrón y estado detallado',
    lowActivation: 'Baja activación',
    highActivation: 'Alta activación',
    positiveWeight: 'Peso +',
    negativeWeight: 'Peso -',
    
    // Neuron tooltip
    neuron: 'Neurona',
    inputLayer: 'Entrada',
    hiddenLayer: 'Oculta',
    outputLayer: 'Salida',
    inputIdentity: 'Entrada (identidad)',
    finalOutput: 'Salida final de la red',
    neuronResponse: 'Respuesta de esta neurona',
    normalizedInput: 'Valor x de entrada normalizado',
    finalPrediction: 'Predicción final: combinación de todas las neuronas ocultas',
    neuronCurve: 'Curva: {activation}(w×x + b) — "qué forma detecta esta neurona"',
    weightToOutput: 'Peso hacia salida',
    addsToOutput: 'suma esta forma a la predicción',
    subtractsFromOutput: 'resta esta forma a la predicción',
    activationFunction: 'Función de Activación',
    range: 'Rango',
    input: 'Entrada',
    output: 'Salida',
    weights: 'Pesos',
    connections: 'conexiones',
    bias: 'Bias',
    interpretation: 'Interpretación',
    formula: 'Fórmula',
    
    // Footer
    footerArchitecture: 'Arquitectura',
    footerDataset: 'Dataset: 200 puntos | sin(x) + ruido(σ=0.2) | x ∈ [-6, 6]',
    footerImplementation: 'Backpropagation implementado desde cero en JavaScript/TypeScript',
    footerEducational: 'Proyecto educativo para visualizar el funcionamiento interno de redes neuronales',
    footerAuthor: 'Desarrollado para IA - UNNE',
    footerYear: '2025',
    
    // Legend labels
    legend: 'Leyenda',
    prediction: 'Predicción (red)',
    realSineLegend: 'sin(x) real',
    dataWithNoiseLegend: 'Datos (con ruido)',
    
    // TrainingPanel additional labels
    progress: 'Progreso (100% - Loss%)',
    slow: 'Lento',
    fast: 'Rápido',
    noInertia: '0 (sin inercia)',
    highInertia: '0.99 (mucha inercia)',
    currentArchitecture: 'Arquitectura actual',
    trainingStatus: 'Estado',
    epochTooltip: 'Una pasada completa por todos los datos de entrenamiento. Cada época, la red ve los 200 puntos y ajusta sus pesos.',
    lossTooltip: 'Mide qué tan lejos están las predicciones de los valores reales. Cuanto menor, mejor.',
    
    // Neuron pattern descriptions
    patternInputSensitivity: 'Sensibilidad al input',
    patternCombinesHidden: 'Combina todas las neuronas ocultas',
    patternMaxWeight: 'Mayor peso de neurona {neuronNum}',
    patternReceivesX: 'Recibe el valor x ∈ [-6, 6]',
    
    // Forward pass visualization
    forwardPass: 'Forward',
    forwardPassTooltip: 'Visualiza el flujo de la señal a través de la red durante una iteración',
    
    // App modes
    appModeRegression: 'Regresión',
    appModeClassification: 'Clasificación',
    appModeRegressionDesc: '🧠 Backprop manual (educativo)',
    appModeClassificationDesc: '📊 Clasificación con TensorFlow.js',
    
    // CSV Dataset
    loadDatasetCsv: 'Cargar Dataset CSV',
    selectDataset: 'Selecciona un dataset',
    loadDataset: 'Cargar Dataset',
    datasetLoaded: 'Dataset cargado',
    loadingDataset: 'Cargando...',
    inputs: 'Entradas',
    outputs: 'Salidas',
    rows: 'Filas',
    noDatasetLoaded: 'Carga un dataset CSV desde el panel derecho',
    datasetFolder: 'Carpeta: /public/datasets/',
    
    // CSV Data Viewer
    training: 'Entrenamiento',
    test: 'Test',
    trainSplit: 'Train',
    testSplit: 'Test',
    trainRows: 'filas train',
    testRows: 'filas test',
    viewAll: 'Ver Todo',
    viewTest: 'Solo Test',
    autoTrain: 'Auto Train',
    autoTest: 'Auto Test',
    pauseAuto: 'Pausar',
    row: 'Fila',
    inputsLabel: 'Entradas',
    predictionLabel: 'Predicción',
    realLabel: 'Real',
    correct: 'CORRECTO',
    incorrect: 'INCORRECTO',
    survives: 'Sobrevive',
    doesNotSurvive: 'No sobrevive',
    accuracy: 'Accuracy',
    accuracyTest: 'Accuracy (test)',
    lossTrain: 'Loss (train)',
    
    // Feature importance
    featureImportance: 'Importancia de Features',
    noImportanceData: 'Sin datos de importancia',
  },
  en: {
    // Header
    title: 'NEURAL SINE LEARNER',
    subtitle: 'Neural Network Visualizer Learning the Sine Function in Real Time',
    
    // Oscilloscope
    functionApproximation: 'Function Approximation',
    dataWithNoise: 'Data with noise',
    networkPrediction: 'Network prediction',
    realSine: 'Real sin(x)',
    
    // Training panel
    trainingControl: 'Training Control',
    epoch: 'Epoch',
    loss: 'Loss (MSE)',
    start: 'Start',
    train: 'Train',
    pause: 'Pause',
    stop: 'Stop',
    stopTooltip: 'Stops training but keeps data and current state. Useful for changing parameters without regenerating data.',
    reset: 'Reset',
    resetTooltip: 'Resets everything: weights, data and epoch to zero.',
    speed: 'Speed',
    speedTooltip: 'Number of epochs to run per visualization update.',
    epochsPerTick: 'epochs/tick',
    
    // Status
    statusTraining: 'Training...',
    statusPaused: 'Paused',
    statusReady: 'Ready to train',
    
    // Architecture changes
    pendingChanges: 'pending changes',
    apply: 'Apply',
    architectureWarning: 'Changing architecture will reset network weights.',
    hiddenLayersHelp: 'E.g.: 16,8 = one layer of 16 and another of 8 neurons. Press Enter or click Apply.',
    
    // Activation
    activation: 'Activation Function',
    activationTooltip: 'Non-linear function applied to each hidden neuron.',
    
    // Data points
    dataPoints: 'Data Points',
    dataPointsTooltip: 'Number of training points generated.',
    
    // Hyperparameters
    hyperparameters: 'Hyperparameters',
    learningRate: 'Learning Rate',
    learningRateTooltip: 'Controls how large adjustment steps are. High values = fast but unstable learning.',
    momentum: 'Momentum',
    momentumTooltip: 'Accumulates velocity in consistent direction. Helps escape local minima.',
    
    // Architecture
    architecture: 'Architecture',
    hiddenLayers: 'Hidden Layers',
    hiddenLayersTooltip: 'Neurons in each hidden layer separated by comma. Example: 8,8 = two layers of 8 neurons.',
    hiddenActivation: 'Hidden Activation',
    hiddenActivationTooltip: 'Activation function for hidden layers.',
    outputActivation: 'Output Activation',
    outputActivationTooltip: 'Activation function for output layer. Linear is common for regression.',
    
    // Network visualization
    networkArchitecture: 'Neural Network Architecture',
    hoverHint: 'Hover over a neuron to see its pattern and detailed state',
    lowActivation: 'Low activation',
    highActivation: 'High activation',
    positiveWeight: 'Weight +',
    negativeWeight: 'Weight -',
    
    // Neuron tooltip
    neuron: 'Neuron',
    inputLayer: 'Input',
    hiddenLayer: 'Hidden',
    outputLayer: 'Output',
    inputIdentity: 'Input (identity)',
    finalOutput: 'Final network output',
    neuronResponse: 'This neuron\'s response',
    normalizedInput: 'Normalized input x value',
    finalPrediction: 'Final prediction: combination of all hidden neurons',
    neuronCurve: 'Curve: {activation}(w×x + b) — "what shape this neuron detects"',
    weightToOutput: 'Weight to output',
    addsToOutput: 'adds this shape to prediction',
    subtractsFromOutput: 'subtracts this shape from prediction',
    activationFunction: 'Activation Function',
    range: 'Range',
    input: 'Input',
    output: 'Output',
    weights: 'Weights',
    connections: 'connections',
    bias: 'Bias',
    interpretation: 'Interpretation',
    formula: 'Formula',
    
    // Footer
    footerArchitecture: 'Architecture',
    footerDataset: 'Dataset: 200 points | sin(x) + noise(σ=0.2) | x ∈ [-6, 6]',
    footerImplementation: 'Backpropagation implemented from scratch in JavaScript/TypeScript',
    footerEducational: 'Educational project to visualize the inner workings of neural networks',
    footerAuthor: 'Developed for AI - UNNE',
    footerYear: '2025',
    
    // Legend labels
    legend: 'Legend',
    prediction: 'Prediction (network)',
    realSineLegend: 'Real sin(x)',
    dataWithNoiseLegend: 'Data (with noise)',
    
    // TrainingPanel additional labels
    progress: 'Progress (100% - Loss%)',
    slow: 'Slow',
    fast: 'Fast',
    noInertia: '0 (no inertia)',
    highInertia: '0.99 (high inertia)',
    currentArchitecture: 'Current architecture',
    trainingStatus: 'Status',
    epochTooltip: 'One complete pass through all training data. Each epoch, the network sees all 200 points and adjusts its weights.',
    lossTooltip: 'Measures how far predictions are from real values. Lower is better.',
    
    // Neuron pattern descriptions
    patternInputSensitivity: 'Input sensitivity',
    patternCombinesHidden: 'Combines all hidden neurons',
    patternMaxWeight: 'Max weight from neuron {neuronNum}',
    patternReceivesX: 'Receives value x ∈ [-6, 6]',
    
    // Forward pass visualization
    forwardPass: 'Forward',
    forwardPassTooltip: 'Visualizes the signal flow through the network during an iteration',
    
    // App modes
    appModeRegression: 'Regression',
    appModeClassification: 'Classification',
    appModeRegressionDesc: '🧠 Manual backprop (educational)',
    appModeClassificationDesc: '📊 Classification with TensorFlow.js',
    
    // CSV Dataset
    loadDatasetCsv: 'Load CSV Dataset',
    selectDataset: 'Select a dataset',
    loadDataset: 'Load Dataset',
    datasetLoaded: 'Dataset loaded',
    loadingDataset: 'Loading...',
    inputs: 'Inputs',
    outputs: 'Outputs',
    rows: 'Rows',
    noDatasetLoaded: 'Load a CSV dataset from the right panel',
    datasetFolder: 'Folder: /public/datasets/',
    
    // CSV Data Viewer
    training: 'Training',
    test: 'Test',
    trainSplit: 'Train',
    testSplit: 'Test',
    trainRows: 'train rows',
    testRows: 'test rows',
    viewAll: 'View All',
    viewTest: 'Test Only',
    autoTrain: 'Auto Train',
    autoTest: 'Auto Test',
    pauseAuto: 'Pause',
    row: 'Row',
    inputsLabel: 'Inputs',
    predictionLabel: 'Prediction',
    realLabel: 'Real',
    correct: 'CORRECT',
    incorrect: 'INCORRECT',
    survives: 'Survives',
    doesNotSurvive: 'Does not survive',
    accuracy: 'Accuracy',
    accuracyTest: 'Accuracy (test)',
    lossTrain: 'Loss (train)',
    
    // Feature importance
    featureImportance: 'Feature Importance',
    noImportanceData: 'No importance data',
  },
};

type TranslationKey = keyof typeof translations.es;

interface I18nContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: TranslationKey, replacements?: Record<string, string>) => string;
}

const I18nContext = createContext<I18nContextType | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>('es');
  
  const t = useCallback((key: TranslationKey, replacements?: Record<string, string>): string => {
    let text = translations[language][key] || translations.es[key] || key;
    
    if (replacements) {
      Object.entries(replacements).forEach(([k, v]) => {
        text = text.replace(`{${k}}`, v);
      });
    }
    
    return text;
  }, [language]);
  
  return (
    <I18nContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}

// Language selector component
export function LanguageSelector() {
  const { language, setLanguage } = useI18n();
  
  return (
    <div className="flex items-center gap-1 text-xs">
      <button
        onClick={() => setLanguage('es')}
        className={`px-2 py-0.5 rounded transition-all ${
          language === 'es' 
            ? 'bg-crt-green/30 text-crt-green border border-crt-green/50' 
            : 'text-crt-green/50 hover:text-crt-green/80 border border-transparent'
        }`}
      >
        ES
      </button>
      <span className="text-crt-green/30">|</span>
      <button
        onClick={() => setLanguage('en')}
        className={`px-2 py-0.5 rounded transition-all ${
          language === 'en' 
            ? 'bg-crt-green/30 text-crt-green border border-crt-green/50' 
            : 'text-crt-green/50 hover:text-crt-green/80 border border-transparent'
        }`}
      >
        EN
      </button>
    </div>
  );
}
