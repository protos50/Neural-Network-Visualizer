'use client';

import { useState, useEffect, useMemo } from 'react';
import { Table, Activity, FlaskConical, GraduationCap, ChevronLeft, ChevronRight, Play } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

interface CsvDataViewerProps {
  X: number[][];
  Y: number[][];
  predictions: number[];
  inputCols: string[];
  outputCols: string[];
  inputSize: number;
  outputSize: number;
  trainSplit: number;
  onTrainSplitChange: (split: number) => void;
  isTraining: boolean;
  epoch: number;
  loss: number;
  datasetName: string;
}

type ViewMode = 'train' | 'test';

export default function CsvDataViewer({
  X,
  Y,
  predictions,
  inputCols,
  outputCols,
  inputSize,
  outputSize,
  trainSplit,
  onTrainSplitChange,
  isTraining,
  epoch,
  loss,
  datasetName,
}: CsvDataViewerProps) {
  const { t } = useI18n();
  const [viewMode, setViewMode] = useState<ViewMode>('train');
  const [localIndex, setLocalIndex] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  
  // Calcular split
  const splitIndex = Math.floor((X.length * trainSplit) / 100);
  const testCount = X.length - splitIndex;
  const trainCount = splitIndex;
  
  // Datos según modo
  const viewData = useMemo(() => {
    if (viewMode === 'test') {
      return {
        X: X.slice(splitIndex),
        Y: Y.slice(splitIndex),
        predictions: predictions.slice(splitIndex),
        offset: splitIndex,
        count: testCount,
      };
    }
    // Modo train - solo datos de entrenamiento
    return {
      X: X.slice(0, splitIndex),
      Y: Y.slice(0, splitIndex),
      predictions: predictions.slice(0, splitIndex),
      offset: 0,
      count: trainCount,
    };
  }, [X, Y, predictions, splitIndex, testCount, trainCount, viewMode]);
  
  // Reset índice cuando cambia el modo
  useEffect(() => {
    setLocalIndex(0);
    setAutoPlay(false);
  }, [viewMode]);
  
  // Auto-play (funciona en ambos modos)
  useEffect(() => {
    // En modo train, solo auto-scroll si está entrenando
    if (viewMode === 'train' && !isTraining) {
      setAutoPlay(false);
      return;
    }
    
    if (!autoPlay) return;
    
    const interval = setInterval(() => {
      setLocalIndex(prev => (prev + 1) % viewData.count);
    }, isTraining ? 50 : 400); // Más rápido durante entrenamiento
    
    return () => clearInterval(interval);
  }, [autoPlay, viewData.count, isTraining, viewMode]);
  
  // En modo train, auto-activar cuando empieza entrenamiento
  useEffect(() => {
    if (viewMode === 'train') {
      setAutoPlay(isTraining);
    }
  }, [isTraining, viewMode]);
  
  // Calcular accuracy sobre test
  const testAccuracy = useMemo(() => {
    if (predictions.length === 0 || Y.length === 0) return 0;
    
    let correct = 0;
    
    if (outputSize > 1) {
      for (let i = splitIndex; i < Y.length; i++) {
        const base = i * outputSize;
        if (base + outputSize > predictions.length) continue;
        const predVec = predictions.slice(base, base + outputSize);
        const targetVec = Y[i];
        
        let predClass = 0;
        let predMax = predVec[0];
        for (let k = 1; k < outputSize; k++) {
          if (predVec[k] > predMax) {
            predMax = predVec[k];
            predClass = k;
          }
        }
        
        let actualClass = 0;
        let actualMax = targetVec[0];
        for (let k = 1; k < outputSize; k++) {
          if (targetVec[k] > actualMax) {
            actualMax = targetVec[k];
            actualClass = k;
          }
        }
        
        if (predClass === actualClass) correct++;
      }
    } else {
      for (let i = splitIndex; i < Y.length; i++) {
        if (i >= predictions.length) continue;
        const pred = predictions[i] > 0.5 ? 1 : 0;
        const actual = Y[i][0] > 0.5 ? 1 : 0;
        if (pred === actual) correct++;
      }
    }
    
    return testCount > 0 ? (correct / testCount) * 100 : 0;
  }, [predictions, Y, splitIndex, testCount, outputSize]);
  
  // Calcular accuracy sobre train
  const trainAccuracy = useMemo(() => {
    if (predictions.length === 0 || Y.length === 0) return 0;
    
    let correct = 0;
    
    if (outputSize > 1) {
      for (let i = 0; i < splitIndex; i++) {
        const base = i * outputSize;
        if (base + outputSize > predictions.length) continue;
        const predVec = predictions.slice(base, base + outputSize);
        const targetVec = Y[i];
        
        let predClass = 0;
        let predMax = predVec[0];
        for (let k = 1; k < outputSize; k++) {
          if (predVec[k] > predMax) {
            predMax = predVec[k];
            predClass = k;
          }
        }
        
        let actualClass = 0;
        let actualMax = targetVec[0];
        for (let k = 1; k < outputSize; k++) {
          if (targetVec[k] > actualMax) {
            actualMax = targetVec[k];
            actualClass = k;
          }
        }
        
        if (predClass === actualClass) correct++;
      }
    } else {
      for (let i = 0; i < splitIndex; i++) {
        if (i >= predictions.length) continue;
        const pred = predictions[i] > 0.5 ? 1 : 0;
        const actual = Y[i][0] > 0.5 ? 1 : 0;
        if (pred === actual) correct++;
      }
    }
    
    return trainCount > 0 ? (correct / trainCount) * 100 : 0;
  }, [predictions, Y, splitIndex, trainCount, outputSize]);
  
  // Datos de la fila actual
  const globalIndex = viewData.offset + localIndex;
  const currentRow = viewData.X[localIndex];
  const currentTarget = viewData.Y[localIndex];
  const isTrainRow = globalIndex < splitIndex;

  // Utilidad para nombres de clases
  const getClassLabel = (classIndex: number | undefined) => {
    if (classIndex === undefined || classIndex < 0) return '--';
    if (outputCols && outputCols[classIndex]) return outputCols[classIndex];
    return `Class ${classIndex}`;
  };

  // Predicción actual (binaria o multiclase)
  let currentPredScalar: number | undefined = undefined;
  let currentPredClass: number | undefined = undefined;
  let currentTargetClass: number | undefined = undefined;

  if (outputSize === 1) {
    // Caso binario clásico (Titanic, XOR, etc.)
    currentPredScalar = viewData.predictions[localIndex];
    if (currentTarget && currentTarget.length > 0) {
      currentTargetClass = currentTarget[0] > 0.5 ? 1 : 0;
    }
  } else {
    // Multiclase: usar argmax de vectores de longitud outputSize
    const base = globalIndex * outputSize;
    if (base + outputSize <= predictions.length) {
      const predVec = predictions.slice(base, base + outputSize);
      // Clase predicha
      let maxIdx = 0;
      let maxVal = predVec[0];
      for (let k = 1; k < outputSize; k++) {
        if (predVec[k] > maxVal) {
          maxVal = predVec[k];
          maxIdx = k;
        }
      }
      currentPredClass = maxIdx;
      currentPredScalar = maxVal;
    }
    // Clase real (argmax del target one-hot)
    if (currentTarget && currentTarget.length === outputSize) {
      let maxIdxT = 0;
      let maxValT = currentTarget[0];
      for (let k = 1; k < outputSize; k++) {
        if (currentTarget[k] > maxValT) {
          maxValT = currentTarget[k];
          maxIdxT = k;
        }
      }
      currentTargetClass = maxIdxT;
    }
  }
  
  // Determinar si la predicción actual coincide con el target
  let isMatch: boolean | undefined = undefined;
  if (outputSize === 1 && currentPredScalar !== undefined && currentTarget && currentTarget.length > 0) {
    const predBin = currentPredScalar > 0.5 ? 1 : 0;
    const targetBin = currentTarget[0] > 0.5 ? 1 : 0;
    isMatch = predBin === targetBin;
  } else if (outputSize > 1 && currentPredClass !== undefined && currentTargetClass !== undefined) {
    isMatch = currentPredClass === currentTargetClass;
  }

  return (
    <div className="bg-black/60 border border-cyan-400/30 rounded-lg p-3 space-y-3 w-[750px]">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-cyan-400 uppercase tracking-wider">
          <Table size={14} />
          <span>📊 {datasetName}</span>
        </div>
        <div className="text-[10px] text-cyan-400/50">
          {X.length} {t('rows')} • {inputSize} {t('inputs')} → {outputSize} {t('outputs')}
        </div>
      </div>
      
      {/* Train/Test Split Slider */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-green-400">🎓 {t('trainSplit')}: {trainSplit}% ({splitIndex} {t('rows')})</span>
          <span className="text-orange-400">🧪 {t('testSplit')}: {100 - trainSplit}% ({testCount} {t('rows')})</span>
        </div>
        <input
          type="range"
          min={50}
          max={95}
          value={trainSplit}
          onChange={(e) => onTrainSplitChange(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          disabled={isTraining}
        />
      </div>
      
      {/* Mode Toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setViewMode('train')}
          className={`flex-1 py-2 px-3 rounded text-xs font-mono flex items-center justify-center gap-2 transition-all ${
            viewMode === 'train'
              ? 'bg-green-400/20 border border-green-400/50 text-green-400'
              : 'bg-gray-600/20 border border-gray-500/30 text-gray-400 hover:bg-gray-600/30'
          }`}
        >
          <GraduationCap size={14} />
          {t('training')} ({trainCount})
        </button>
        <button
          onClick={() => setViewMode('test')}
          className={`flex-1 py-2 px-3 rounded text-xs font-mono flex items-center justify-center gap-2 transition-all ${
            viewMode === 'test'
              ? 'bg-orange-400/20 border border-orange-400/50 text-orange-400'
              : 'bg-gray-600/20 border border-gray-500/30 text-gray-400 hover:bg-gray-600/30'
          }`}
        >
          <FlaskConical size={14} />
          {t('test')} ({testCount})
        </button>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-4 gap-2 text-center">
        <div className="p-2 bg-blue-400/10 border border-blue-400/20 rounded">
          <div className="text-[10px] text-blue-400/60 uppercase">{t('epoch')}</div>
          <div className="text-lg font-mono text-blue-400">{epoch}</div>
        </div>
        <div className="p-2 bg-yellow-400/10 border border-yellow-400/20 rounded">
          <div className="text-[10px] text-yellow-400/60 uppercase">{t('lossTrain')}</div>
          <div className="text-lg font-mono text-yellow-400">{loss.toFixed(4)}</div>
        </div>
        <div className="p-2 bg-green-400/10 border border-green-400/20 rounded">
          <div className="text-[10px] text-green-400/60 uppercase">Acc Train</div>
          <div className="text-lg font-mono text-green-400">{trainAccuracy.toFixed(1)}%</div>
        </div>
        <div className="p-2 bg-orange-400/10 border border-orange-400/20 rounded">
          <div className="text-[10px] text-orange-400/60 uppercase">Acc Test</div>
          <div className="text-lg font-mono text-orange-400">{testAccuracy.toFixed(1)}%</div>
        </div>
      </div>
      
      {/* Current Row Visualization */}
      <div className="space-y-2 p-3 bg-black/40 rounded border border-cyan-400/10">
        <div className="flex items-center justify-between">
          <div className="text-[10px] text-cyan-400/60 uppercase flex items-center gap-2">
            <Activity size={12} className={isTraining ? 'animate-pulse' : ''} />
            <span>{t('row')} #{globalIndex + 1}</span>
            <span className={`px-1.5 py-0.5 rounded text-[8px] ${isTrainRow ? 'bg-green-400/20 text-green-400' : 'bg-orange-400/20 text-orange-400'}`}>
              {isTrainRow ? 'TRAIN' : 'TEST'}
            </span>
          </div>
          <button
            onClick={() => setAutoPlay(!autoPlay)}
            className={`text-[9px] px-2 py-1 rounded flex items-center gap-1 ${
              autoPlay 
                ? viewMode === 'train' ? 'bg-green-400/20 text-green-400' : 'bg-orange-400/20 text-orange-400'
                : 'bg-gray-600/20 text-gray-400'
            }`}
          >
            <Play size={10} />
            {autoPlay ? t('pauseAuto') : viewMode === 'train' ? t('autoTrain') : t('autoTest')}
          </button>
        </div>
        
        {/* Input → Output Flow */}
        <div className="flex items-stretch gap-2">
          {/* Inputs */}
          <div className="flex-1 p-2 bg-green-400/5 border border-green-400/20 rounded">
            <div className="text-[9px] text-green-400/60 uppercase mb-2">{t('inputsLabel')}</div>
            <div className="grid grid-cols-3 gap-1">
              {currentRow?.map((val, i) => (
                <div key={i} className="flex flex-col items-center p-1 bg-green-400/5 rounded">
                  <span className="text-[8px] text-green-400/40">{inputCols[i]}</span>
                  <span className="text-sm font-mono text-green-400">
                    {val.toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Arrow */}
          <div className="flex items-center text-cyan-400/50 text-2xl">→</div>
          
          {/* Prediction vs Target */}
          <div className="flex gap-2">
            {/* Prediction */}
            <div className="p-2 bg-cyan-400/5 border border-cyan-400/20 rounded min-w-[120px]">
              <div className="text-[9px] text-cyan-400/60 uppercase mb-1">{t('predictionLabel')}</div>
              <div className="flex flex-col items-center">
                <span className="text-2xl font-mono text-cyan-400">
                  {outputSize === 1
                    ? (currentPredScalar !== undefined ? currentPredScalar.toFixed(3) : '--')
                    : currentPredClass !== undefined
                      ? getClassLabel(currentPredClass)
                      : '--'}
                </span>
                {outputSize === 1 ? (
                  currentPredScalar !== undefined && (
                    <span className={`text-xs px-2 py-0.5 rounded mt-1 ${
                      currentPredScalar > 0.5 ? 'bg-green-400/20 text-green-400' : 'bg-red-400/20 text-red-400'
                    }`}>
                      {(outputCols?.[0] || 'y') + '=' + (currentPredScalar > 0.5 ? '1' : '0')}
                    </span>
                  )
                ) : (
                  currentPredScalar !== undefined && (
                    <span className="text-xs px-2 py-0.5 rounded mt-1 bg-cyan-400/10 text-cyan-300">
                      p = {currentPredScalar.toFixed(3)}
                    </span>
                  )
                )}
              </div>
            </div>
            
            {/* Target */}
            <div className="p-2 bg-orange-400/5 border border-orange-400/20 rounded min-w-[120px]">
              <div className="text-[9px] text-orange-400/60 uppercase mb-1">{t('realLabel')}</div>
              <div className="flex flex-col items-center">
                <span className="text-2xl font-mono text-orange-400">
                  {outputSize === 1
                    ? (currentTarget?.[0]?.toFixed(0) ?? '--')
                    : currentTargetClass !== undefined
                      ? getClassLabel(currentTargetClass)
                      : '--'}
                </span>
                {outputSize === 1 ? (
                  currentTarget && (
                    <span className={`text-xs px-2 py-0.5 rounded mt-1 ${
                      currentTarget[0] > 0.5 ? 'bg-green-400/20 text-green-400' : 'bg-red-400/20 text-red-400'
                    }`}>
                      {(outputCols?.[0] || 'y') + '=' + (currentTarget[0] > 0.5 ? '1' : '0')}
                    </span>
                  )
                ) : (
                  currentTargetClass !== undefined && (
                    <span className="text-xs px-2 py-0.5 rounded mt-1 bg-orange-400/10 text-orange-300">
                      {getClassLabel(currentTargetClass)}
                    </span>
                  )
                )}
              </div>
            </div>
          </div>
        </div>
        
        {/* Match indicator */}
        {isMatch !== undefined && (
          <div className={`text-center py-2 rounded text-sm font-mono ${
            isMatch
              ? 'bg-green-400/10 text-green-400 border border-green-400/30'
              : 'bg-red-400/10 text-red-400 border border-red-400/30'
          }`}>
            {isMatch ? `✓ ${t('correct')}` : `✗ ${t('incorrect')}`}
            {outputSize > 1 && currentPredClass !== undefined && currentTargetClass !== undefined && (
              <div className="mt-1 text-[10px] text-cyan-200/80">
                Pred: {getClassLabel(currentPredClass)} • Real: {getClassLabel(currentTargetClass)}
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Row navigator */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={() => setLocalIndex(prev => Math.max(0, prev - 1))}
          disabled={localIndex === 0}
          className="p-2 bg-cyan-400/10 border border-cyan-400/30 rounded text-cyan-400 hover:bg-cyan-400/20 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronLeft size={16} />
        </button>
        
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={1}
            max={viewData.count}
            value={localIndex + 1}
            onChange={(e) => setLocalIndex(Math.max(0, Math.min(viewData.count - 1, parseInt(e.target.value) - 1 || 0)))}
            className="w-16 bg-black/50 border border-cyan-400/30 rounded px-2 py-1 text-sm font-mono text-cyan-400 text-center"
          />
          <span className="text-sm text-cyan-400/50">/ {viewData.count}</span>
          <span className="text-[10px] text-cyan-400/30">
            ({viewMode === 'test' ? t('test') : t('training')})
          </span>
        </div>
        
        <button
          onClick={() => setLocalIndex(prev => Math.min(viewData.count - 1, prev + 1))}
          disabled={localIndex >= viewData.count - 1}
          className="p-2 bg-cyan-400/10 border border-cyan-400/30 rounded text-cyan-400 hover:bg-cyan-400/20 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}
