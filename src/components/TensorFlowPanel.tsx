'use client';

import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Square, Zap, ZapOff, Info, ChevronDown, ChevronUp, Plus, X, Cpu } from 'lucide-react';
import DatasetLoader from './DatasetLoader';
import type { TFModelConfig, TFActivation, OptimizerName, LossFn } from '@/lib/neural-network-tfjs';
import { 
  ALLOWED_ACTIVATIONS, 
  OPTIMIZERS, 
  LOSS_FUNCTIONS,
  activationDescriptions,
  optimizerDescriptions,
  lossDescriptions
} from '@/lib/neural-network-tfjs';
import { useI18n } from '@/lib/i18n';

interface LoadedDataset {
  X: number[][];
  Y: number[][];
  inputCols: string[];
  outputCols: string[];
  inputSize: number;
  outputSize: number;
  rows: number;
  name: string;
}

interface TensorFlowPanelProps {
  // Estado del entrenamiento
  epoch: number;
  loss: number;
  isTraining: boolean;
  speed: number;
  // Configuración TF
  config: TFModelConfig;
  // Callbacks
  onToggleTraining: () => void;
  onReset: () => void;
  onStop?: () => void;
  onSpeedChange: (speed: number) => void;
  onConfigChange: (config: Partial<TFModelConfig>) => void;
  onDatasetLoad?: (dataset: LoadedDataset) => void;
  disabled?: boolean;
}

// Tooltip component
function Tooltip({ children, content }: { children: React.ReactNode; content: React.ReactNode }) {
  const [show, setShow] = useState(false);
  
  return (
    <div className="relative inline-block">
      <div 
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="cursor-help"
      >
        {children}
      </div>
      {show && (
        <div className="absolute z-50 w-72 p-3 mt-1 text-xs bg-black/95 border border-cyan-400/50 rounded shadow-lg left-0 top-full">
          {content}
        </div>
      )}
    </div>
  );
}

export default function TensorFlowPanel({
  epoch,
  loss,
  isTraining,
  speed,
  config,
  onToggleTraining,
  onReset,
  onStop,
  onSpeedChange,
  onConfigChange,
  onDatasetLoad,
  disabled = false,
}: TensorFlowPanelProps) {
  const { t } = useI18n();
  const [showArchitecture, setShowArchitecture] = useState(false);

  // Arquitectura como string (usa inputSize del config)
  const architectureStr = `${config.inputSize || 1} → ${config.layers.map(l => l.units).join(' → ')}`;
  
  // Cambiar activación de una capa específica
  const handleLayerActivationChange = (layerIndex: number, activation: TFActivation) => {
    const newLayers = [...config.layers];
    newLayers[layerIndex] = { ...newLayers[layerIndex], activation };
    onConfigChange({ layers: newLayers });
  };

  // Cambiar unidades de una capa
  const handleLayerUnitsChange = (layerIndex: number, units: number) => {
    if (units < 1 || units > 128) return;
    const newLayers = [...config.layers];
    newLayers[layerIndex] = { ...newLayers[layerIndex], units };
    onConfigChange({ layers: newLayers });
  };

  // Agregar capa
  const handleAddLayer = () => {
    if (config.layers.length >= 6) return;
    const newLayers = [...config.layers];
    // Insertar antes de la última capa (output)
    newLayers.splice(newLayers.length - 1, 0, { units: 8, activation: 'swish' });
    onConfigChange({ layers: newLayers });
  };

  // Eliminar capa
  const handleRemoveLayer = (layerIndex: number) => {
    if (config.layers.length <= 2) return; // Mínimo: 1 hidden + 1 output
    const newLayers = config.layers.filter((_, i) => i !== layerIndex);
    onConfigChange({ layers: newLayers });
  };

  // Cambiar tamaño de entrada
  const handleInputSizeChange = (size: number) => {
    if (size < 1 || size > 32) return;
    onConfigChange({ inputSize: size });
  };

  // Cambiar tamaño de salida (actualiza última capa)
  const handleOutputSizeChange = (size: number) => {
    if (size < 1 || size > 32) return;
    const newLayers = [...config.layers];
    newLayers[newLayers.length - 1] = { 
      ...newLayers[newLayers.length - 1], 
      units: size 
    };
    onConfigChange({ outputSize: size, layers: newLayers });
  };

  return (
    <div className="terminal-panel p-4 space-y-4 border-cyan-400/30" style={{ width: 360, borderColor: 'rgba(34, 211, 238, 0.3)' }}>
      {/* Header - TensorFlow style */}
      <div className="text-center text-xs text-cyan-400/70 uppercase tracking-wider border-b border-cyan-400/20 pb-2 flex items-center justify-center gap-2">
        <Cpu className="w-4 h-4" />
        <span>TensorFlow.js Control</span>
      </div>
      
      {/* Stats principales */}
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-xs text-cyan-400/50 uppercase flex items-center justify-center gap-1">
            {t('epoch')}
            <Tooltip content={
              <div className="text-cyan-300">
                <strong>{t('epoch')}</strong>
                <br/><br/>
                {t('epochTooltip')}
              </div>
            }>
              <Info size={10} />
            </Tooltip>
          </div>
          <div className="text-2xl font-bold text-cyan-400 tabular-nums" style={{ textShadow: '0 0 10px rgba(34, 211, 238, 0.5)' }}>
            {epoch.toString().padStart(4, '0')}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-cyan-400/50 uppercase flex items-center justify-center gap-1">
            {t('loss')} ({config.loss.replace(/([A-Z])/g, ' $1').trim()})
            <Tooltip content={
              <div className="text-cyan-300">
                <strong>{config.loss}</strong>
                <br/><br/>
                {lossDescriptions[config.loss as LossFn] || ''}
              </div>
            }>
              <Info size={10} />
            </Tooltip>
          </div>
          <div className="text-2xl font-bold text-cyan-400 tabular-nums" style={{ textShadow: '0 0 10px rgba(34, 211, 238, 0.5)' }}>
            {loss.toFixed(4)}
          </div>
        </div>
      </div>
      
      {/* Loss bar visual */}
      <div className="space-y-1">
        <div className="text-xs text-cyan-400/50">{t('progress')}</div>
        <div className="h-2 bg-black border border-cyan-400/30 rounded overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 transition-all duration-300"
            style={{ 
              width: `${Math.max(0, Math.min(100, 100 - loss * 100))}%`,
              boxShadow: '0 0 10px rgba(34, 211, 238, 0.5)'
            }}
          />
        </div>
      </div>
      
      {/* Controls principales */}
      <div className="flex justify-center gap-2">
        <button
          onClick={onToggleTraining}
          disabled={disabled}
          className={`p-3 rounded border transition-all ${
            isTraining 
              ? 'border-yellow-500/50 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20' 
              : 'border-cyan-400/50 bg-cyan-400/10 text-cyan-400 hover:bg-cyan-400/20'
          } disabled:opacity-50`}
          title={isTraining ? t('pause') : t('start')}
        >
          {isTraining ? <Pause size={20} /> : <Play size={20} />}
        </button>
        
        {onStop && (
          <button
            onClick={onStop}
            disabled={epoch === 0}
            className="p-3 rounded border border-orange-500/50 bg-orange-500/10 text-orange-400 hover:bg-orange-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            title={t('stopTooltip')}
          >
            <Square size={20} />
          </button>
        )}
        
        <button
          onClick={onReset}
          disabled={disabled && isTraining}
          className="p-3 rounded border border-red-500/50 bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-all disabled:opacity-50"
          title={t('resetTooltip')}
        >
          <RotateCcw size={20} />
        </button>
      </div>
      
      {/* ===== HIPERPARÁMETROS TF.js ===== */}
      <div className="border-t border-cyan-400/20 pt-3 space-y-3">
        <div className="text-xs text-cyan-400/70 uppercase tracking-wider text-center">
          ⚙️ {t('hyperparameters')}
        </div>
        
        {/* Speed control - escala logarítmica para mejor control */}
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-xs text-cyan-400/50 uppercase">{t('speed')}</span>
            <span className="text-xs text-cyan-400 font-mono">
              {speed < 1 ? speed.toFixed(2) : speed < 10 ? speed.toFixed(1) : Math.round(speed)} {t('epochsPerTick')}
            </span>
          </div>
          {/* Slider con escala logarítmica: 0-100 mapea a 0.1-1000 */}
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={Math.log10(speed * 10) * 25} // Convertir speed a posición del slider
            onChange={(e) => {
              // Convertir posición del slider a speed con escala logarítmica
              const pos = parseFloat(e.target.value);
              const newSpeed = Math.pow(10, pos / 25) / 10;
              onSpeedChange(Math.max(0.1, Math.min(1000, newSpeed)));
            }}
            className="w-full accent-cyan-400"
          />
          <div className="flex justify-between text-[9px] text-cyan-400/30">
            <span>0.1</span>
            <span>1</span>
            <span>10</span>
            <span>100</span>
            <span>1000</span>
          </div>
          <div className="text-[9px] text-cyan-400/40 text-center">
            {speed >= 100 ? '⚡ Turbo' : speed >= 10 ? '🚀 Rápido' : speed >= 1 ? '▶️ Normal' : '🐢 Lento'}
          </div>
          
          {/* Turbo mode toggle - solo actualiza UI cada N épocas */}
          {speed >= 50 && (
            <div className="flex items-center justify-center gap-2 mt-1 p-1 bg-yellow-500/10 border border-yellow-500/30 rounded">
              <Zap size={12} className="text-yellow-400" />
              <span className="text-[9px] text-yellow-400">
                Modo Turbo: UI cada {Math.round(speed/10)*10} épocas
              </span>
            </div>
          )}
        </div>
        
        {/* Learning rate */}
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-xs text-cyan-400/50 uppercase flex items-center gap-1">
              {t('learningRate')} (α)
              <Tooltip content={
                <div className="text-cyan-300">
                  <strong>{t('learningRate')}</strong>
                  <br/><br/>
                  {t('learningRateTooltip')}
                </div>
              }>
                <Info size={10} />
              </Tooltip>
            </span>
            <span className="text-xs text-cyan-400 font-mono">{config.learningRate.toFixed(4)}</span>
          </div>
          <input
            type="range"
            min="1"
            max="100"
            value={config.learningRate * 1000}
            onChange={(e) => onConfigChange({ learningRate: parseInt(e.target.value) / 1000 })}
            className="w-full accent-cyan-400"
          />
          <div className="flex justify-between text-[10px] text-cyan-400/30">
            <span>0.001</span>
            <span>0.1</span>
          </div>
        </div>
        
        {/* Optimizer */}
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-xs text-cyan-400/50 uppercase flex items-center gap-1">
              Optimizer
              <Tooltip content={
                <div className="text-cyan-300">
                  <strong>{config.optimizer.toUpperCase()}</strong>
                  <br/><br/>
                  {optimizerDescriptions[config.optimizer as OptimizerName] || ''}
                </div>
              }>
                <Info size={10} />
              </Tooltip>
            </span>
          </div>
          <select
            value={config.optimizer}
            onChange={(e) => onConfigChange({ optimizer: e.target.value as OptimizerName })}
            className="w-full bg-black/50 border border-cyan-400/30 rounded px-2 py-1.5 text-xs font-mono text-cyan-400 focus:border-cyan-400 focus:outline-none"
          >
            {OPTIMIZERS.map(opt => (
              <option key={opt} value={opt}>{opt.toUpperCase()}</option>
            ))}
          </select>
          <div className="text-[10px] text-cyan-400/40">
            {(optimizerDescriptions[config.optimizer as OptimizerName] || '').slice(0, 60)}
          </div>
        </div>
        
        {/* Loss function */}
        <div className="space-y-1">
          <span className="text-xs text-cyan-400/50 uppercase flex items-center gap-1">
            Loss Function
            <Tooltip content={
              <div className="text-cyan-300">
                <strong>{config.loss}</strong>
                <br/><br/>
                {lossDescriptions[config.loss as LossFn] || ''}
              </div>
            }>
              <Info size={10} />
            </Tooltip>
          </span>
          <select
            value={config.loss}
            onChange={(e) => onConfigChange({ loss: e.target.value as LossFn })}
            className="w-full bg-black/50 border border-cyan-400/30 rounded px-2 py-1.5 text-xs font-mono text-cyan-400 focus:border-cyan-400 focus:outline-none"
          >
            {LOSS_FUNCTIONS.map(lf => (
              <option key={lf} value={lf}>{lf}</option>
            ))}
          </select>
        </div>
      </div>
      
      {/* ===== ARQUITECTURA (expandible) ===== */}
      <div className="border-t border-cyan-400/20 pt-3">
        <button
          onClick={() => setShowArchitecture(!showArchitecture)}
          className="w-full flex items-center justify-between text-xs text-cyan-400/70 uppercase tracking-wider hover:text-cyan-400 transition-colors"
        >
          <span>🧠 {t('architecture')}</span>
          {showArchitecture ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        
        {/* Arquitectura actual siempre visible */}
        <div className="text-center p-2 mt-2 bg-cyan-400/5 border border-cyan-400/20 rounded">
          <div className="text-[10px] text-cyan-400/50 mb-1">{t('currentArchitecture')}</div>
          <div className="text-sm font-mono text-cyan-400">{architectureStr}</div>
        </div>
        
        {showArchitecture && (
          <div className="mt-3 space-y-3">
            {/* Input layer (configurable) */}
            <div className="p-2 bg-green-400/5 border border-green-400/20 rounded">
              <div className="text-[10px] text-green-400/50 uppercase mb-2">Input Layer</div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="text-[9px] text-green-400/40 mb-1">Neuronas entrada</div>
                  <input
                    type="number"
                    min={1}
                    max={32}
                    value={config.inputSize || 1}
                    onChange={(e) => handleInputSizeChange(parseInt(e.target.value) || 1)}
                    className="w-full bg-black/50 border border-green-400/30 rounded px-2 py-1 text-xs font-mono text-green-400 focus:border-green-400 focus:outline-none"
                    disabled={disabled}
                  />
                </div>
                <div className="flex items-end">
                  <div className="text-[9px] text-green-400/40 pb-1.5">
                    {(config.inputSize || 1) === 1 ? '(x)' : `(x₁...x${config.inputSize})`}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Capas ocultas configurables */}
            {config.layers.slice(0, -1).map((layer, idx) => (
              <div key={idx} className="p-2 bg-cyan-400/5 border border-cyan-400/20 rounded">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-[10px] text-cyan-400/50 uppercase">
                    Hidden Layer {idx + 1}
                  </div>
                  {config.layers.length > 2 && (
                    <button
                      onClick={() => handleRemoveLayer(idx)}
                      className="text-red-400/70 hover:text-red-400 transition-colors"
                      disabled={disabled}
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  {/* Neuronas */}
                  <div>
                    <div className="text-[9px] text-cyan-400/40 mb-1">Neuronas</div>
                    <input
                      type="number"
                      min={1}
                      max={128}
                      value={layer.units}
                      onChange={(e) => handleLayerUnitsChange(idx, parseInt(e.target.value) || 1)}
                      className="w-full bg-black/50 border border-cyan-400/30 rounded px-2 py-1 text-xs font-mono text-cyan-400 focus:border-cyan-400 focus:outline-none"
                      disabled={disabled}
                    />
                  </div>
                  
                  {/* Activación */}
                  <div>
                    <div className="text-[9px] text-cyan-400/40 mb-1">Activación</div>
                    <select
                      value={layer.activation}
                      onChange={(e) => handleLayerActivationChange(idx, e.target.value as TFActivation)}
                      className="w-full bg-black/50 border border-cyan-400/30 rounded px-2 py-1 text-xs font-mono text-cyan-400 focus:border-cyan-400 focus:outline-none"
                      disabled={disabled}
                    >
                      {ALLOWED_ACTIVATIONS.map(act => (
                        <option key={act} value={act}>{act}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            ))}
            
            {/* Output layer (configurable) */}
            <div className="p-2 bg-orange-400/5 border border-orange-400/20 rounded">
              <div className="text-[10px] text-orange-400/50 uppercase mb-2">Output Layer</div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="text-[9px] text-orange-400/40 mb-1">Neuronas salida</div>
                  <input
                    type="number"
                    min={1}
                    max={32}
                    value={config.outputSize || config.layers[config.layers.length - 1].units}
                    onChange={(e) => handleOutputSizeChange(parseInt(e.target.value) || 1)}
                    className="w-full bg-black/50 border border-orange-400/30 rounded px-2 py-1 text-xs font-mono text-orange-400 focus:border-orange-400 focus:outline-none"
                    disabled={disabled}
                  />
                </div>
                
                {/* Activación output */}
                <div>
                  <div className="text-[9px] text-orange-400/40 mb-1">Activación</div>
                  <select
                    value={config.layers[config.layers.length - 1].activation}
                    onChange={(e) => handleLayerActivationChange(config.layers.length - 1, e.target.value as TFActivation)}
                    className="w-full bg-black/50 border border-orange-400/30 rounded px-2 py-1 text-xs font-mono text-orange-400 focus:border-orange-400 focus:outline-none"
                    disabled={disabled}
                  >
                    {ALLOWED_ACTIVATIONS.map(act => (
                      <option key={act} value={act}>{act}</option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="text-[9px] text-orange-400/40 mt-1">
                {(config.outputSize || 1) === 1 ? '(ŷ)' : `(ŷ₁...ŷ${config.outputSize})`}
              </div>
            </div>
            
            {/* Botón agregar capa */}
            {config.layers.length < 6 && (
              <button
                onClick={handleAddLayer}
                className="w-full p-2 border border-dashed border-cyan-400/30 rounded text-xs text-cyan-400/50 hover:text-cyan-400 hover:border-cyan-400/50 transition-all flex items-center justify-center gap-1"
                disabled={disabled}
              >
                <Plus size={14} />
                Agregar capa oculta
              </button>
            )}
          </div>
        )}
      </div>
      
      {/* Dataset Loader */}
      {onDatasetLoad && (
        <div className="pt-3 border-t border-cyan-400/20">
          <DatasetLoader 
            onDatasetLoad={onDatasetLoad}
            disabled={disabled || isTraining}
          />
        </div>
      )}
      
      {/* Status indicator */}
      <div className="flex items-center justify-center gap-2 pt-2 border-t border-cyan-400/20">
        {isTraining ? (
          <>
            <Zap size={14} className="text-cyan-400 animate-pulse" />
            <span className="text-xs text-cyan-400 animate-pulse uppercase">{t('statusTraining')}</span>
          </>
        ) : (
          <>
            <ZapOff size={14} className="text-cyan-400/50" />
            <span className="text-xs text-cyan-400/50 uppercase">{t('statusPaused')}</span>
          </>
        )}
      </div>
    </div>
  );
}
