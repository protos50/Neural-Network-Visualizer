'use client';

import { Play, Pause, RotateCcw, Square, Zap, ZapOff, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { useState, useEffect } from 'react';
import type { ActivationFunction, NetworkConfig } from '@/lib/neural-network';
import { activationDescriptions } from '@/lib/neural-network';
import { useI18n } from '@/lib/i18n';

interface TrainingPanelProps {
  epoch: number;
  loss: number;
  isTraining: boolean;
  speed: number;
  config: NetworkConfig;
  onToggleTraining: () => void;
  onReset: () => void;
  onStop?: () => void;
  onSpeedChange: (speed: number) => void;
  onConfigChange: (config: Partial<NetworkConfig>) => void;
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
        <div className="absolute z-50 w-64 p-2 mt-1 text-xs bg-black/95 border border-crt-green/50 rounded shadow-lg -left-24">
          {content}
        </div>
      )}
    </div>
  );
}

export default function TrainingPanel({
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
}: TrainingPanelProps) {
  const { t } = useI18n();
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Local state for hidden layers input
  const [hiddenLayersInput, setHiddenLayersInput] = useState(config.hiddenLayers.join(','));
  const [pendingChange, setPendingChange] = useState(false);
  
  // Sync when config changes externally
  useEffect(() => {
    setHiddenLayersInput(config.hiddenLayers.join(','));
    setPendingChange(false);
  }, [config.hiddenLayers]);
  
  const handleHiddenLayersChange = (value: string) => {
    setHiddenLayersInput(value);
    const parsed = value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
    setPendingChange(parsed.length > 0 && parsed.join(',') !== config.hiddenLayers.join(','));
  };
  
  const applyHiddenLayers = () => {
    const layers = hiddenLayersInput.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
    if (layers.length > 0) {
      onConfigChange({ hiddenLayers: layers });
      setPendingChange(false);
    }
  };
  
  // Arquitectura como string
  const architectureStr = `1 → ${config.hiddenLayers.join(' → ')} → 1`;
  
  return (
    <div className="terminal-panel p-4 space-y-4" style={{ width: 340 }}>
      {/* Header */}
      <div className="text-center text-xs text-crt-green/70 uppercase tracking-wider border-b border-crt-green/20 pb-2">
        🎮 {t('trainingControl')}
      </div>
      
      {/* Stats principales */}
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-xs text-crt-green/50 uppercase flex items-center justify-center gap-1">
            {t('epoch')}
            <Tooltip content={
              <div>
                <strong>{t('epoch')}</strong>
                <br/><br/>
                {t('epochTooltip')}
              </div>
            }>
              <Info size={10} />
            </Tooltip>
          </div>
          <div className="text-2xl font-bold glow-text tabular-nums">
            {epoch.toString().padStart(4, '0')}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-crt-green/50 uppercase flex items-center justify-center gap-1">
            {t('loss')}
            <Tooltip content={
              <div>
                <strong>MSE (Mean Squared Error)</strong>
                <br/><br/>
                {t('formula')}: MSE = Σ(pred - real)² / n
                <br/><br/>
                {t('lossTooltip')}
              </div>
            }>
              <Info size={10} />
            </Tooltip>
          </div>
          <div className="text-2xl font-bold glow-text tabular-nums">
            {loss.toFixed(4)}
          </div>
        </div>
      </div>
      
      {/* Loss bar visual */}
      <div className="space-y-1">
        <div className="text-xs text-crt-green/50">{t('progress')}</div>
        <div className="h-2 bg-crt-bg border border-crt-green/30 rounded overflow-hidden">
          <div 
            className="h-full bg-crt-green transition-all duration-300"
            style={{ 
              width: `${Math.max(0, Math.min(100, 100 - loss * 100))}%`,
              boxShadow: '0 0 10px rgba(0, 255, 65, 0.5)'
            }}
          />
        </div>
      </div>
      
      {/* Controls principales */}
      <div className="flex justify-center gap-2">
        <button
          onClick={onToggleTraining}
          className={`p-3 rounded border transition-all ${
            isTraining 
              ? 'border-yellow-500/50 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20' 
              : 'border-crt-green/50 bg-crt-green/10 text-crt-green hover:bg-crt-green/20'
          }`}
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
          className="p-3 rounded border border-red-500/50 bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-all"
          title={t('resetTooltip')}
        >
          <RotateCcw size={20} />
        </button>
      </div>
      
      {/* ===== HIPERPARÁMETROS ===== */}
      <div className="border-t border-crt-green/20 pt-3 space-y-3">
        <div className="text-xs text-crt-green/70 uppercase tracking-wider text-center">
          ⚙️ {t('hyperparameters')}
        </div>
        
        {/* Speed control */}
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-xs text-crt-green/50 uppercase flex items-center gap-1">
              {t('speed')}
              <Tooltip content={
                <div>
                  <strong>{t('epochsPerTick')}</strong>
                  <br/><br/>
                  {t('speedTooltip')}
                </div>
              }>
                <Info size={10} />
              </Tooltip>
            </span>
            <span className="text-xs text-crt-green">
              {speed < 1 ? speed.toFixed(1) : speed} {t('epochsPerTick')}
            </span>
          </div>
          <input
            type="range"
            min="0.1"
            max="50"
            step="0.1"
            value={speed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            className="w-full accent-crt-green"
          />
          <div className="flex justify-between text-[10px] text-crt-green/30">
            <span>🐢 {t('slow')}</span>
            <span>{t('fast')} ⚡</span>
          </div>
        </div>
        
        {/* Learning rate control */}
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-xs text-crt-green/50 uppercase flex items-center gap-1">
              {t('learningRate')} (α)
              <Tooltip content={
                <div>
                  <strong>{t('learningRate')} (α)</strong>
                  <br/><br/>
                  {t('learningRateTooltip')}
                </div>
              }>
                <Info size={10} />
              </Tooltip>
            </span>
            <span className="text-xs text-crt-green font-mono">{config.learningRate.toFixed(4)}</span>
          </div>
          <input
            type="range"
            min="1"
            max="100"
            value={config.learningRate * 1000}
            onChange={(e) => onConfigChange({ learningRate: parseInt(e.target.value) / 1000 })}
            className="w-full accent-crt-green"
          />
          <div className="flex justify-between text-[10px] text-crt-green/30">
            <span>0.001 ({t('slow')})</span>
            <span>0.1 ({t('fast')})</span>
          </div>
        </div>
        
        {/* Momentum control */}
        <div className="space-y-1">
          <div className="flex justify-between items-center">
            <span className="text-xs text-crt-green/50 uppercase flex items-center gap-1">
              {t('momentum')} (β)
              <Tooltip content={
                <div>
                  <strong>{t('momentum')}</strong>
                  <br/><br/>
                  {t('momentumTooltip')}
                </div>
              }>
                <Info size={10} />
              </Tooltip>
            </span>
            <span className="text-xs text-crt-green font-mono">{config.momentum.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="99"
            value={config.momentum * 100}
            onChange={(e) => onConfigChange({ momentum: parseInt(e.target.value) / 100 })}
            className="w-full accent-crt-green"
          />
          <div className="flex justify-between text-[10px] text-crt-green/30">
            <span>{t('noInertia')}</span>
            <span>{t('highInertia')}</span>
          </div>
        </div>
      </div>
      
      {/* ===== ARQUITECTURA (expandible) ===== */}
      <div className="border-t border-crt-green/20 pt-3">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between text-xs text-crt-green/70 uppercase tracking-wider hover:text-crt-green transition-colors"
        >
          <span>🧠 {t('architecture')}</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        
        {showAdvanced && (
          <div className="mt-3 space-y-3">
            {/* Visualización de arquitectura actual */}
            <div className="text-center p-2 bg-crt-green/5 border border-crt-green/20 rounded">
              <div className="text-xs text-crt-green/50 mb-1">{t('currentArchitecture')}</div>
              <div className="text-sm font-mono text-crt-green">{architectureStr}</div>
            </div>
            
            {/* Hidden layers config */}
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs text-crt-green/50 uppercase flex items-center gap-1">
                  {t('hiddenLayers')}
                  <Tooltip content={
                    <div>
                      <strong>{t('hiddenLayers')}</strong>
                      <br/><br/>
                      {t('hiddenLayersTooltip')}
                      <br/><br/>
                      {t('architectureWarning')}
                    </div>
                  }>
                    <Info size={10} />
                  </Tooltip>
                </span>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={hiddenLayersInput}
                  onChange={(e) => handleHiddenLayersChange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && pendingChange) {
                      applyHiddenLayers();
                    }
                  }}
                  className={`flex-1 bg-black/50 border rounded px-2 py-1 text-xs font-mono text-crt-green focus:outline-none ${
                    pendingChange ? 'border-yellow-500/50' : 'border-crt-green/30 focus:border-crt-green'
                  }`}
                  placeholder="8,8"
                />
                {pendingChange && (
                  <button
                    onClick={applyHiddenLayers}
                    className="px-2 py-1 text-xs bg-crt-green/20 border border-crt-green/50 rounded text-crt-green hover:bg-crt-green/30 transition-all"
                  >
                    {t('apply')}
                  </button>
                )}
              </div>
              <div className="text-[10px] text-crt-green/30">
                {t('hiddenLayersHelp')}
              </div>
            </div>
            
            {/* Hidden activation */}
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs text-crt-green/50 uppercase flex items-center gap-1">
                  {t('hiddenActivation')}
                  <Tooltip content={
                    <div>
                      <strong>{t('hiddenActivation')}</strong>
                      <br/><br/>
                      {t('hiddenActivationTooltip')}
                    </div>
                  }>
                    <Info size={10} />
                  </Tooltip>
                </span>
              </div>
              <select
                value={config.hiddenActivation}
                onChange={(e) => onConfigChange({ hiddenActivation: e.target.value as ActivationFunction })}
                className="w-full bg-black/50 border border-crt-green/30 rounded px-2 py-1 text-xs font-mono text-crt-green focus:border-crt-green focus:outline-none"
              >
                <option value="tanh">tanh [-1, 1]</option>
                <option value="relu">ReLU [0, ∞)</option>
                <option value="sigmoid">sigmoid (0, 1)</option>
              </select>
              <div className="text-[10px] text-crt-green/40 font-mono">
                {activationDescriptions[config.hiddenActivation].formula}
              </div>
            </div>
            
            {/* Output activation */}
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs text-crt-green/50 uppercase flex items-center gap-1">
                  {t('outputActivation')}
                  <Tooltip content={
                    <div>
                      <strong>{t('outputActivation')}</strong>
                      <br/><br/>
                      {t('outputActivationTooltip')}
                    </div>
                  }>
                    <Info size={10} />
                  </Tooltip>
                </span>
              </div>
              <select
                value={config.outputActivation}
                onChange={(e) => onConfigChange({ outputActivation: e.target.value as ActivationFunction })}
                className="w-full bg-black/50 border border-crt-green/30 rounded px-2 py-1 text-xs font-mono text-crt-green focus:border-crt-green focus:outline-none"
              >
                <option value="linear">linear (regresión)</option>
                <option value="tanh">tanh [-1, 1]</option>
                <option value="sigmoid">sigmoid (0, 1)</option>
              </select>
            </div>
          </div>
        )}
      </div>
      
      {/* Status indicator */}
      <div className="flex items-center justify-center gap-2 pt-2 border-t border-crt-green/20">
        {isTraining ? (
          <>
            <Zap size={14} className="text-crt-green animate-pulse" />
            <span className="text-xs text-crt-green animate-pulse uppercase">{t('statusTraining')}</span>
          </>
        ) : (
          <>
            <ZapOff size={14} className="text-crt-green/50" />
            <span className="text-xs text-crt-green/50 uppercase">{t('statusPaused')}</span>
          </>
        )}
      </div>
    </div>
  );
}
