'use client';

import { useMemo, useState, useEffect } from 'react';
import type { NetworkState, NeuronInfo, LayerType, PatternInfo } from '@/lib/neural-network';
import { activationDescriptions } from '@/lib/neural-network';
import { useI18n } from '@/lib/i18n';

interface NeuralNetworkVizProps {
  networkState: NetworkState;
  getNeuronInfo: (layer: number, index: number) => NeuronInfo;
  getNeuronPattern: (layer: number, index: number) => { x: number; y: number }[];
  width?: number;
  height?: number;
  // Optional controls
  isTraining?: boolean;
  epoch?: number;
  onToggleTraining?: () => void;
  onStop?: () => void;
  onReset?: () => void;
}

// Mini gráfico del patrón de la neurona
function MiniPatternGraph({ points, width = 120, height = 60 }: { 
  points: { x: number; y: number }[]; 
  width?: number; 
  height?: number;
}) {
  if (!points || points.length === 0) return null;
  
  // Encontrar rangos
  const xMin = Math.min(...points.map(p => p.x));
  const xMax = Math.max(...points.map(p => p.x));
  const yValues = points.map(p => p.y);
  const yMin = Math.min(...yValues, -1);
  const yMax = Math.max(...yValues, 1);
  const yRange = Math.max(yMax - yMin, 0.1);
  
  // Crear path
  const pathPoints = points.map((p, i) => {
    const px = ((p.x - xMin) / (xMax - xMin)) * (width - 10) + 5;
    const py = height - 5 - ((p.y - yMin) / yRange) * (height - 10);
    return `${i === 0 ? 'M' : 'L'} ${px} ${py}`;
  }).join(' ');
  
  // Línea del cero
  const zeroY = height - 5 - ((0 - yMin) / yRange) * (height - 10);
  
  return (
    <svg width={width} height={height} className="bg-black/50 rounded border border-crt-green/30">
      {/* Línea del cero */}
      <line 
        x1={5} y1={zeroY} x2={width - 5} y2={zeroY} 
        stroke="rgba(0, 255, 65, 0.2)" 
        strokeDasharray="2"
      />
      {/* Patrón */}
      <path 
        d={pathPoints} 
        fill="none" 
        stroke="rgba(0, 255, 65, 0.9)" 
        strokeWidth={1.5}
        style={{ filter: 'drop-shadow(0 0 3px rgba(0, 255, 65, 0.5))' }}
      />
    </svg>
  );
}

// Helper functions for i18n translation
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getLayerName(info: NeuronInfo, t: (key: any) => string): string {
  switch (info.layerType) {
    case 'input': return t('inputLayer');
    case 'output': return t('outputLayer');
    case 'hidden': return `${t('hiddenLayer')} ${info.layerNumber}`;
    default: return '';
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getPatternText(patternInfo: PatternInfo, t: (key: any, replacements?: Record<string, string>) => string): string {
  switch (patternInfo.type) {
    case 'inputSensitivity':
      return `${t('patternInputSensitivity')}: ${patternInfo.sign}${patternInfo.value?.toFixed(3)}`;
    case 'combinesHidden':
      return t('patternCombinesHidden');
    case 'maxWeight':
      return `${t('patternMaxWeight', { neuronNum: String(patternInfo.neuronNum) })}: ${patternInfo.sign}${patternInfo.value?.toFixed(3)}`;
    case 'receivesX':
      return t('patternReceivesX');
    default:
      return '';
  }
}

// Panel de información de neurona (tooltip expandido) - usa fixed para evitar overflow
function NeuronInfoPanel({ 
  info, 
  pattern,
  position,
  networkState,
  t
}: { 
  info: NeuronInfo | null; 
  pattern: { x: number; y: number }[];
  position: { x: number; y: number };
  networkState: NetworkState;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  t: (key: any, replacements?: Record<string, string>) => string;
}) {
  if (!info) return null;
  
  const activationInfo = activationDescriptions[info.activation];
  
  // Para neuronas de la última capa oculta, obtener el peso hacia la salida
  const numLayers = networkState.layers.length;
  const isLastHiddenLayer = info.layer === numLayers - 2;
  const outputWeight = isLastHiddenLayer && networkState.weights.length > 0
    ? networkState.weights[networkState.weights.length - 1][0]?.[info.index] ?? null
    : null;
  
  // Calcular posición segura para el tooltip (fixed positioning)
  const tooltipWidth = 320;
  const tooltipHeight = 400; // altura aproximada máxima
  const padding = 10;
  
  // Posición X: preferir a la derecha, pero si no cabe, a la izquierda
  let left = position.x + 20;
  if (left + tooltipWidth > window.innerWidth - padding) {
    left = position.x - tooltipWidth - 20;
  }
  left = Math.max(padding, Math.min(left, window.innerWidth - tooltipWidth - padding));
  
  // Posición Y: centrar verticalmente respecto al mouse, pero mantener en pantalla
  let top = position.y - tooltipHeight / 2;
  top = Math.max(padding, Math.min(top, window.innerHeight - tooltipHeight - padding));
  
  return (
    <div 
      className="fixed z-[9999] w-80 p-3 bg-black/95 border border-crt-green/70 rounded-lg shadow-2xl pointer-events-none"
      style={{ 
        left,
        top,
        maxHeight: 'calc(100vh - 20px)',
        overflowY: 'auto',
        boxShadow: '0 0 20px rgba(0, 255, 65, 0.3)'
      }}
    >
      {/* Header */}
      <div className="border-b border-crt-green/30 pb-2 mb-2">
        <div className="text-sm font-bold text-crt-green">
          🧠 {getLayerName(info, t)} - {t('neuron')} {info.index + 1}
        </div>
      </div>
      
      {/* Mini gráfico del patrón */}
      <div className="mb-3">
        <div className="text-[10px] text-crt-green/50 uppercase mb-1">
          📈 {info.layer === 0 ? t('inputIdentity') : 
              info.layer === numLayers - 1 ? t('finalOutput') :
              t('neuronResponse')}
        </div>
        <MiniPatternGraph points={pattern} width={260} height={70} />
        <div className="text-[9px] text-crt-green/40 mt-1">
          {info.layer === 0 
            ? t('normalizedInput')
            : info.layer === numLayers - 1 
              ? t('finalPrediction')
              : t('neuronCurve', { activation: info.activation })
          }
        </div>
        {outputWeight !== null && (
          <div className={`text-[10px] font-mono mt-1 px-2 py-0.5 rounded ${
            outputWeight > 0 ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
          }`}>
            ⚡ {t('weightToOutput')}: {outputWeight > 0 ? '+' : ''}{outputWeight.toFixed(4)}
            <span className="text-crt-green/40 ml-2">
              ({outputWeight > 0 ? t('addsToOutput') : t('subtractsFromOutput')})
            </span>
          </div>
        )}
      </div>
      
      {/* Activación */}
      <div className="mb-2">
        <div className="text-[10px] text-crt-green/50 uppercase">{t('activationFunction')}</div>
        <div className="text-xs text-crt-green font-mono">{info.activation}</div>
        <div className="text-[10px] text-crt-green/40">{activationInfo.formula}</div>
        <div className="text-[10px] text-crt-green/40">{t('range')}: {activationInfo.range}</div>
      </div>
      
      {/* Estado actual */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        <div className="bg-crt-green/5 p-1.5 rounded border border-crt-green/20">
          <div className="text-[10px] text-crt-green/50 uppercase">{t('input')} (z)</div>
          <div className="text-xs text-crt-green font-mono">{info.currentInput.toFixed(4)}</div>
        </div>
        <div className="bg-crt-green/5 p-1.5 rounded border border-crt-green/20">
          <div className="text-[10px] text-crt-green/50 uppercase">{t('output')} (a)</div>
          <div className="text-xs text-crt-green font-mono">{info.currentOutput.toFixed(4)}</div>
        </div>
      </div>
      
      {/* Pesos */}
      {info.weights.length > 0 && (
        <div className="mb-2">
          <div className="text-[10px] text-crt-green/50 uppercase mb-1">
            {t('weights')} ({info.weights.length} {t('connections')})
          </div>
          <div className="flex flex-wrap gap-1 max-h-14 overflow-y-auto">
            {info.weights.map((w, i) => (
              <span 
                key={i}
                className={`text-[10px] font-mono px-1 rounded ${
                  w > 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
                }`}
              >
                w{i + 1}: {w.toFixed(3)}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* Bias */}
      {info.layer > 0 && (
        <div className="mb-2">
          <div className="text-[10px] text-crt-green/50 uppercase">{t('bias')}</div>
          <div className="text-xs text-crt-green font-mono">{info.bias.toFixed(4)}</div>
        </div>
      )}
      
      {/* Patrón descripción */}
      <div className="border-t border-crt-green/30 pt-2">
        <div className="text-[10px] text-crt-green/50 uppercase">{t('interpretation')}</div>
        <div className="text-xs text-crt-green/80">{getPatternText(info.patternInfo, t)}</div>
      </div>
      
      {/* Fórmula */}
      {info.layer > 0 && (
        <div className="mt-2 p-1.5 bg-crt-green/5 rounded border border-crt-green/20">
          <div className="text-[10px] text-crt-green/40 font-mono">
            {t('formula')}: z = Σ(wᵢ × aᵢ) + b = {info.currentInput.toFixed(3)}
          </div>
          <div className="text-[10px] text-crt-green/40 font-mono">
            a = {info.activation}(z) = {info.currentOutput.toFixed(3)}
          </div>
        </div>
      )}
    </div>
  );
}

export default function NeuralNetworkViz({
  networkState,
  getNeuronInfo,
  getNeuronPattern,
  width = 800,
  height = 350,
  isTraining,
  epoch,
  onToggleTraining,
  onStop,
  onReset,
}: NeuralNetworkVizProps) {
  const { t } = useI18n();
  const [hoveredNeuron, setHoveredNeuron] = useState<{ info: NeuronInfo; pattern: { x: number; y: number }[] } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  
  // Forward pass animation state
  const [forwardPassLayer, setForwardPassLayer] = useState<number>(-1); // -1 = no animation, 0-n = current layer
  const [showForwardPass, setShowForwardPass] = useState(true); // Toggle for animation
  
  const hasControls = onToggleTraining !== undefined;
  
  // Obtener arquitectura desde el state
  const networkConfig = useMemo(() => {
    if (!networkState.layers || networkState.layers.length === 0) {
      return [1, 8, 8, 1];
    }
    return networkState.layers.map(l => l.activation.length);
  }, [networkState.layers]);
  
  // Calcular posiciones de las neuronas
  const neurons = useMemo(() => {
    const result: { 
      x: number; 
      y: number; 
      activation: number; 
      layer: number; 
      index: number;
      layerName: string;
    }[] = [];
    
    const padding = 60;
    const usableWidth = width - padding * 2;
    const usableHeight = height - 60;
    const layerSpacing = usableWidth / (networkConfig.length - 1);
    
    networkConfig.forEach((count, layerIdx) => {
      const neuronSpacing = usableHeight / (count + 1);
      
      let layerName: string;
      if (layerIdx === 0) layerName = t('inputLayer');
      else if (layerIdx === networkConfig.length - 1) layerName = t('outputLayer');
      else layerName = `${t('hiddenLayer')} ${layerIdx}`;
      
      for (let i = 0; i < count; i++) {
        let activation = 0;
        
        if (networkState.layers && networkState.layers[layerIdx]) {
          activation = Math.abs(networkState.layers[layerIdx].activation[i] || 0);
        }
        
        result.push({
          x: padding + layerSpacing * layerIdx,
          y: neuronSpacing * (i + 1) + 20,
          activation: Math.min(activation, 1),
          layer: layerIdx,
          index: i,
          layerName,
        });
      }
    });
    
    return result;
  }, [networkState.layers, networkConfig, width, height]);

  // Calcular conexiones con pesos
  const connections = useMemo(() => {
    const result: { 
      x1: number; 
      y1: number; 
      x2: number; 
      y2: number; 
      weight: number;
      opacity: number;
      toLayer: number;
    }[] = [];
    
    let offset = 0;
    for (let l = 0; l < networkConfig.length - 1; l++) {
      const currentLayerSize = networkConfig[l];
      const nextLayerSize = networkConfig[l + 1];
      const currentLayerStart = offset;
      const nextLayerStart = offset + currentLayerSize;
      
      for (let i = 0; i < currentLayerSize; i++) {
        for (let j = 0; j < nextLayerSize; j++) {
          const from = neurons[currentLayerStart + i];
          const to = neurons[nextLayerStart + j];
          
          if (from && to) {
            // Obtener peso real si existe
            let weight = 0;
            if (networkState.weights && networkState.weights[l] && networkState.weights[l][j]) {
              weight = networkState.weights[l][j][i] || 0;
            }
            
            result.push({
              x1: from.x,
              y1: from.y,
              x2: to.x,
              y2: to.y,
              weight,
              opacity: Math.min(Math.abs(weight) * 0.5 + 0.1, 0.8),
              toLayer: l + 1,
            });
          }
        }
      }
      
      offset += currentLayerSize;
    }
    
    return result;
  }, [neurons, networkConfig, networkState.weights]);

  // Layer labels
  const layerLabels = useMemo(() => {
    const labels: { x: number; text: string; subtext: string }[] = [];
    const padding = 60;
    const usableWidth = width - padding * 2;
    const layerSpacing = usableWidth / (networkConfig.length - 1);
    
    networkConfig.forEach((count, idx) => {
      let text: string;
      let subtext: string;
      
      if (idx === 0) {
        text = t('inputLayer');
        subtext = 'x';
      } else if (idx === networkConfig.length - 1) {
        text = t('outputLayer');
        subtext = networkState.config?.outputActivation || 'linear';
      } else {
        text = `${t('hiddenLayer')} ${idx}`;
        subtext = networkState.config?.hiddenActivation || 'tanh';
      }
      
      labels.push({
        x: padding + layerSpacing * idx,
        text: `${text} (${count})`,
        subtext,
      });
    });
    
    return labels;
  }, [networkConfig, networkState.config, width, t]);

  // Forward pass animation effect
  useEffect(() => {
    if (!showForwardPass || !isTraining) {
      setForwardPassLayer(-1);
      return;
    }
    
    // Animar el forward pass: ciclar por las capas
    const totalLayers = networkConfig.length;
    const animationSpeed = 200; // ms por capa
    
    const interval = setInterval(() => {
      setForwardPassLayer(prev => {
        const next = prev + 1;
        return next >= totalLayers ? 0 : next;
      });
    }, animationSpeed);
    
    return () => clearInterval(interval);
  }, [showForwardPass, isTraining, networkConfig.length]);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    // Usar coordenadas globales (clientX/Y) para el tooltip fixed
    setMousePos({
      x: e.clientX,
      y: e.clientY,
    });
  };

  const handleNeuronHover = (layer: number, index: number) => {
    const info = getNeuronInfo(layer, index);
    const pattern = getNeuronPattern(layer, index);
    setHoveredNeuron({ info, pattern });
  };

  return (
    <div className="terminal-panel p-4 relative" style={{ width }}>
      <div className="text-center mb-2 text-xs text-crt-green/70 uppercase tracking-wider flex items-center justify-center gap-2">
        <span>🧠 {t('networkArchitecture')}</span>
        <span className="text-crt-green/40">|</span>
        <span className="font-mono">{networkConfig.join(' → ')}</span>
      </div>
      
      {/* Mini controls if provided */}
      {hasControls && (
        <div className="flex justify-center gap-2 mb-2">
          <button
            onClick={onToggleTraining}
            className={`px-3 py-1 text-xs rounded border transition-all flex items-center gap-1 ${
              isTraining 
                ? 'bg-yellow-600/20 hover:bg-yellow-600/40 border-yellow-500/50 text-yellow-400'
                : 'bg-green-600/20 hover:bg-green-600/40 border-green-500/50 text-green-400'
            }`}
          >
            {isTraining ? (
              <>
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M5.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75A.75.75 0 007.25 3h-1.5zM12.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75a.75.75 0 00-.75-.75h-1.5z" />
                </svg>
                {t('pause')}
              </>
            ) : (
              <>
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                </svg>
                {t('start')}
              </>
            )}
          </button>
          
          {onStop && (
            <button
              onClick={onStop}
              disabled={epoch === 0}
              className="px-3 py-1 text-xs rounded border bg-orange-600/20 hover:bg-orange-600/40 border-orange-500/50 text-orange-400 transition-all flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <rect x="4" y="4" width="12" height="12" rx="1" />
              </svg>
              {t('stop')}
            </button>
          )}
          
          {onReset && (
            <button
              onClick={onReset}
              className="px-3 py-1 text-xs rounded border bg-red-600/20 hover:bg-red-600/40 border-red-500/50 text-red-400 transition-all flex items-center gap-1"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              {t('reset')}
            </button>
          )}
          
          {/* Toggle forward pass visualization */}
          <button
            onClick={() => setShowForwardPass(!showForwardPass)}
            className={`px-3 py-1 text-xs rounded border transition-all flex items-center gap-1 ${
              showForwardPass
                ? 'bg-cyan-600/20 hover:bg-cyan-600/40 border-cyan-500/50 text-cyan-400'
                : 'bg-gray-600/20 hover:bg-gray-600/40 border-gray-500/50 text-gray-400'
            }`}
            title={t('forwardPassTooltip' as any)}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            {t('forwardPass' as any)}
          </button>
        </div>
      )}
      
      <div className="text-center text-[10px] text-crt-green/40 mb-2">
        💡 {t('hoverHint')}
        {showForwardPass && isTraining && (
          <span className="ml-2 text-cyan-400">| ⚡ {t('forwardPass' as any)}: {t('hiddenLayer' as any).toLowerCase()} {forwardPassLayer >= 0 ? forwardPassLayer : '-'}</span>
        )}
      </div>
      
      <svg 
        width={width - 32} 
        height={height} 
        className="mx-auto"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredNeuron(null)}
      >
        {/* Fondo de capas */}
        {layerLabels.map((label, idx) => (
          <g key={`bg-${idx}`}>
            <rect
              x={label.x - 30}
              y={15}
              width={60}
              height={height - 50}
              fill="rgba(0, 255, 65, 0.02)"
              stroke="rgba(0, 255, 65, 0.1)"
              strokeDasharray="4"
              rx={5}
            />
          </g>
        ))}
        
        {/* Conexiones */}
        {connections.map((conn, i) => {
          // Highlight connections leading TO the current forward pass layer
          const isForwardPassConnection = showForwardPass && conn.toLayer === forwardPassLayer;
          const baseStroke = conn.weight > 0 
            ? `rgba(0, 255, 65, ${conn.opacity})` 
            : `rgba(255, 100, 100, ${conn.opacity})`;
          const forwardStroke = `rgba(0, 255, 255, ${Math.min(1, conn.opacity + 0.5)})`;
          
          return (
            <line
              key={i}
              x1={conn.x1}
              y1={conn.y1}
              x2={conn.x2}
              y2={conn.y2}
              stroke={isForwardPassConnection ? forwardStroke : baseStroke}
              strokeWidth={isForwardPassConnection ? 2 : 0.5 + Math.abs(conn.weight) * 0.5}
              style={{
                transition: 'stroke 0.15s ease, stroke-width 0.15s ease',
                filter: isForwardPassConnection ? 'drop-shadow(0 0 4px cyan)' : 'none'
              }}
            />
          );
        })}
        
        {/* Neuronas */}
        {neurons.map((neuron, i) => {
          const isHovered = hoveredNeuron?.info?.layer === neuron.layer && hoveredNeuron?.info?.index === neuron.index;
          const baseRadius = neuron.layer === 0 || neuron.layer === networkConfig.length - 1 ? 14 : 10;
          const isForwardPassNeuron = showForwardPass && neuron.layer === forwardPassLayer;
          
          return (
            <g 
              key={i}
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => handleNeuronHover(neuron.layer, neuron.index)}
            >
              {/* Glow exterior - cyan para forward pass, verde para activación normal */}
              <circle
                cx={neuron.x}
                cy={neuron.y}
                r={baseRadius + (isForwardPassNeuron ? 12 : neuron.activation * 10) + (isHovered ? 6 : 0)}
                fill={isForwardPassNeuron 
                  ? `rgba(0, 255, 255, 0.6)` 
                  : `rgba(0, 255, 65, ${neuron.activation * 0.5 + (isHovered ? 0.25 : 0)})`
                }
                style={{ 
                  filter: isForwardPassNeuron ? 'blur(6px)' : neuron.activation > 0.5 ? 'blur(3px)' : 'blur(2px)',
                  transition: 'all 0.15s ease'
                }}
              />
              
              {/* Círculo medio - cyan para forward pass, o de gris oscuro a verde brillante */}
              <circle
                cx={neuron.x}
                cy={neuron.y}
                r={baseRadius}
                fill={isForwardPassNeuron
                  ? `rgba(0, 200, 220, 0.9)`
                  : `rgba(${neuron.activation * 50}, ${50 + neuron.activation * 205}, ${30 + neuron.activation * 35}, ${0.7 + neuron.activation * 0.3})`
                }
                stroke={isForwardPassNeuron 
                  ? 'rgba(0, 255, 255, 1)' 
                  : isHovered 
                    ? 'rgba(255, 255, 255, 0.9)' 
                    : `rgba(0, 255, 65, ${0.3 + neuron.activation * 0.5})`
                }
                strokeWidth={isForwardPassNeuron ? 3 : isHovered ? 2.5 : 1.5}
                style={{ transition: 'all 0.15s ease' }}
              />
              
              {/* Centro brillante - más contraste */}
              <circle
                cx={neuron.x}
                cy={neuron.y}
                r={baseRadius * 0.5}
                fill={isForwardPassNeuron
                  ? `rgba(200, 255, 255, 0.9)`
                  : `rgba(${150 + neuron.activation * 105}, ${180 + neuron.activation * 75}, ${150 + neuron.activation * 105}, ${0.6 + neuron.activation * 0.4})`
                }
                style={{ transition: 'all 0.15s ease' }}
              />
              
              {/* Índice de neurona */}
              {(isHovered || neuron.layer === 0 || neuron.layer === networkConfig.length - 1) && (
                <text
                  x={neuron.x}
                  y={neuron.y + 4}
                  textAnchor="middle"
                  fill="rgba(0, 0, 0, 0.9)"
                  fontSize="9"
                  fontWeight="bold"
                >
                  {neuron.layer === 0 ? 'x' : neuron.layer === networkConfig.length - 1 ? 'ŷ' : (neuron.index + 1)}
                </text>
              )}
            </g>
          );
        })}
        
        {/* Labels de capas */}
        {layerLabels.map((label, idx) => (
          <g key={`label-${idx}`}>
            <text 
              x={label.x} 
              y={height - 15} 
              fill="rgba(0, 255, 65, 0.7)" 
              fontSize="10" 
              textAnchor="middle"
              fontWeight="bold"
            >
              {label.text}
            </text>
            <text 
              x={label.x} 
              y={height - 3} 
              fill="rgba(0, 255, 65, 0.4)" 
              fontSize="9" 
              textAnchor="middle"
              fontFamily="monospace"
            >
              {label.subtext}
            </text>
          </g>
        ))}
      </svg>
      
      {/* Leyenda */}
      <div className="flex justify-center gap-6 mt-2 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-crt-green/30" />
          <span className="text-crt-green/50">{t('lowActivation')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-crt-green" style={{ boxShadow: '0 0 8px rgba(0,255,65,0.8)' }} />
          <span className="text-crt-green/50">{t('highActivation')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-0.5 bg-green-500/50" />
          <span className="text-crt-green/50">{t('positiveWeight')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-0.5 bg-red-500/50" />
          <span className="text-crt-green/50">{t('negativeWeight')}</span>
        </div>
      </div>
      
      {/* Panel de información de neurona - usa position fixed, coordenadas globales */}
      {hoveredNeuron && (
        <NeuronInfoPanel 
          info={hoveredNeuron.info}
          pattern={hoveredNeuron.pattern}
          position={mousePos}
          networkState={networkState}
          t={t}
        />
      )}
    </div>
  );
}
