'use client';

import { useState, useRef, useCallback } from 'react';
import { useI18n } from '@/lib/i18n';
import { 
  PredefinedFunction, 
  DatasetConfig, 
  functionDescriptions,
  parseCSV,
  DEFAULT_DATASET_CONFIG 
} from '@/lib/dataset-generator';
import { Tooltip } from './Tooltip';

interface DatasetPanelProps {
  config: DatasetConfig;
  onConfigChange: (config: DatasetConfig) => void;
  onImportCSV: (X: number[], Y: number[]) => void;
  disabled?: boolean;
  isTestMode?: boolean;
  onToggleTestMode?: () => void;
}

// Agrupar funciones por categoría
const functionsByCategory = {
  trigonometric: ['sin', 'cos', 'tan', 'sin+cos', 'sin*cos', 'sin²', 'cos²', 'sin³'] as PredefinedFunction[],
  activation: ['tanh', 'sigmoid', 'gaussian'] as PredefinedFunction[],
  wave: ['sawtooth', 'square', 'triangle'] as PredefinedFunction[],
  custom: ['custom'] as PredefinedFunction[],
};

const categoryNames = {
  trigonometric: '📐 Trigonométricas',
  activation: '🧠 Activación',
  wave: '〰️ Ondas',
  custom: '✏️ Personalizada',
};

export function DatasetPanel({
  config,
  onConfigChange,
  onImportCSV,
  disabled = false,
  isTestMode = false,
  onToggleTestMode,
}: DatasetPanelProps) {
  const { t } = useI18n();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [customFormula, setCustomFormula] = useState(config.customFormula || 'sin(x) + 0.5*cos(2*x)');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [formulaError, setFormulaError] = useState<string | null>(null);

  const handleFunctionChange = useCallback((funcType: PredefinedFunction) => {
    onConfigChange({
      ...config,
      functionType: funcType,
      customFormula: funcType === 'custom' ? customFormula : undefined,
    });
    setFormulaError(null);
  }, [config, customFormula, onConfigChange]);

  const handleCustomFormulaChange = useCallback((formula: string) => {
    setCustomFormula(formula);
    
    // Validar fórmula básica
    try {
      const testFormula = formula
        .toLowerCase()
        .replace(/sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|log10|sqrt|abs|sign|floor|ceil|round/g, 'Math.$&')
        .replace(/pi/g, 'Math.PI')
        .replace(/\^/g, '**');
      
      const fn = new Function('x', `return ${testFormula}`);
      const result = fn(1);
      
      if (typeof result !== 'number' || !isFinite(result)) {
        setFormulaError('Resultado inválido');
      } else {
        setFormulaError(null);
        if (config.functionType === 'custom') {
          onConfigChange({
            ...config,
            customFormula: formula,
          });
        }
      }
    } catch (e) {
      setFormulaError('Sintaxis inválida');
    }
  }, [config, onConfigChange]);

  const applyCustomFormula = useCallback(() => {
    if (!formulaError) {
      onConfigChange({
        ...config,
        functionType: 'custom',
        customFormula: customFormula,
      });
    }
  }, [config, customFormula, formulaError, onConfigChange]);

  const handleFileImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const parsed = parseCSV(text);
      
      if (parsed && parsed.X.length > 0) {
        onImportCSV(parsed.X, parsed.Y);
      } else {
        alert('Error: No se pudo parsear el archivo CSV. Formato esperado: x,y en cada línea.');
      }
    };
    reader.readAsText(file);
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [onImportCSV]);

  const currentFunc = functionDescriptions[config.functionType];

  return (
    <div className="terminal-panel p-3 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-xs text-crt-green/70 uppercase tracking-wider flex items-center gap-2">
          📊 {t('datasetConfig' as any) || 'Dataset Configuration'}
        </div>
        <div className="flex items-center gap-2">
          {onToggleTestMode && (
            <button
              onClick={onToggleTestMode}
              className={`px-2 py-1 text-[10px] rounded border transition-all ${
                isTestMode
                  ? 'bg-purple-600/30 border-purple-500/50 text-purple-400'
                  : 'bg-gray-600/20 border-gray-500/30 text-gray-400 hover:bg-gray-600/30'
              }`}
            >
              {isTestMode ? '🔬 Test Mode ON' : '🔬 Test Mode'}
            </button>
          )}
        </div>
      </div>

      {/* Función actual */}
      <div className="bg-crt-green/5 border border-crt-green/20 rounded p-2">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-crt-green font-mono">{currentFunc.name}</div>
            <div className="text-xs text-crt-green/60">f(x) = {currentFunc.formula}</div>
          </div>
          <div className="text-[10px] text-crt-green/40">{currentFunc.description}</div>
        </div>
      </div>

      {/* Selector de función por categoría */}
      <div className="space-y-2">
        {Object.entries(functionsByCategory).map(([category, functions]) => (
          <div key={category}>
            <div className="text-[10px] text-crt-green/50 mb-1">
              {categoryNames[category as keyof typeof categoryNames]}
            </div>
            <div className="flex flex-wrap gap-1">
              {functions.map((funcType) => {
                const func = functionDescriptions[funcType];
                return (
                  <Tooltip key={funcType} content={`${func.formula}\n${func.description}`}>
                    <button
                      onClick={() => handleFunctionChange(funcType)}
                      disabled={disabled}
                      className={`px-2 py-1 text-[10px] rounded border transition-all ${
                        config.functionType === funcType
                          ? 'bg-crt-green/20 border-crt-green/50 text-crt-green'
                          : 'bg-black/30 border-crt-green/20 text-crt-green/60 hover:border-crt-green/40'
                      } disabled:opacity-50`}
                    >
                      {func.name}
                    </button>
                  </Tooltip>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Fórmula personalizada */}
      {config.functionType === 'custom' && (
        <div className="space-y-2 p-2 bg-yellow-400/5 border border-yellow-400/20 rounded">
          <div className="text-[10px] text-yellow-400/70 uppercase">✏️ Fórmula Personalizada</div>
          <div className="flex gap-2">
            <input
              type="text"
              value={customFormula}
              onChange={(e) => handleCustomFormulaChange(e.target.value)}
              disabled={disabled}
              placeholder="sin(x) + 0.5*cos(2*x)"
              className="flex-1 px-2 py-1 text-xs font-mono bg-black/50 border border-crt-green/30 rounded text-crt-green focus:border-crt-green/60 focus:outline-none"
            />
            <button
              onClick={applyCustomFormula}
              disabled={disabled || !!formulaError}
              className="px-2 py-1 text-[10px] bg-crt-green/20 border border-crt-green/50 rounded text-crt-green hover:bg-crt-green/30 disabled:opacity-50"
            >
              Aplicar
            </button>
          </div>
          {formulaError && (
            <div className="text-[10px] text-red-400">⚠️ {formulaError}</div>
          )}
          <div className="text-[9px] text-crt-green/40">
            Funciones: sin, cos, tan, exp, log, sqrt, abs | Constantes: pi, e | Operadores: +, -, *, /, ^
          </div>
        </div>
      )}

      {/* Toggle opciones avanzadas */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full text-[10px] text-crt-green/50 hover:text-crt-green/70 flex items-center justify-center gap-1"
      >
        {showAdvanced ? '▲' : '▼'} Opciones avanzadas
      </button>

      {/* Opciones avanzadas */}
      {showAdvanced && (
        <div className="space-y-3 pt-2 border-t border-crt-green/20">
          {/* Número de puntos */}
          <div>
            <div className="flex justify-between text-[10px] mb-1">
              <span className="text-crt-green/60">Puntos de datos</span>
              <span className="text-crt-green font-mono">{config.numPoints}</span>
            </div>
            <input
              type="range"
              min={50}
              max={500}
              step={10}
              value={config.numPoints}
              onChange={(e) => onConfigChange({ ...config, numPoints: parseInt(e.target.value) })}
              disabled={disabled}
              className="w-full accent-crt-green"
            />
          </div>

          {/* Nivel de ruido */}
          <div>
            <div className="flex justify-between text-[10px] mb-1">
              <span className="text-crt-green/60">Nivel de ruido</span>
              <span className="text-crt-green font-mono">{(config.noiseLevel * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={config.noiseLevel * 100}
              onChange={(e) => onConfigChange({ ...config, noiseLevel: parseInt(e.target.value) / 100 })}
              disabled={disabled}
              className="w-full accent-crt-green"
            />
          </div>

          {/* Rango X */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] text-crt-green/60 mb-1">X mín</div>
              <input
                type="number"
                value={config.xMin}
                onChange={(e) => onConfigChange({ ...config, xMin: parseFloat(e.target.value) || -6 })}
                disabled={disabled}
                className="w-full px-2 py-1 text-xs font-mono bg-black/50 border border-crt-green/30 rounded text-crt-green"
              />
            </div>
            <div>
              <div className="text-[10px] text-crt-green/60 mb-1">X máx</div>
              <input
                type="number"
                value={config.xMax}
                onChange={(e) => onConfigChange({ ...config, xMax: parseFloat(e.target.value) || 6 })}
                disabled={disabled}
                className="w-full px-2 py-1 text-xs font-mono bg-black/50 border border-crt-green/30 rounded text-crt-green"
              />
            </div>
          </div>

          {/* Importar CSV */}
          <div className="pt-2 border-t border-crt-green/20">
            <div className="text-[10px] text-crt-green/60 mb-2">📁 Importar Dataset</div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.txt"
              onChange={handleFileImport}
              disabled={disabled}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              className="w-full px-3 py-2 text-xs bg-cyan-600/20 border border-cyan-500/30 rounded text-cyan-400 hover:bg-cyan-600/30 transition-all disabled:opacity-50"
            >
              📂 Cargar archivo CSV
            </button>
            <div className="text-[9px] text-crt-green/40 mt-1">
              Formato: x,y en cada línea (con o sin header)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DatasetPanel;
