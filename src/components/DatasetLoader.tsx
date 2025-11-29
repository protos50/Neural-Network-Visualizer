'use client';

import { useState, useEffect } from 'react';
import { Database, Upload, FileSpreadsheet, Check, AlertCircle } from 'lucide-react';

interface DatasetInfo {
  name: string;
  filename: string;
  rows: number;
  inputCols: string[];
  outputCols: string[];
  inputSize: number;
  outputSize: number;
  headers: string[];
}

interface LoadedDataset {
  X: number[][];  // Matriz de inputs
  Y: number[][];  // Matriz de outputs
  inputCols: string[];
  outputCols: string[];
  inputSize: number;
  outputSize: number;
  rows: number;
  name: string;
}

interface DatasetLoaderProps {
  onDatasetLoad: (dataset: LoadedDataset) => void;
  disabled?: boolean;
}

export default function DatasetLoader({ onDatasetLoad, disabled = false }: DatasetLoaderProps) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [loadedDataset, setLoadedDataset] = useState<string | null>(null);

  // Cargar lista de datasets disponibles
  useEffect(() => {
    fetch('/api/datasets')
      .then(res => res.json())
      .then(data => {
        setDatasets(data.datasets || []);
      })
      .catch(err => {
        console.error('Error fetching datasets:', err);
        setError('No se pudo cargar la lista de datasets');
      });
  }, []);

  // Parsear CSV a números
  const parseCSV = (content: string, info: DatasetInfo): LoadedDataset | null => {
    try {
      const lines = content.split('\n')
        .filter(l => l.trim() && !l.startsWith('#'));
      
      if (lines.length < 2) {
        setError('El CSV debe tener al menos un header y una fila de datos');
        return null;
      }

      const headers = lines[0].split(',').map(h => h.trim());
      const inputIndices = info.inputCols.map(col => headers.indexOf(col));
      const outputIndices = info.outputCols.map(col => headers.indexOf(col));

      // Verificar que todas las columnas existan
      if (inputIndices.includes(-1) || outputIndices.includes(-1)) {
        setError('Algunas columnas especificadas no existen en el CSV');
        return null;
      }

      const X: number[][] = [];
      const Y: number[][] = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => parseFloat(v.trim()));
        
        // Verificar que todos los valores sean números válidos
        if (values.some(isNaN)) {
          console.warn(`Fila ${i} contiene valores no numéricos, saltando...`);
          continue;
        }

        const inputRow = inputIndices.map(idx => values[idx]);
        const outputRow = outputIndices.map(idx => values[idx]);

        X.push(inputRow);
        Y.push(outputRow);
      }

      if (X.length === 0) {
        setError('No se encontraron filas válidas en el CSV');
        return null;
      }

      return {
        X,
        Y,
        inputCols: info.inputCols,
        outputCols: info.outputCols,
        inputSize: info.inputSize,
        outputSize: info.outputSize,
        rows: X.length,
        name: info.name,
      };
    } catch (err) {
      setError('Error parseando el CSV');
      return null;
    }
  };

  // Cargar dataset seleccionado
  const handleLoadDataset = async () => {
    if (!selectedDataset) return;

    const info = datasets.find(d => d.name === selectedDataset);
    if (!info) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/datasets/${info.filename}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const content = await response.text();
      const parsed = parseCSV(content, info);
      
      if (parsed) {
        setLoadedDataset(selectedDataset);
        onDatasetLoad(parsed);
      }
    } catch (err) {
      setError('Error cargando el dataset: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const selectedInfo = datasets.find(d => d.name === selectedDataset);

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2 text-xs text-cyan-400/70 uppercase tracking-wider">
        <Database size={14} />
        <span>Cargar Dataset CSV</span>
      </div>

      {/* Selector */}
      <div className="space-y-2">
        <select
          value={selectedDataset || ''}
          onChange={(e) => setSelectedDataset(e.target.value || null)}
          className="w-full bg-black/50 border border-cyan-400/30 rounded px-2 py-1.5 text-xs font-mono text-cyan-400 focus:border-cyan-400 focus:outline-none"
          disabled={disabled || loading}
        >
          <option value="">-- Seleccionar dataset --</option>
          {datasets.map(d => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.rows} filas)
            </option>
          ))}
        </select>

        {/* Info del dataset seleccionado */}
        {selectedInfo && (
          <div className="p-2 bg-cyan-400/5 border border-cyan-400/20 rounded text-[10px] space-y-1">
            <div className="flex justify-between">
              <span className="text-cyan-400/50">Entradas:</span>
              <span className="text-green-400 font-mono">
                {selectedInfo.inputSize} ({selectedInfo.inputCols.join(', ')})
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-cyan-400/50">Salidas:</span>
              <span className="text-orange-400 font-mono">
                {selectedInfo.outputSize} ({selectedInfo.outputCols.join(', ')})
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-cyan-400/50">Filas:</span>
              <span className="text-cyan-400 font-mono">{selectedInfo.rows}</span>
            </div>
          </div>
        )}

        {/* Botón cargar */}
        <button
          onClick={handleLoadDataset}
          disabled={!selectedDataset || disabled || loading}
          className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-mono transition-all ${
            selectedDataset && !disabled && !loading
              ? 'bg-cyan-400/20 border border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/30'
              : 'bg-gray-600/20 border border-gray-500/30 text-gray-500 cursor-not-allowed'
          }`}
        >
          {loading ? (
            <>
              <div className="w-3 h-3 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
              Cargando...
            </>
          ) : loadedDataset === selectedDataset ? (
            <>
              <Check size={14} />
              Dataset cargado
            </>
          ) : (
            <>
              <Upload size={14} />
              Cargar Dataset
            </>
          )}
        </button>

        {/* Estado cargado */}
        {loadedDataset && (
          <div className="flex items-center gap-2 text-[10px] text-green-400/70">
            <FileSpreadsheet size={12} />
            <span>Usando: <strong>{loadedDataset}</strong></span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 text-[10px] text-red-400">
            <AlertCircle size={12} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Nota */}
      <div className="text-[9px] text-cyan-400/40 leading-relaxed">
        💡 Los CSVs deben estar en <code className="bg-black/30 px-1 rounded">/public/datasets/</code>
        <br />
        Usa comentarios <code className="bg-black/30 px-1 rounded"># INPUT_COLS:</code> y <code className="bg-black/30 px-1 rounded"># OUTPUT_COLS:</code> para definir columnas.
      </div>
    </div>
  );
}
