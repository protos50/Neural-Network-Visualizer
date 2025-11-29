'use client';

import { Brain, Cpu } from 'lucide-react';

export type NetworkMode = 'manual' | 'tensorflow';

interface ModeSelectorProps {
  mode: NetworkMode;
  onModeChange: (mode: NetworkMode) => void;
  disabled?: boolean;
}

export default function ModeSelector({ mode, onModeChange, disabled = false }: ModeSelectorProps) {
  return (
    <div className="flex items-center gap-2 justify-center">
      <button
        onClick={() => onModeChange('manual')}
        disabled={disabled}
        className={`flex items-center gap-2 px-3 py-1.5 rounded border transition-all text-xs ${
          mode === 'manual'
            ? 'border-crt-green bg-crt-green/20 text-crt-green'
            : 'border-crt-green/30 text-crt-green/50 hover:border-crt-green/50'
        } disabled:opacity-50`}
      >
        <Brain className="w-4 h-4" />
        <span>Manual (Educativo)</span>
      </button>
      
      <button
        onClick={() => onModeChange('tensorflow')}
        disabled={disabled}
        className={`flex items-center gap-2 px-3 py-1.5 rounded border transition-all text-xs ${
          mode === 'tensorflow'
            ? 'border-cyan-400 bg-cyan-400/20 text-cyan-400'
            : 'border-cyan-400/30 text-cyan-400/50 hover:border-cyan-400/50'
        } disabled:opacity-50`}
      >
        <Cpu className="w-4 h-4" />
        <span>TensorFlow.js (Avanzado)</span>
      </button>
    </div>
  );
}
