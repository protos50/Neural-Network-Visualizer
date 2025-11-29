/**
 * Generador de Datasets para el Neural Network Trainer
 * Soporta funciones predefinidas, personalizadas y carga de datasets externos
 */

// Tipos de funciones predefinidas disponibles
export type PredefinedFunction = 
  | 'sin' 
  | 'cos' 
  | 'tan' 
  | 'sin+cos' 
  | 'sin*cos'
  | 'sin²' 
  | 'cos²' 
  | 'sin³'
  | 'tanh'
  | 'sigmoid'
  | 'gaussian'
  | 'sawtooth'
  | 'square'
  | 'triangle'
  | 'custom';

// Descripciones de cada función
export const functionDescriptions: Record<PredefinedFunction, { 
  name: string; 
  formula: string; 
  description: string;
  category: 'trigonometric' | 'activation' | 'wave' | 'custom';
}> = {
  sin: {
    name: 'Seno',
    formula: 'sin(x)',
    description: 'Función seno clásica, ciclo completo en 2π',
    category: 'trigonometric',
  },
  cos: {
    name: 'Coseno',
    formula: 'cos(x)',
    description: 'Función coseno, desfasada 90° del seno',
    category: 'trigonometric',
  },
  tan: {
    name: 'Tangente',
    formula: 'tan(x)',
    description: 'Función tangente, con asíntotas (limitada a ±3)',
    category: 'trigonometric',
  },
  'sin+cos': {
    name: 'Seno + Coseno',
    formula: 'sin(x) + cos(x)',
    description: 'Suma de seno y coseno, amplitud √2',
    category: 'trigonometric',
  },
  'sin*cos': {
    name: 'Seno × Coseno',
    formula: 'sin(x) × cos(x)',
    description: 'Producto de seno y coseno = ½sin(2x)',
    category: 'trigonometric',
  },
  'sin²': {
    name: 'Seno²',
    formula: 'sin²(x)',
    description: 'Seno al cuadrado, siempre positivo',
    category: 'trigonometric',
  },
  'cos²': {
    name: 'Coseno²',
    formula: 'cos²(x)',
    description: 'Coseno al cuadrado, siempre positivo',
    category: 'trigonometric',
  },
  'sin³': {
    name: 'Seno³',
    formula: 'sin³(x)',
    description: 'Seno al cubo, mantiene signo',
    category: 'trigonometric',
  },
  tanh: {
    name: 'Tangente Hiperbólica',
    formula: 'tanh(x)',
    description: 'Función de activación clásica, rango (-1, 1)',
    category: 'activation',
  },
  sigmoid: {
    name: 'Sigmoide',
    formula: '1/(1+e⁻ˣ)',
    description: 'Función logística, rango (0, 1)',
    category: 'activation',
  },
  gaussian: {
    name: 'Gaussiana',
    formula: 'e^(-x²/2)',
    description: 'Curva de campana centrada en 0',
    category: 'activation',
  },
  sawtooth: {
    name: 'Diente de Sierra',
    formula: 'x mod 2π - π',
    description: 'Onda triangular asimétrica',
    category: 'wave',
  },
  square: {
    name: 'Onda Cuadrada',
    formula: 'sign(sin(x))',
    description: 'Onda cuadrada ±1',
    category: 'wave',
  },
  triangle: {
    name: 'Onda Triangular',
    formula: 'asin(sin(x)) × 2/π',
    description: 'Onda triangular suave',
    category: 'wave',
  },
  custom: {
    name: 'Personalizada',
    formula: 'f(x) = ...',
    description: 'Define tu propia función matemática',
    category: 'custom',
  },
};

// Configuración del dataset
export interface DatasetConfig {
  functionType: PredefinedFunction;
  customFormula?: string;
  numPoints: number;
  noiseLevel: number;
  xMin: number;
  xMax: number;
}

// Dataset generado
export interface Dataset {
  X: number[];
  Y: number[];      // Con ruido
  YTrue: number[];  // Sin ruido
  config: DatasetConfig;
}

// Evaluador de funciones predefinidas
function evaluatePredefinedFunction(type: PredefinedFunction, x: number): number {
  switch (type) {
    case 'sin':
      return Math.sin(x);
    case 'cos':
      return Math.cos(x);
    case 'tan':
      // Limitar tangente para evitar valores extremos
      const tanVal = Math.tan(x);
      return Math.max(-3, Math.min(3, tanVal));
    case 'sin+cos':
      return Math.sin(x) + Math.cos(x);
    case 'sin*cos':
      return Math.sin(x) * Math.cos(x);
    case 'sin²':
      return Math.sin(x) ** 2;
    case 'cos²':
      return Math.cos(x) ** 2;
    case 'sin³':
      return Math.sin(x) ** 3;
    case 'tanh':
      return Math.tanh(x);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-x));
    case 'gaussian':
      return Math.exp(-(x * x) / 2);
    case 'sawtooth':
      // Normalizar a [-1, 1]
      const period = 2 * Math.PI;
      const normalized = ((x % period) + period) % period;
      return (normalized / Math.PI) - 1;
    case 'square':
      return Math.sign(Math.sin(x));
    case 'triangle':
      return (2 / Math.PI) * Math.asin(Math.sin(x));
    default:
      return Math.sin(x);
  }
}

// Evaluador de fórmulas personalizadas (con seguridad básica)
function evaluateCustomFormula(formula: string, x: number): number {
  try {
    // Sanitizar y preparar la fórmula
    let safeFormula = formula
      .toLowerCase()
      .replace(/\s+/g, '')
      // Funciones matemáticas
      .replace(/sin/g, 'Math.sin')
      .replace(/cos/g, 'Math.cos')
      .replace(/tan/g, 'Math.tan')
      .replace(/asin/g, 'Math.asin')
      .replace(/acos/g, 'Math.acos')
      .replace(/atan/g, 'Math.atan')
      .replace(/sinh/g, 'Math.sinh')
      .replace(/cosh/g, 'Math.cosh')
      .replace(/tanh/g, 'Math.tanh')
      .replace(/exp/g, 'Math.exp')
      .replace(/log/g, 'Math.log')
      .replace(/log10/g, 'Math.log10')
      .replace(/sqrt/g, 'Math.sqrt')
      .replace(/abs/g, 'Math.abs')
      .replace(/sign/g, 'Math.sign')
      .replace(/floor/g, 'Math.floor')
      .replace(/ceil/g, 'Math.ceil')
      .replace(/round/g, 'Math.round')
      // Constantes
      .replace(/pi/g, 'Math.PI')
      .replace(/e(?![xp])/g, 'Math.E')
      // Operadores de potencia
      .replace(/\^/g, '**');
    
    // Verificar caracteres permitidos
    const allowedPattern = /^[0-9x+\-*/().Math\s,]+$/;
    if (!allowedPattern.test(safeFormula.replace(/Math\.\w+/g, ''))) {
      console.warn('Fórmula contiene caracteres no permitidos');
      return Math.sin(x);
    }
    
    // Evaluar
    const fn = new Function('x', `return ${safeFormula}`);
    const result = fn(x);
    
    // Validar resultado
    if (typeof result !== 'number' || !isFinite(result)) {
      return 0;
    }
    
    // Limitar valores extremos
    return Math.max(-10, Math.min(10, result));
  } catch (error) {
    console.warn('Error evaluando fórmula:', error);
    return Math.sin(x);
  }
}

// Generar dataset
export function generateDataset(config: DatasetConfig): Dataset {
  const { functionType, customFormula, numPoints, noiseLevel, xMin, xMax } = config;
  
  const X: number[] = [];
  const Y: number[] = [];
  const YTrue: number[] = [];
  
  for (let i = 0; i < numPoints; i++) {
    const x = xMin + (xMax - xMin) * (i / (numPoints - 1));
    
    let yTrue: number;
    if (functionType === 'custom' && customFormula) {
      yTrue = evaluateCustomFormula(customFormula, x);
    } else {
      yTrue = evaluatePredefinedFunction(functionType, x);
    }
    
    // Agregar ruido gaussiano
    const noise = noiseLevel * (Math.random() - 0.5) * 2;
    const yNoisy = yTrue + noise;
    
    X.push(x);
    YTrue.push(yTrue);
    Y.push(yNoisy);
  }
  
  return { X, Y, YTrue, config };
}

// Dataset de prueba para animación cíclica
export interface TestDataset {
  points: { x: number; y: number }[];
  currentIndex: number;
  isAnimating: boolean;
}

// Generar puntos de prueba para animación
export function generateTestPoints(
  config: DatasetConfig,
  numCycles: number = 3,
  pointsPerCycle: number = 100
): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = [];
  const range = config.xMax - config.xMin;
  
  for (let cycle = 0; cycle < numCycles; cycle++) {
    for (let i = 0; i < pointsPerCycle; i++) {
      const x = config.xMin + range * (i / pointsPerCycle);
      
      let y: number;
      if (config.functionType === 'custom' && config.customFormula) {
        y = evaluateCustomFormula(config.customFormula, x);
      } else {
        y = evaluatePredefinedFunction(config.functionType, x);
      }
      
      // Agregar ruido ligero para el test
      const noise = config.noiseLevel * 0.5 * (Math.random() - 0.5) * 2;
      
      points.push({ x, y: y + noise });
    }
  }
  
  return points;
}

// Parsear CSV para importar datasets
export function parseCSV(csvText: string): { X: number[]; Y: number[] } | null {
  try {
    const lines = csvText.trim().split('\n');
    const X: number[] = [];
    const Y: number[] = [];
    
    // Detectar si tiene header
    const firstLine = lines[0].split(',');
    const startIndex = isNaN(parseFloat(firstLine[0])) ? 1 : 0;
    
    for (let i = startIndex; i < lines.length; i++) {
      const parts = lines[i].split(',').map(s => parseFloat(s.trim()));
      if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        X.push(parts[0]);
        Y.push(parts[1]);
      }
    }
    
    if (X.length === 0) return null;
    
    return { X, Y };
  } catch (error) {
    console.error('Error parsing CSV:', error);
    return null;
  }
}

// Crear dataset desde CSV importado
export function createDatasetFromCSV(
  X: number[], 
  Y: number[]
): Dataset {
  return {
    X,
    Y,
    YTrue: [...Y], // Sin ruido adicional para datos importados
    config: {
      functionType: 'custom',
      customFormula: 'imported data',
      numPoints: X.length,
      noiseLevel: 0,
      xMin: Math.min(...X),
      xMax: Math.max(...X),
    },
  };
}

// Configuración por defecto
export const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  functionType: 'sin',
  numPoints: 200,
  noiseLevel: 0.2,
  xMin: -6,
  xMax: 6,
};
