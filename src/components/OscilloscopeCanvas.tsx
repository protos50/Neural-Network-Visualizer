'use client';

import { useRef, useEffect, useCallback } from 'react';

interface OscilloscopeCanvasProps {
  dataPoints: { x: number; y: number }[];
  predictions: { x: number; y: number }[];
  trueSine: { x: number; y: number }[];
  width?: number;
  height?: number;
}

export default function OscilloscopeCanvas({
  dataPoints,
  predictions,
  trueSine,
  width = 800,
  height = 500,
}: OscilloscopeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  // Convertir coordenadas de datos a coordenadas de canvas
  const toCanvasCoords = useCallback((x: number, y: number) => {
    const padding = 40;
    const xRange = { min: -6.5, max: 6.5 };
    const yRange = { min: -1.5, max: 1.5 };
    
    const canvasX = padding + ((x - xRange.min) / (xRange.max - xRange.min)) * (width - 2 * padding);
    const canvasY = height - padding - ((y - yRange.min) / (yRange.max - yRange.min)) * (height - 2 * padding);
    
    return { x: canvasX, y: canvasY };
  }, [width, height]);

  // Dibujar grilla de fondo estilo osciloscopio
  const drawGrid = useCallback((ctx: CanvasRenderingContext2D, time: number) => {
    const padding = 40;
    
    // Fondo
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    // Efecto de resplandor verde suave
    const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, width/2);
    gradient.addColorStop(0, 'rgba(0, 255, 65, 0.03)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Líneas de grilla
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.1)';
    ctx.lineWidth = 1;
    
    // Líneas verticales
    for (let x = padding; x <= width - padding; x += (width - 2 * padding) / 13) {
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }
    
    // Líneas horizontales
    for (let y = padding; y <= height - padding; y += (height - 2 * padding) / 6) {
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }
    
    // Ejes principales
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.4)';
    ctx.lineWidth = 2;
    
    // Eje X (y = 0)
    const zeroY = toCanvasCoords(0, 0).y;
    ctx.beginPath();
    ctx.moveTo(padding, zeroY);
    ctx.lineTo(width - padding, zeroY);
    ctx.stroke();
    
    // Eje Y (x = 0)
    const zeroX = toCanvasCoords(0, 0).x;
    ctx.beginPath();
    ctx.moveTo(zeroX, padding);
    ctx.lineTo(zeroX, height - padding);
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = 'rgba(0, 255, 65, 0.5)';
    ctx.font = '10px monospace';
    ctx.fillText('-6', padding - 5, height - padding + 15);
    ctx.fillText('6', width - padding - 5, height - padding + 15);
    ctx.fillText('1', zeroX + 5, padding + 5);
    ctx.fillText('-1', zeroX + 5, height - padding - 5);
    ctx.fillText('0', zeroX + 5, zeroY - 5);
  }, [width, height, toCanvasCoords]);

  // Dibujar puntos del dataset con ruido
  const drawDataPoints = useCallback((ctx: CanvasRenderingContext2D, time: number) => {
    dataPoints.forEach((point, i) => {
      const { x, y } = toCanvasCoords(point.x, point.y);
      
      // Jitter sutil para efecto de "vida"
      const jitterX = Math.sin(time * 0.01 + i * 0.5) * 0.5;
      const jitterY = Math.cos(time * 0.01 + i * 0.3) * 0.5;
      
      // Glow
      const grd = ctx.createRadialGradient(x + jitterX, y + jitterY, 0, x + jitterX, y + jitterY, 8);
      grd.addColorStop(0, 'rgba(0, 255, 65, 0.8)');
      grd.addColorStop(0.5, 'rgba(0, 255, 65, 0.2)');
      grd.addColorStop(1, 'rgba(0, 255, 65, 0)');
      
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(x + jitterX, y + jitterY, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Centro brillante
      ctx.fillStyle = 'rgba(200, 255, 200, 0.9)';
      ctx.beginPath();
      ctx.arc(x + jitterX, y + jitterY, 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [dataPoints, toCanvasCoords]);

  // Helper para dibujar línea con detección de saltos (evita línea fantasma al reiniciar ciclo)
  const drawLineWithGaps = useCallback((
    ctx: CanvasRenderingContext2D, 
    points: { x: number; y: number }[],
    jumpThreshold: number = 2 // Si el salto en X es mayor a esto, empezar nuevo segmento
  ) => {
    if (points.length < 2) return;
    
    const first = toCanvasCoords(points[0].x, points[0].y);
    ctx.moveTo(first.x, first.y);
    
    for (let i = 1; i < points.length; i++) {
      const prevX = points[i - 1].x;
      const currX = points[i].x;
      const { x, y } = toCanvasCoords(currX, points[i].y);
      
      // Detectar salto grande en X (vuelta al inicio del ciclo)
      if (Math.abs(currX - prevX) > jumpThreshold) {
        ctx.moveTo(x, y); // Empezar nuevo segmento
      } else {
        ctx.lineTo(x, y);
      }
    }
  }, [toCanvasCoords]);

  // Dibujar la curva seno verdadera (más visible)
  const drawTrueSine = useCallback((ctx: CanvasRenderingContext2D) => {
    if (trueSine.length < 2) return;
    
    // Glow exterior amarillo
    ctx.strokeStyle = 'rgba(255, 200, 50, 0.15)';
    ctx.lineWidth = 6;
    ctx.setLineDash([]);
    
    ctx.beginPath();
    drawLineWithGaps(ctx, trueSine);
    ctx.stroke();
    
    // Línea principal amarilla punteada
    ctx.strokeStyle = 'rgba(255, 200, 50, 0.7)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    
    ctx.beginPath();
    drawLineWithGaps(ctx, trueSine);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [trueSine, drawLineWithGaps]);

  // Dibujar la predicción de la red (láser verde)
  const drawPredictions = useCallback((ctx: CanvasRenderingContext2D, time: number) => {
    if (predictions.length < 2) return;
    
    // Capa de glow exterior
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.15)';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    ctx.beginPath();
    drawLineWithGaps(ctx, predictions);
    ctx.stroke();
    
    // Capa de glow medio
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.4)';
    ctx.lineWidth = 6;
    
    ctx.beginPath();
    drawLineWithGaps(ctx, predictions);
    ctx.stroke();
    
    // Capa central brillante
    ctx.strokeStyle = 'rgba(150, 255, 150, 0.9)';
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    drawLineWithGaps(ctx, predictions);
    ctx.stroke();
    
    // Centro blanco
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 1;
    
    ctx.beginPath();
    drawLineWithGaps(ctx, predictions);
    ctx.stroke();
  }, [predictions, drawLineWithGaps]);

  // Dibujar leyenda
  const drawLegend = useCallback((ctx: CanvasRenderingContext2D) => {
    const padding = 40;
    const legendX = width - 180;
    const legendY = padding + 10;
    
    // Fondo de la leyenda
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(legendX - 10, legendY - 5, 160, 55);
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX - 10, legendY - 5, 160, 55);
    
    // Predicción de la red (verde)
    ctx.fillStyle = 'rgba(0, 255, 65, 0.9)';
    ctx.fillRect(legendX, legendY + 3, 20, 3);
    ctx.fillStyle = 'rgba(0, 255, 65, 0.8)';
    ctx.font = '10px monospace';
    ctx.fillText('Predicción (red)', legendX + 30, legendY + 10);
    
    // Sin(x) real (amarillo punteado)
    ctx.strokeStyle = 'rgba(255, 200, 50, 0.7)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + 23);
    ctx.lineTo(legendX + 20, legendY + 23);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255, 200, 50, 0.8)';
    ctx.fillText('sin(x) real', legendX + 30, legendY + 27);
    
    // Datos con ruido (punto verde)
    ctx.fillStyle = 'rgba(0, 255, 65, 0.8)';
    ctx.beginPath();
    ctx.arc(legendX + 10, legendY + 40, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText('Datos (con ruido)', legendX + 30, legendY + 43);
  }, [width]);

  // Loop de animación
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      timeRef.current += 1;
      
      drawGrid(ctx, timeRef.current);
      drawTrueSine(ctx);
      drawDataPoints(ctx, timeRef.current);
      drawPredictions(ctx, timeRef.current);
      drawLegend(ctx);
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [drawGrid, drawDataPoints, drawTrueSine, drawPredictions, drawLegend]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg border border-crt-green/30"
      style={{
        boxShadow: '0 0 20px rgba(0, 255, 65, 0.2), inset 0 0 60px rgba(0, 0, 0, 0.5)',
      }}
    />
  );
}
