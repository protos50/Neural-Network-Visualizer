import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const datasetsDir = path.join(process.cwd(), 'public', 'datasets');
    
    // Verificar si el directorio existe
    if (!fs.existsSync(datasetsDir)) {
      return NextResponse.json({ datasets: [] });
    }
    
    // Leer archivos CSV
    const files = fs.readdirSync(datasetsDir)
      .filter(file => file.endsWith('.csv'))
      .map(file => {
        const filePath = path.join(datasetsDir, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.split('\n').filter(l => l.trim() && !l.startsWith('#'));
        
        // Parsear metadata de comentarios
        const comments = content.split('\n').filter(l => l.startsWith('#'));
        let inputCols: string[] = [];
        let outputCols: string[] = [];
        
        comments.forEach(c => {
          if (c.includes('INPUT_COLS:')) {
            inputCols = c.split('INPUT_COLS:')[1].trim().split(',').map(s => s.trim());
          }
          if (c.includes('OUTPUT_COLS:')) {
            outputCols = c.split('OUTPUT_COLS:')[1].trim().split(',').map(s => s.trim());
          }
        });
        
        // Si no hay metadata, usar la primera fila como headers
        const headers = lines[0]?.split(',').map(h => h.trim()) || [];
        
        // Si no hay metadata, asumir última columna como output
        if (inputCols.length === 0 && headers.length > 0) {
          inputCols = headers.slice(0, -1);
          outputCols = [headers[headers.length - 1]];
        }
        
        return {
          name: file.replace('.csv', ''),
          filename: file,
          rows: lines.length - 1, // -1 por header
          inputCols,
          outputCols,
          inputSize: inputCols.length,
          outputSize: outputCols.length,
          headers,
        };
      });
    
    return NextResponse.json({ datasets: files });
  } catch (error) {
    console.error('Error reading datasets:', error);
    return NextResponse.json({ datasets: [], error: 'Error reading datasets' }, { status: 500 });
  }
}
