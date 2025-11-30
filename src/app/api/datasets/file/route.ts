import { NextResponse, type NextRequest } from 'next/server';
import fs from 'fs';
import path from 'path';

// Ruta dinámica para leer el contenido de un CSV de datasets en tiempo real
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(req: NextRequest) {
  try {
    const url = new URL(req.url);
    const name = url.searchParams.get('name') || url.searchParams.get('filename');

    if (!name) {
      return new NextResponse('Missing name parameter', { status: 400 });
    }

    // Sanitizar: evitar path traversal, solo nombre de archivo
    const safeName = path.basename(name);

    if (!safeName.endsWith('.csv')) {
      return new NextResponse('Invalid file type', { status: 400 });
    }

    const filePath = path.join(process.cwd(), 'public', 'datasets', safeName);

    if (!fs.existsSync(filePath)) {
      return new NextResponse('Dataset not found', { status: 404 });
    }

    const content = await fs.promises.readFile(filePath, 'utf-8');

    return new NextResponse(content, {
      status: 200,
      headers: {
        'Content-Type': 'text/csv; charset=utf-8',
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    console.error('Error reading dataset file:', error);
    return new NextResponse('Error reading dataset file', { status: 500 });
  }
}
