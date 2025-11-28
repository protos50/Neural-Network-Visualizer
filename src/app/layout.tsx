import type { Metadata } from 'next'
import './globals.css'
import { I18nProvider } from '@/lib/i18n'

export const metadata: Metadata = {
  title: 'Neural Sine Learner',
  description: 'Visualizador de red neuronal aprendiendo la función seno en tiempo real',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es" className="scroll-smooth">
      <body className="min-h-screen overflow-x-hidden">
        {/* CRT scanlines overlay */}
        <div className="crt-overlay" />
        <I18nProvider>
          {children}
        </I18nProvider>
      </body>
    </html>
  )
}
