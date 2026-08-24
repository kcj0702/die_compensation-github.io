import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ADC | Ajin Die Compensation',
  description: '아진산업 3D 스캔 이미지 기반 금형 보정 분석 서비스',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><body>{children}</body></html>;
}
