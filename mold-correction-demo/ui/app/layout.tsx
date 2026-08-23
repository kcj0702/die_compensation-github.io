import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AJIN Die Insight | 금형 보정 워크벤치',
  description: '3D 스캔 이미지의 라벨 제거, 편차 포인트, 제로라인 결과를 한 번에 확인하는 금형 보정 분석 UI',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><body>{children}</body></html>;
}
