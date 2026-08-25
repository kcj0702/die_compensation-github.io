'use client';

/* Blob/data URLs are local engine outputs and cannot use Next's remote image loader. */
/* eslint-disable @next/next/no-img-element */

import {
  Activity, ArrowLeft, ArrowRight, BarChart3, Check, ChevronDown, ChevronRight,
  CircleHelp, Eye, EyeOff, File, Folder, FolderOpen, Gauge, Grid2X2, Image as ImageIcon,
  Layers3, ListFilter, Maximize2, MoveRight, PanelLeftClose, Play, Settings2,
  ShieldCheck, Sparkles, UploadCloud, X, ZoomIn, ZoomOut, Box,
} from 'lucide-react';
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react';

import { CadViewer, type CadMesh } from './cad-viewer';

const API_BASE = 'http://127.0.0.1:8000';

type View = 'workspace' | 'results' | 'service' | 'cad';
type Engine = 'label' | 'deviation' | 'zero';
type ScanStatus = 'ready' | 'analyzing' | 'done' | 'error';
type PointResult = { id: string; xPx: number; yPx: number; x: number; y: number; value: number; labelColor: string; confidence: string };
type ZeroAnchor = { anchor_id: number; x: number; y: number; boundary_arclen: number; source?: string; kind?: 'point' | 'zone'; strength?: number };
type ValleyLine = { id: string; anchorStartId: number | null; anchorEndId: number | null; points: [number, number][]; length_px: number; mean_abs_deviation: number; source: 'ai' | 'manual' };
type AdvanceLine = { points: [number, number][]; warnings: string[]; confidence: 'high' | 'low' };
type ReferenceLine = { kind: 'line' | 'areas'; points: [number, number][]; contours: [number, number][][]; partNo: string; sourceSheet: string; mirrored: boolean };
type ZeroPointCluster = { cluster_id: number; loop: string; kind: 'point' | 'zone'; center: [number, number]; members: [number, number][]; contour: [number, number][]; strength: number; span: number };
type LabelZeroLine = { points: [number, number][]; length_px: number; mean_abs_deviation: number };
type ZeroLineCandidate = { rank: number; anchor_start_id: number; anchor_end_id: number; points: [number, number][]; length_px: number; mean_abs_deviation: number; separation: number; balance: number; score: number };
type AnalysisResult = {
  analysisId: string | null;
  source: { name: string; width: number; height: number };
  cleanImage: string | null;
  zeroOverlay: string | null;
  zeroMask: string | null;
  zeroAnchors: ZeroAnchor[];
  advanceLine: AdvanceLine | null;
  zeroLineCandidates: ZeroLineCandidate[];
  zeroPointClusters: ZeroPointCluster[];
  labelZeroLine: LabelZeroLine | null;
  referenceLine: ReferenceLine | null;
  points: PointResult[];
  stats: {
    labelsRemoved: number;
    pointsDetected: number;
    detectedCandidates?: number;
    validCandidates?: number;
    qwenReads: number;
    qwenUnread?: number;
    zeroRegions: number;
    zeroRatio: number;
    zeroTolerance: number | null;
  };
  warnings: string[];
  warningsByEngine?: Partial<Record<Engine, string[]>>;
  errors: Partial<Record<Engine, string>>;
  valueMode: string;
};
type ScanItem = { id: string; name: string; partNo: string; size: string; url: string; file: File; status: ScanStatus; tone: number; result?: AnalysisResult; error?: string };
type FolderEntry = { name: string; path: string; isDirectory: boolean; size: number | null; modified: string };

const engineMeta: Record<Engine, { name: string; short: string; color: string }> = {
  label: { name: '라벨 제거 · 복원', short: 'label_removal', color: '#7058e8' },
  deviation: { name: '편차 포인트 추출', short: 'deviation_extraction', color: '#ee6b3c' },
  zero: { name: '제로라인 검출', short: 'zero_line_detection', color: '#17a58b' },
};

function formatBytes(value: number | null) {
  if (value == null) return '';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function Heatmap({ imageUrl, width, height, children, lightBackground = false }: { imageUrl?: string | null; width: number; height: number; children?: React.ReactNode; lightBackground?: boolean }) {
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number } | null>(null);
  const clampPan = (next: { x: number; y: number }, nextScale: number) => {
    const bounds = viewportRef.current?.getBoundingClientRect();
    if (!bounds || nextScale <= 1) return { x: 0, y: 0 };
    const maxX = bounds.width * (nextScale - 1) / 2;
    const maxY = bounds.height * (nextScale - 1) / 2;
    return { x: Math.max(-maxX, Math.min(maxX, next.x)), y: Math.max(-maxY, Math.min(maxY, next.y)) };
  };
  const setZoom = (nextScale: number) => {
    const bounded = Math.max(1, Math.min(4, nextScale));
    setScale(bounded);
    setPan((current) => clampPan(current, bounded));
  };
  const resetView = () => { setScale(1); setPan({ x: 0, y: 0 }); };

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || !imageUrl) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const amount = event.deltaY < 0 ? 0.25 : -0.25;
      setScale((current) => {
        const next = Math.max(1, Math.min(4, current + amount));
        setPan((currentPan) => clampPan(currentPan, next));
        return next;
      });
    };
    viewport.addEventListener('wheel', handleWheel, { passive: false });
    return () => viewport.removeEventListener('wheel', handleWheel);
  }, [imageUrl]);

  return <div
    ref={viewportRef}
    className={`heatmap heatmap--actual ${lightBackground ? 'heatmap--light' : ''} ${children ? 'heatmap--annotated' : ''} ${scale > 1 ? 'heatmap--zoomed' : ''} ${dragging ? 'heatmap--dragging' : ''}`}
    onDoubleClick={resetView}
    onPointerDown={(event) => {
      if (scale <= 1 || (event.target as Element).closest('.measure-point, .anchor-point, .zoom-controls')) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
      setDragging(true);
    }}
    onPointerMove={(event) => {
      const start = dragRef.current;
      if (!start || start.pointerId !== event.pointerId) return;
      setPan(clampPan({ x: start.panX + event.clientX - start.x, y: start.panY + event.clientY - start.y }, scale));
    }}
    onPointerUp={(event) => {
      if (dragRef.current?.pointerId !== event.pointerId) return;
      dragRef.current = null; setDragging(false); event.currentTarget.releasePointerCapture(event.pointerId);
    }}
    onPointerCancel={() => { dragRef.current = null; setDragging(false); }}
  >
    {imageUrl && <div className="zoom-controls" aria-label="이미지 확대 및 이동 도구">
      <button type="button" onClick={() => setZoom(scale - 0.25)} disabled={scale <= 1} aria-label="축소" title="축소"><ZoomOut size={15} /></button>
      <span aria-live="polite">{Math.round(scale * 100)}%</span>
      <button type="button" onClick={() => setZoom(scale + 0.25)} disabled={scale >= 4} aria-label="확대" title="확대"><ZoomIn size={15} /></button>
      <button type="button" onClick={resetView} disabled={scale === 1 && pan.x === 0 && pan.y === 0} aria-label="화면 맞춤" title="화면 맞춤"><Maximize2 size={15} /></button>
    </div>}
    {imageUrl ? <div className="heatmap__media" style={{ aspectRatio: `${width} / ${height}` }}>
      <div className="heatmap__transform" style={{ transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${scale})` }}>
        <img src={imageUrl} alt="엔진이 처리한 3D 스캔 편차 이미지" />
        {children}
      </div>
    </div> : <div className="heatmap__empty"><ImageIcon size={34} /><span>분석 결과 이미지가 없습니다.</span></div>}
  </div>;
}

function CorrectionPoints({ coefficient, points, labels = true, visibleLabelIds, onLabelToggle }: { coefficient: number; points: PointResult[]; labels?: boolean; visibleLabelIds?: Set<string>; onLabelToggle?: (id: string) => void }) {
  const labelHeight = 17;
  const formatCorrection = (point: PointResult) => {
    const correction = -(point.value * coefficient);
    return `${correction > 0 ? '+' : ''}${correction.toFixed(1)}`;
  };
  const getLabelWidth = (point: PointResult) => Math.max(24, formatCorrection(point).length * 5.2 + 8);
  const layerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: string; x: number; y: number; clientX: number; clientY: number; moved: boolean } | null>(null);
  const ignoreClickRef = useRef(false);
  const layoutKeyRef = useRef('');
  const [layerSize, setLayerSize] = useState({ width: 0, height: 0 });
  const [labelPositions, setLabelPositions] = useState<Record<string, { x: number; y: number }>>({});
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const updateSize = () => setLayerSize({ width: layer.clientWidth, height: layer.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(layer);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!layerSize.width) return;
    const layoutKey = `${labelHeight}:${points.map((point) => `${point.id}:${formatCorrection(point).length}`).join('|')}`;
    const resetLayout = layoutKeyRef.current !== layoutKey;
    layoutKeyRef.current = layoutKey;
    setLabelPositions((current) => {
      const next = resetLayout ? {} : { ...current };
      const centerX = points.length > 1 ? points.reduce((sum, point) => sum + layerSize.width * point.x / 100, 0) / points.length : layerSize.width / 2;
      const centerY = points.length > 1 ? points.reduce((sum, point) => sum + layerSize.height * point.y / 100, 0) / points.length : layerSize.height / 2;
      points.forEach((point) => {
        if (next[point.id]) return;
        const labelWidth = getLabelWidth(point);
        const pointX = layerSize.width * point.x / 100;
        const pointY = layerSize.height * point.y / 100;
        let dx = pointX - centerX;
        let dy = pointY - centerY;
        if (Math.abs(dx) + Math.abs(dy) < 1) { dx = 1; dy = 0; }
        const length = Math.hypot(dx, dy);
        const unitX = dx / length;
        const unitY = dy / length;
        const leaderEndX = pointX + unitX * 20;
        const leaderEndY = pointY + unitY * 20;
        next[point.id] = {
          x: unitX >= 0 ? leaderEndX : leaderEndX - labelWidth,
          y: unitY >= 0 ? leaderEndY : leaderEndY - labelHeight,
        };
      });
      return next;
    });
  }, [layerSize, points, coefficient, labelHeight]);
  const beginLabelDrag = (event: React.PointerEvent<HTMLSpanElement>, id: string) => {
    const position = labelPositions[id];
    if (!position) return;
    event.preventDefault(); event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { id, x: position.x, y: position.y, clientX: event.clientX, clientY: event.clientY, moved: false };
  };
  const moveLabel = (event: React.PointerEvent<HTMLSpanElement>) => {
    const drag = dragRef.current;
    const layer = layerRef.current;
    if (!drag || drag.id !== event.currentTarget.dataset.pointId || !layer) return;
    const scale = layer.getBoundingClientRect().width / Math.max(layer.clientWidth, 1);
    const movedX = drag.x + (event.clientX - drag.clientX) / scale;
    drag.moved ||= Math.hypot(event.clientX - drag.clientX, event.clientY - drag.clientY) > 3;
    setLabelPositions((current) => ({ ...current, [drag.id]: {
      x: Math.max(-140, Math.min(layer.clientWidth + 62, movedX)),
      y: Math.max(-8, Math.min(layer.clientHeight - 24, drag.y + (event.clientY - drag.clientY) / scale)),
    }}));
  };
  const endLabelDrag = (event: React.PointerEvent<HTMLSpanElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.id !== event.currentTarget.dataset.pointId) return;
    dragRef.current = null; ignoreClickRef.current = drag.moved;
    if (drag.moved) window.setTimeout(() => { ignoreClickRef.current = false; }, 0);
    event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return <div className="point-layer" ref={layerRef}>
    <svg className="point-leaders" aria-hidden="true" viewBox={`0 0 ${layerSize.width} ${layerSize.height}`}>{points.map((point) => {
       const position = labelPositions[point.id]; const visible = !visibleLabelIds || visibleLabelIds.has(point.id);
       if (!position || !visible) return null;
       const labelWidth = getLabelWidth(point);
       const x1 = layerSize.width * point.x / 100; const y1 = layerSize.height * point.y / 100;
      let x2 = Math.max(position.x, Math.min(position.x + labelWidth, x1));
      let y2 = Math.max(position.y, Math.min(position.y + labelHeight, y1));
      if (x2 === x1 && y2 === y1) {
        const distances = [
          { value: x1 - position.x, edge: 'left' },
          { value: position.x + labelWidth - x1, edge: 'right' },
          { value: y1 - position.y, edge: 'top' },
          { value: position.y + labelHeight - y1, edge: 'bottom' },
        ].sort((a, b) => a.value - b.value);
        if (distances[0].edge === 'left') x2 = position.x;
        if (distances[0].edge === 'right') x2 = position.x + labelWidth;
        if (distances[0].edge === 'top') y2 = position.y;
        if (distances[0].edge === 'bottom') y2 = position.y + labelHeight;
      }
      return <line key={point.id} x1={x1} y1={y1} x2={x2} y2={y2} />;
    })}</svg>{points.map((point) => {
    const correction = -(point.value * coefficient);
    const labelVisible = !visibleLabelIds || visibleLabelIds.has(point.id);
    const position = labelPositions[point.id];
    return <button type="button" className={`measure-point ${correction >= 0 ? 'measure-point--plus' : 'measure-point--minus'} ${onLabelToggle ? 'measure-point--interactive' : ''}`} style={{ left: `${point.x}%`, top: `${point.y}%` }} key={point.id} onClick={() => { if (ignoreClickRef.current) { ignoreClickRef.current = false; return; } onLabelToggle?.(point.id); }} aria-label={`${point.id} 라벨 ${labelVisible ? '숨기기' : '표시하기'}`} aria-pressed={labelVisible} title={`${point.id} 편차 ${point.value > 0 ? '+' : ''}${point.value.toFixed(3)} · 포인트 또는 라벨 클릭으로 표시 전환`}>
      <span className="measure-point__dot" />
      {labels && labelVisible && position && <span className="measure-point__label" data-point-id={point.id} style={{ left: `${position.x - layerSize.width * point.x / 100}px`, top: `${position.y - layerSize.height * point.y / 100}px` }} onPointerDown={(event) => beginLabelDrag(event, point.id)} onPointerMove={moveLabel} onPointerUp={endLabelDrag} onPointerCancel={endLabelDrag}>{formatCorrection(point)}</span>}
    </button>;
  })}</div>;
}

function AnchorPicker({ anchors, width, height, selectedIds, onToggle }: { anchors: ZeroAnchor[]; width: number; height: number; selectedIds: number[]; onToggle: (id: number) => void }) {
  return <div className="point-layer anchor-layer">{anchors.map((anchor) => {
    const selected = selectedIds.includes(anchor.anchor_id);
    const order = selectedIds.indexOf(anchor.anchor_id);
    return <button type="button" key={anchor.anchor_id} className={`anchor-point ${selected ? 'anchor-point--selected' : ''}`}
      style={{ left: `${(anchor.x / width) * 100}%`, top: `${(anchor.y / height) * 100}%` }}
      onClick={() => onToggle(anchor.anchor_id)}
      aria-pressed={selected}
      title={`앵커 ${anchor.anchor_id} · 클릭해서 ${selected ? '선택 해제' : '제로라인 시작/끝점으로 선택'}`}>
      <span className="anchor-point__ring" />
      {selected && <span className="anchor-point__order">{order + 1}</span>}
    </button>;
  })}</div>;
}

function ZeroZoneOverlay({ clusters, width, height }: { clusters: ZeroPointCluster[]; width: number; height: number }) {
  // 백엔드가 0포인트 군집을 전부 면으로 넓혀서 준다 — 보정시트가 제로를
  // 선 하나가 아니라 여러 구간으로 표기하는 부품이 있어서다(실측:
  // 점 2개를 이은 선만 내면 정답 커버리지가 67XX6 19.8%, 64XX2 5.3%
  // 였는데 구간으로 내면 5.6% / 1.7% 로 좋아진다).
  const areas = clusters.filter((c) => c.contour.length >= 3);
  if (!areas.length) return null;
  // 히트맵 자체가 무지개색이라 옅은 채움만으론 안 보인다는 피드백 —
  // 흰 테두리로 후광을 깔아 어떤 배경색 위에서도 도드라지게 한다.
  // 글자는 넣지 않는다: 구간이 7~10개씩 나오는 부품에서 라벨을 전부
  // 찍으면 화면이 글자로 뒤덮인다.
  const strokeWidth = Math.max(width, height) * 0.004;
  return <svg className="zero-zone-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {areas.map((cluster) => {
      const points = cluster.contour.map(([x, y]) => `${x},${y}`).join(' ');
      return <g key={cluster.cluster_id}>
        <polygon points={points} fill="none" stroke="#ffffff" strokeWidth={strokeWidth * 2.4} strokeLinejoin="round" opacity={0.95} />
        <polygon points={points} fill="rgba(255,45,45,0.55)" stroke="#b30f0f" strokeWidth={strokeWidth} strokeLinejoin="round" />
      </g>;
    })}
  </svg>;
}

function ValleyLineOverlay({ lines, width, height }: { lines: ValleyLine[]; width: number; height: number }) {
  if (!lines.length) return null;
  const strokeWidth = Math.max(width, height) * 0.0026;
  return <svg className="valley-line-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {lines.map((line) => {
      // sheet-reference-* 는 정답지 비교용 오버레이라 실제 검출선과
      // 헷갈리지 않도록 파란 점선으로 따로 그린다.
      const isSheetCompare = line.id.startsWith('sheet-reference');
      return <polyline key={line.id} points={line.points.map(([x, y]) => `${x},${y}`).join(' ')} fill="none"
        stroke={isSheetCompare ? '#2563eb' : '#e0303f'} strokeWidth={strokeWidth}
        strokeDasharray={isSheetCompare ? `${strokeWidth * 2.5} ${strokeWidth * 2}` : undefined}
        strokeLinecap="round" strokeLinejoin="round" />;
    })}
  </svg>;
}

function Sidebar({ view, setView, collapsed, setCollapsed, hasResult }: { view: View; setView: (view: View) => void; collapsed: boolean; setCollapsed: (value: boolean) => void; hasResult: boolean }) {
  const items = [
    { id: 'workspace' as const, label: '분석 작업실', icon: Grid2X2 },
    { id: 'results' as const, label: '엔진 결과', icon: BarChart3 },
    { id: 'service' as const, label: 'ADC 보정 시트', icon: Layers3 },
    { id: 'cad' as const, label: '3D 데이터', icon: Box },
  ];
  return <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
    <div className="brand"><img className="brand__logo" src="/ajin-industrial-logo.png" alt="아진산업" /><div className="brand__copy"><strong>ADC</strong><span>Ajin Die Compensation</span></div></div>
    <nav className="sidebar__nav" aria-label="주 메뉴"><span className="sidebar__eyebrow">ADC WORKSPACE</span>{items.map((item) => { const Icon = item.icon; const disabled = item.id !== 'workspace' && item.id !== 'cad' && !hasResult; return <button key={item.id} disabled={disabled} onClick={() => !disabled && setView(item.id)} className={view === item.id ? 'active' : ''}><Icon size={19} /><span>{item.label}</span></button>; })}</nav>
    <div className="sidebar__guide"><CircleHelp size={19} /><div><b>실제 엔진 연결</b><span>모든 처리는 이 PC 안에서 실행</span></div><ChevronRight size={16} /></div>
    <button className="sidebar__collapse" onClick={() => setCollapsed(!collapsed)} aria-label="사이드바 접기"><PanelLeftClose size={18} /><span>메뉴 접기</span></button>
  </aside>;
}

function Header({ scans, activeId, setActiveId }: { scans: ScanItem[]; activeId?: string; setActiveId: (id: string) => void }) {
  return <header className="topbar"><div><span className="topbar__context">AJIN INDUSTRIAL · DIE ENGINEERING</span><h1>ADC <small>Ajin Die Compensation</small></h1></div><div className="topbar__actions">
    <label className="item-select"><span>현재 품번</span><select value={activeId || ''} disabled={!scans.length} onChange={(e) => setActiveId(e.target.value)}><option value="">등록된 이미지 없음</option>{scans.map((scan) => <option value={scan.id} key={scan.id}>{scan.partNo} · {scan.name}</option>)}</select></label>
    <button className="icon-button" aria-label="설정"><Settings2 size={19} /></button><div className="profile"><span>KJ</span><div><b>금형생산팀</b><small>관리자</small></div></div>
  </div></header>;
}

function Workspace({ scans, setScans, onOpenResults, backendOnline }: { scans: ScanItem[]; setScans: React.Dispatch<React.SetStateAction<ScanItem[]>>; onOpenResults: (id: string) => void; backendOnline: boolean | null }) {
  const [dragging, setDragging] = useState(false);
  const analyzingCount = scans.filter((scan) => scan.status === 'analyzing').length;
  const addFiles = (files: FileList | File[]) => {
    const accepted = Array.from(files).filter((file) => file.type.startsWith('image/'));
    const next = accepted.map((file, index): ScanItem => ({
      id: `${file.name}-${file.lastModified}-${crypto.randomUUID()}`,
      name: file.name,
      partNo: file.name.match(/[0-9]{2}[A-Z0-9]{2,4}/)?.[0] || `NEW-${String(scans.length + index + 1).padStart(2, '0')}`,
      size: `${(file.size / 1024 / 1024).toFixed(1)} MB`, url: URL.createObjectURL(file), file,
      status: 'ready', tone: (scans.length + index) % 3,
    }));
    setScans((current) => [...current, ...next]);
  };
  const analyzeAll = async () => {
    const targets = scans.filter((scan) => scan.status === 'ready' || scan.status === 'error');
    for (const target of targets) {
      setScans((current) => current.map((scan) => scan.id === target.id ? { ...scan, status: 'analyzing', error: undefined } : scan));
      try {
        const form = new FormData(); form.append('file', target.file, target.name);
        const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '분석 중 오류가 발생했습니다.');
        setScans((current) => current.map((scan) => scan.id === target.id ? { ...scan, status: 'done', result: data } : scan));
      } catch (error) {
        const message = error instanceof Error ? error.message : '분석 서버에 연결할 수 없습니다.';
        setScans((current) => current.map((scan) => scan.id === target.id ? { ...scan, status: 'error', error: message } : scan));
      }
    }
  };
  const removeScan = (id: string) => setScans((current) => { const target = current.find((item) => item.id === id); if (target) URL.revokeObjectURL(target.url); return current.filter((item) => item.id !== id); });
  return <section className="page page--workspace">
    <div className="page-heading"><div><span className="kicker"><Sparkles size={14} /> 실제 로컬 엔진</span><h2>스캔 이미지를 한 번에 분석하세요</h2><p>업로드한 이미지는 이 PC의 세 엔진으로 처리되며 외부 서버로 전송되지 않습니다.</p></div><div className="step-pills"><span className="done"><Check size={14} /> 1. 이미지 등록</span><span className={analyzingCount ? 'active' : ''}>2. 엔진 분석</span><span>3. 보정 시트</span></div></div>
    <div className="workspace-grid"><div className="upload-panel card"><div className="card-title"><div><h3>스캔 이미지 등록</h3><p>PNG, JPG, WEBP · 여러 파일 동시 선택 가능</p></div><span className="count-chip">{scans.length}개 등록</span></div>
      <label className={`dropzone ${dragging ? 'dropzone--active' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e: DragEvent<HTMLLabelElement>) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}><input type="file" multiple accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff" onChange={(e: ChangeEvent<HTMLInputElement>) => e.target.files && addFiles(e.target.files)} /><span className="dropzone__icon"><UploadCloud size={29} /></span><b>스캔 이미지를 여기에 놓으세요</b><span>또는 클릭하여 파일 선택</span><em>여러 품번의 이미지를 동시에 올릴 수 있습니다</em></label>
      <div className="file-list"><div className="file-list__head"><span>등록된 이미지</span><button><ListFilter size={15} /> 상태순</button></div>{!scans.length && <div className="empty-file-list">아직 등록된 이미지가 없습니다.</div>}{scans.map((scan) => <div className="file-row" key={scan.id}><div className={`file-thumb tone-${scan.tone}`}><img src={scan.url} alt="" /></div><div className="file-row__name"><b>{scan.name}</b><span>{scan.partNo} · {scan.error || scan.size}</span></div><span className={`status status--${scan.status}`}>{scan.status === 'done' ? <><Check size={13} /> 분석 완료</> : scan.status === 'analyzing' ? <><Activity size={13} /> 분석 중</> : scan.status === 'error' ? '오류' : '대기'}</span>{scan.status === 'done' ? <button className="text-button" onClick={() => onOpenResults(scan.id)}>결과 보기 <ArrowRight size={14} /></button> : <button className="icon-button icon-button--small" onClick={() => removeScan(scan.id)} aria-label={`${scan.name} 삭제`}><X size={15} /></button>}</div>)}</div>
      <button className="primary-button primary-button--wide" onClick={analyzeAll} disabled={!backendOnline || analyzingCount > 0 || !scans.some((scan) => scan.status === 'ready' || scan.status === 'error')}><Play size={17} fill="currentColor" /> {analyzingCount ? `${analyzingCount}개 이미지 분석 중` : backendOnline === false ? '로컬 엔진 서버 연결 필요' : '대기 이미지 전체 분석 시작'}<ArrowRight size={18} /></button>
    </div><aside className="engine-panel"><div className="card engine-overview"><div className="card-title"><div><h3>분석 엔진</h3><p>실제 연결 상태</p></div><span className={`live-dot ${backendOnline === false ? 'offline' : ''}`}>{backendOnline == null ? '확인 중' : backendOnline ? '연결됨' : '연결 안 됨'}</span></div>{(Object.keys(engineMeta) as Engine[]).map((key, index) => { const meta = engineMeta[key]; return <div className="engine-row" key={key}><span className="engine-row__number" style={{ background: `${meta.color}16`, color: meta.color }}>0{index + 1}</span><div><b>{meta.name}</b><span>{meta.short}</span></div>{backendOnline ? <ShieldCheck size={18} color={meta.color} /> : <X size={18} color="#a2aab4" />}</div>; })}</div><div className="tip-card"><span><Gauge size={20} /></span><div><b>편차값 판독 방식</b><p>Qwen2.5-VL-3B를 RTX GPU에서 실행하며, 모델은 인터넷 없이 로컬 파일만 사용합니다.</p></div></div></aside></div>
  </section>;
}

function engineSummary(engine: Engine, result: AnalysisResult) {
  if (result.errors[engine]) return { stat: '실패', detail: result.errors[engine] || '엔진 오류' };
  if (engine === 'label') return { stat: `${result.stats.labelsRemoved}개`, detail: '검출된 라벨 제거 및 주변 색상 복원 완료' };
  if (engine === 'deviation') {
    const detected = result.stats.detectedCandidates ?? result.stats.pointsDetected;
    const connected = result.stats.validCandidates ?? result.stats.pointsDetected;
    return {
      stat: `${result.stats.pointsDetected}개`,
      detail: `라벨 후보 ${detected}개 · 스캔 연결 ${connected}개 · Qwen 실제 판독 ${result.stats.qwenReads}개`,
    };
  }
  return { stat: `${result.stats.zeroRegions}개`, detail: `부품 면적의 ${(result.stats.zeroRatio * 100).toFixed(1)}% · 실제 검출 결과` };
}

function Results({ scan, onService, hiddenPointIds, onPointToggle, onAllPointsToggle, valleyLines, setValleyLines }: { scan: ScanItem; onService: () => void; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; onAllPointsToggle: (visible: boolean) => void; valleyLines: ValleyLine[]; setValleyLines: (updater: ValleyLine[] | ((current: ValleyLine[]) => ValleyLine[])) => void }) {
  const [engine, setEngine] = useState<Engine>('label');
  const result = scan.result!; const meta = engineMeta[engine]; const summary = engineSummary(engine, result);
  const engineWarnings = result.warningsByEngine?.[engine] ?? (engine === 'zero' ? result.warnings : []);
  const visibleLabelIds = new Set(result.points.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id));
  const image = engine === 'zero' ? result.zeroOverlay : result.cleanImage || scan.url;
  const toggleLabel = onPointToggle;
  const allLabelsVisible = result.points.length > 0 && visibleLabelIds.size === result.points.length;

  const zeroAnchors = result.zeroAnchors || [];
  // 백엔드가 군집을 전부 면으로 넓혀 보내므로 폴리곤이 있는 것만 거른다
  // (kind 로 거르면 확장 전 데이터가 섞였을 때 조용히 빠진다).
  const zeroAreaClusters = (result.zeroPointClusters || []).filter((c) => c.contour.length >= 3);
  const [selectedAnchors, setSelectedAnchors] = useState<number[]>([]);
  const [valleyStatus, setValleyStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [valleyError, setValleyError] = useState<string | null>(null);
  useEffect(() => { setSelectedAnchors([]); setValleyStatus('idle'); setValleyError(null); }, [scan.id]);
  const toggleAnchor = (id: number) => setSelectedAnchors((current) => {
    if (current.includes(id)) return current.filter((value) => value !== id);
    if (current.length >= 2) return [id];
    return [...current, id];
  });
  useEffect(() => {
    if (selectedAnchors.length !== 2) return;
    if (!result.analysisId) { setValleyStatus('error'); setValleyError('분석 결과가 만료됐습니다. 이미지를 다시 분석하세요.'); return; }
    let cancelled = false;
    setValleyStatus('loading'); setValleyError(null);
    fetch(`${API_BASE}/api/zero-valley-line`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysisId: result.analysisId, anchorIds: selectedAnchors }),
    })
      .then((response) => response.json().then((data) => ({ ok: response.ok, data })))
      .then(({ ok, data }) => {
        if (cancelled) return;
        if (!ok) { setValleyStatus('error'); setValleyError(data.error || '선을 잇지 못했습니다.'); setSelectedAnchors([]); return; }
        const line = data.line;
        // 사람이 직접 이은 선은 AI 1차 제안을 대체한다 — 신뢰도 낮은 AI
        // 추천선과 사람이 고친 선이 동시에 겹쳐 그려져 화면이 지저분해지는
        // 걸 막는다(AI 추천선이 틀렸을 때 특히 두드러졌던 문제).
        setValleyLines((current) => [...current.filter((item) => item.source !== 'ai'), {
          id: `${line.anchor_start_id}-${line.anchor_end_id}-${current.length}`,
          anchorStartId: line.anchor_start_id, anchorEndId: line.anchor_end_id,
          points: line.points, length_px: line.length_px, mean_abs_deviation: line.mean_abs_deviation,
          source: 'manual',
        }]);
        setValleyStatus('idle'); setSelectedAnchors([]);
      })
      .catch(() => { if (!cancelled) { setValleyStatus('error'); setValleyError('엔진 서버에 연결할 수 없습니다.'); setSelectedAnchors([]); } });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAnchors, result.analysisId]);
  return <section className="page page--results"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">분석 작업실 <ChevronRight size={14} /> {scan.partNo}</span><h2>엔진별 실제 분석 결과</h2><p>{scan.name} · {result.source.width} × {result.source.height}px</p></div><button className="primary-button" onClick={onService}>보정 시트 만들기 <ArrowRight size={17} /></button></div>
    <div className="result-tabs" role="tablist">{(Object.keys(engineMeta) as Engine[]).map((key, index) => { const item = engineMeta[key]; const failed = Boolean(result.errors[key]); return <button role="tab" aria-selected={engine === key} className={engine === key ? 'active' : ''} onClick={() => setEngine(key)} key={key}><span style={{ color: failed ? '#bd4650' : item.color }}>0{index + 1}</span><div><b>{item.name}</b><small>{failed ? '실행 오류' : item.short}</small></div>{!failed && <Check size={17} />}</button>; })}</div>
    <div className="results-layout"><div className="viewer-card card"><div className="viewer-toolbar"><div><span className={`status ${result.errors[engine] ? 'status--error' : 'status--done'}`}>{result.errors[engine] ? <><X size={13} /> 실행 실패</> : <><Check size={13} /> 실제 분석 완료</>}</span><b>{meta.name}</b></div>{engine === 'deviation' && <button className="tool-button" onClick={() => onAllPointsToggle(!allLabelsVisible)}>{allLabelsVisible ? <EyeOff size={14} /> : <Eye size={14} />} 라벨 전체 {allLabelsVisible ? 'OFF' : 'ON'}</button>}</div><div className={`viewer-stage ${engine === 'deviation' ? 'viewer-stage--light' : ''}`}><Heatmap key={`${scan.id}-${engine}`} imageUrl={image} width={result.source.width} height={result.source.height} lightBackground={engine === 'deviation'}>{engine === 'deviation' && <CorrectionPoints coefficient={-1} points={result.points} visibleLabelIds={visibleLabelIds} onLabelToggle={toggleLabel} />}{engine === 'zero' && <>
        <ZeroZoneOverlay clusters={zeroAreaClusters} width={result.source.width} height={result.source.height} />
        <ValleyLineOverlay lines={valleyLines} width={result.source.width} height={result.source.height} />
        <AnchorPicker anchors={zeroAnchors} width={result.source.width} height={result.source.height} selectedIds={selectedAnchors} onToggle={toggleAnchor} />
      </>}</Heatmap></div><div className="viewer-legend"><span><i className="legend-dot" style={{ background: meta.color }} /> 현재 표시: {meta.name}</span><span>{engine === 'deviation' ? '라벨이나 포인트 점을 누르면 개별 표시를 켜고 끌 수 있습니다.' : '표시된 값과 위치는 업로드 이미지의 실제 엔진 결과입니다.'}</span></div></div>
      <aside className="inspection-panel"><div className="score-card card"><span className="score-card__icon" style={{ color: meta.color, background: `${meta.color}12` }}>{engine === 'label' ? <Sparkles /> : engine === 'deviation' ? <Activity /> : <Gauge />}</span><span>핵심 결과</span><strong style={{ color: result.errors[engine] ? '#bd4650' : meta.color }}>{summary.stat}</strong><p>{summary.detail}</p></div><div className="card plain-summary"><h3>쉽게 보는 결과</h3><div className="summary-line"><Check size={16} /><div><b>처리 방식</b><span>{engine === 'label' ? 'label_removal의 인페인팅 결과입니다.' : engine === 'deviation' ? '라벨 제거 이미지에 deviation_extraction의 지시선 끝점과 판독값을 겹쳐 표시합니다.' : 'zero_line_detection의 컬러바 기반 결과입니다.'}</span></div></div>{engineWarnings.length > 0 && <div className="summary-line warning"><MoveRight size={16} /><div><b>확인 필요</b><span>{engineWarnings[0]}</span></div></div>}</div><div className="card mini-table"><div className="card-title"><h3>검출 포인트</h3><span>라벨 {visibleLabelIds.size}/{result.points.length}</span></div>{result.points.map((point) => { const visible = visibleLabelIds.has(point.id); return <div className="point-list-row" key={point.id}><span>{point.id}</span><b className={point.value > 0 ? 'positive' : 'negative'}>{point.value > 0 ? '+' : ''}{point.value.toFixed(3)} mm</b><small>{point.xPx}, {point.yPx}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleLabel(point.id)} aria-label={`${point.id} 라벨 ${visible ? '숨기기' : '표시하기'}`} title={`라벨 ${visible ? 'OFF' : 'ON'}`}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}{!result.points.length && <p className="empty-mini">검출된 포인트가 없습니다.</p>}</div>
      {engine === 'zero' && <div className="card mini-table anchor-panel">
        <div className="card-title"><h3>제로라인</h3><span>앵커 {zeroAnchors.length}개{zeroAreaClusters.length > 0 ? ` · 구간 ${zeroAreaClusters.length}개` : ''}</span></div>
        {zeroAreaClusters.length > 0 && <p className="anchor-panel__hint">
          빨간 면은 편차가 0에 가까운 것으로 검출된 구간입니다.
          점 2개를 이은 선이 이 구간을 다 지나가지 않을 수 있으니 함께 참고하세요.
        </p>}
        {result.labelZeroLine && valleyLines.some((line) => line.id === 'label-zero-line') && (
          <p className="anchor-panel__status">작업자가 실측한 라벨값이 부호를 바꾸는 지점(0포인트)들을 윤곽선을 따라 이어 검출한 선입니다. 정답지를 베낀 게 아니라 스캔 실측값에서 계산했습니다. 실제와 다르면 아래에서 지우고 앵커 2개를 직접 골라 다시 이으세요.</p>
        )}
        {!result.labelZeroLine && result.advanceLine && (
          result.advanceLine.confidence === 'high'
            ? <p className="anchor-panel__status">AI가 "0.0" 표시점 기준으로 자동 제안한 선입니다. 실제와 다르면 아래에서 지우고 앵커 2개를 직접 골라 다시 이으세요.</p>
            : <p className="anchor-panel__status anchor-panel__status--error">AI 추천선의 신뢰도가 낮습니다{result.advanceLine.warnings[0] ? `: ${result.advanceLine.warnings[0]}` : ''}. 아래에서 AI 추천선을 지우고 앵커 2개를 직접 골라 이으세요.</p>
        )}
        {!result.labelZeroLine && !result.advanceLine && (result.zeroLineCandidates || []).length > 0 && <div className="candidate-list">
          <p className="anchor-panel__hint">AI가 모든 앵커 조합을 &quot;부품을 실제로 둘로 가르는 정도&quot;로 채점해 상위 {result.zeroLineCandidates.length}개를 추렸습니다. 1등을 기본으로 그렸으니, 실제와 다르면 아래에서 다른 후보를 눌러 바꾸세요. (실측 검증: 정답이 이 목록 안에는 들어오지만 1등은 아닐 수 있습니다.)</p>
          {result.zeroLineCandidates.map((cand) => {
            const active = valleyLines.some((line) => line.id === `cand-${cand.rank}`);
            return <button type="button" key={cand.rank} className={`candidate-chip ${active ? 'candidate-chip--active' : ''}`}
              onClick={() => setValleyLines((current) => {
                const others = current.filter((line) => !line.id.startsWith('cand-') && line.source !== 'ai');
                if (active) return others;
                return [...others, { id: `cand-${cand.rank}`, anchorStartId: cand.anchor_start_id, anchorEndId: cand.anchor_end_id, points: cand.points, length_px: cand.length_px, mean_abs_deviation: cand.mean_abs_deviation, source: 'ai' as const }];
              })}>
              <b>{cand.rank}위</b>
              <span>앵커 {cand.anchor_start_id}↔{cand.anchor_end_id}</span>
              <small>분리도 {cand.separation.toFixed(2)}</small>
            </button>;
          })}
        </div>}
        {result.referenceLine && (() => {
          const ref = result.referenceLine;
          const showing = valleyLines.some((line) => line.id.startsWith('sheet-reference'));
          return <div className="anchor-panel__compare">
            <p className="anchor-panel__hint">
              품번 {ref.partNo} 의 보정시트({ref.sourceSheet})에 표기된 제로{ref.kind === 'areas' ? ` 구간 ${ref.contours.length}개` : '라인'}을 정답 비교용으로 참고할 수 있습니다{ref.mirrored ? ' (시트 그림이 좌우반전이라 뒤집어 맞춤)' : ''}. 위 검출선은 이 시트를 베낀 게 아니라 실측값에서 별도로 계산한 결과입니다.
            </p>
            <button type="button" className={`text-button ${showing ? 'candidate-chip--active' : ''}`} onClick={() => setValleyLines((current) => {
              if (showing) return current.filter((line) => !line.id.startsWith('sheet-reference'));
              const extra: ValleyLine[] = ref.kind === 'areas'
                ? ref.contours.map((contour, i) => ({ id: `sheet-reference-${i}`, anchorStartId: null, anchorEndId: null, points: contour, length_px: 0, mean_abs_deviation: 0, source: 'ai' as const }))
                : [{ id: 'sheet-reference-0', anchorStartId: null, anchorEndId: null, points: ref.points, length_px: 0, mean_abs_deviation: 0, source: 'ai' as const }];
              return [...current, ...extra];
            })}>{showing ? '정답지 비교선(파란 점선) 숨기기' : '정답지 비교선(파란 점선) 보기'}</button>
          </div>;
        })()}
        <p className="anchor-panel__hint">{zeroAnchors[0]?.source === 'label_zero_point'
          ? '초록 점은 작업자가 측정한 라벨값에서 부호가 바뀌는 지점(0포인트)입니다. 2개를 순서대로 클릭하면 그 사이를 편차가 낮은 경로로 잇습니다.'
          : '목록에 정답이 없으면, 이미지 위 초록 점(앵커) 2개를 순서대로 클릭해 직접 이으세요 (경로 정확도 검증: 대각선 대비 오차 약 3.68%).'}</p>
        {selectedAnchors.length > 0 && valleyStatus !== 'error' && <p className="anchor-panel__status">선택됨: {selectedAnchors.join(', ')} {valleyStatus === 'loading' && '· 잇는 중…'}</p>}
        {valleyStatus === 'error' && valleyError && <p className="anchor-panel__status anchor-panel__status--error">{valleyError}</p>}
        {valleyLines.map((line) => <div className="point-list-row" key={line.id}>
          <span>{line.id === 'label-zero-line' ? '라벨 검출' : line.id.startsWith('cand-') ? 'AI 후보' : line.id === 'ai-suggestion' ? 'AI 추천' : line.id.startsWith('sheet-reference') ? '정답지 비교' : line.source === 'ai' ? 'AI 추천' : `${line.anchorStartId}↔${line.anchorEndId}`}</span>
          <b className="positive">{line.source === 'ai' ? `점 ${line.points.length}개` : `${Math.round(line.length_px)}px`}</b>
          <small>{line.source === 'ai' ? '' : `평균|편차| ${line.mean_abs_deviation.toFixed(3)}`}</small>
          <button type="button" className="label-visibility" onClick={() => setValleyLines((current) => current.filter((item) => item.id !== line.id))} aria-label="이 선 지우기" title="이 선 지우기"><X size={14} /></button>
        </div>)}
        {valleyLines.length > 0 && <button type="button" className="text-button anchor-panel__reset" onClick={() => { setValleyLines([]); setSelectedAnchors([]); }}>전체 초기화</button>}
        {!zeroAnchors.length && <p className="empty-mini">검출된 앵커가 없습니다.</p>}
      </div>}</aside>
    </div></section>;
}

function FolderTreeNode({ entry, selectedPath, onOpen }: { entry: FolderEntry; selectedPath: string; onOpen: (path: string) => void }) {
  const [expanded, setExpanded] = useState(false); const [children, setChildren] = useState<FolderEntry[]>([]); const [loaded, setLoaded] = useState(false);
  const toggle = async () => {
    if (!loaded) {
      const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(entry.path)}`); const data = await response.json();
      if (response.ok) { setChildren((data.entries || []).filter((item: FolderEntry) => item.isDirectory)); setLoaded(true); }
    }
    setExpanded((value) => !value); onOpen(entry.path);
  };
  return <div className="tree-node"><button className={`tree-root ${selectedPath === entry.path ? 'selected' : ''}`} onClick={toggle}>{expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}<Folder size={17} fill="currentColor" /> <span>{entry.name}</span></button>{expanded && <div className="tree-children">{children.map((child) => <FolderTreeNode key={child.path} entry={child} selectedPath={selectedPath} onOpen={onOpen} />)}{loaded && !children.length && <span className="tree-empty">하위 폴더 없음</span>}</div>}</div>;
}

function Explorer() {
  const [available, setAvailable] = useState<boolean | null>(null); const [rootName, setRootName] = useState('품번별 폴더'); const [rootEntries, setRootEntries] = useState<FolderEntry[]>([]); const [entries, setEntries] = useState<FolderEntry[]>([]); const [path, setPath] = useState(''); const [query, setQuery] = useState('');
  const openFolder = async (nextPath: string) => {
    const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(nextPath)}`); const data = await response.json();
    if (!response.ok || data.available === false) { setAvailable(false); return; }
    setAvailable(true); setRootName(data.rootName); setEntries(data.entries || []); setPath(data.path || '');
    if (!nextPath) setRootEntries((data.entries || []).filter((item: FolderEntry) => item.isDirectory));
  };
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/folders?path=`)
      .then((response) => response.json().then((data) => ({ ok: response.ok, data })))
      .then(({ ok, data }) => {
        if (cancelled) return;
        if (!ok || data.available === false) { setAvailable(false); return; }
        const nextEntries = data.entries || [];
        setAvailable(true); setRootName(data.rootName); setEntries(nextEntries); setPath('');
        setRootEntries(nextEntries.filter((item: FolderEntry) => item.isDirectory));
      })
      .catch(() => { if (!cancelled) setAvailable(false); });
    return () => { cancelled = true; };
  }, []);
  if (available === false) return null;
  const filtered = entries.filter((entry) => entry.name.toLowerCase().includes(query.toLowerCase())); const segments = path ? path.split('/') : [];
  return <div className="explorer card"><div className="explorer__title"><div><FolderOpen size={20} /><b>실시간 품번별 폴더</b></div><span>{available == null ? '연결 확인 중' : '현재 PC 폴더와 연결됨'}</span></div><div className="explorer__bar"><div className="explorer__crumb"><button disabled={!path} onClick={() => openFolder(segments.slice(0, -1).join('/'))}><ArrowLeft size={14} /></button><span><button onClick={() => openFolder('')}>{rootName}</button>{segments.map((segment, index) => <span key={`${segment}-${index}`}><ChevronRight size={13} /><button onClick={() => openFolder(segments.slice(0, index + 1).join('/'))}>{segment}</button></span>)}</span></div><label><ZoomIn size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="현재 폴더 검색" /></label></div><div className="explorer__body"><div className="folder-tree"><button className={`tree-root ${!path ? 'selected' : ''}`} onClick={() => openFolder('')}><ChevronDown size={15} /><FolderOpen size={17} /> <span>{rootName}</span></button><div className="tree-children">{rootEntries.map((entry) => <FolderTreeNode key={entry.path} entry={entry} selectedPath={path} onOpen={openFolder} />)}</div></div><div className="folder-content"><div className="folder-content__head"><span>이름</span><span>수정한 날짜</span><span>크기</span></div>{filtered.map((entry) => <button className="folder-row" key={entry.path} onDoubleClick={() => entry.isDirectory && openFolder(entry.path)} onClick={() => entry.isDirectory && openFolder(entry.path)}><span>{entry.isDirectory ? <Folder size={19} fill="currentColor" /> : <File size={18} />}{entry.name}</span><small>{new Date(entry.modified).toLocaleString('ko-KR')}</small><small>{entry.isDirectory ? '파일 폴더' : formatBytes(entry.size)}</small></button>)}{!filtered.length && <div className="empty-search">이 폴더는 비어 있습니다.</div>}<div className="folder-content__status">{filtered.length}개 항목 <span>·</span> 실시간 로컬 조회</div></div></div></div>;
}

function SheetTitleBlock({ scan }: { scan: ScanItem }) {
  const partName = scan.name.replace(/\.[^.]+$/, '');
  const appliedDate = new Intl.DateTimeFormat('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' }).format(new Date());

  return <section className="sheet-title-block" aria-label="보정 적용 내용">
    <div className="sheet-title-block__heading"><strong>보정 적용 내용</strong></div>
    <div className="sheet-title-block__label">관리 NO</div><div className="sheet-title-block__value">ADC-{scan.partNo}</div>
    <div className="sheet-title-block__label">PART NAME</div><div className="sheet-title-block__value" title={partName}>{partName}</div>
    <div className="sheet-title-block__label">공정</div><div className="sheet-title-block__value">금형 보정</div>
    <div className="sheet-title-block__label">PART NO</div><div className="sheet-title-block__value">{scan.partNo}</div>
    <div className="sheet-title-block__label">원소재</div><div className="sheet-title-block__value">3D SCAN DATA</div>
    <div className="sheet-title-block__label">적용일자</div><div className="sheet-title-block__value">{appliedDate}</div>
  </section>;
}

function ServicePreview({ scan, folderAvailable, hiddenPointIds, onPointToggle, valleyLines }: { scan: ScanItem; folderAvailable: boolean; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; valleyLines: ValleyLine[] }) {
  const result = scan.result!; const points = result.points; const visiblePointIds = new Set(points.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id)); const [coefficient, setCoefficient] = useState(1); const [showPoints, setShowPoints] = useState(true); const [showZero, setShowZero] = useState(true);
  const maxCorrection = useMemo(() => points.length ? Math.max(...points.map((point) => Math.abs(point.value * coefficient))) : 0, [coefficient, points]);
  const baseImage = result.zeroOverlay || result.cleanImage || scan.url;
  return <section className="page page--service"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">ADC · Ajin Die Compensation <span className="demo-badge">DEMO</span></span><h2>ADC 금형 보정 시트</h2><p>실제 라벨 제거 이미지에 검출된 편차값과 제로라인을 합성합니다.</p></div></div><div className="service-grid"><div className="correction-card card"><div className="viewer-toolbar"><div><span className="status status--done"><Check size={13} /> 실제 결과 합성</span><b>{scan.partNo} · 보정 작업 지시도</b></div><div className="layer-toggles"><button className={showPoints ? 'active orange' : ''} onClick={() => setShowPoints(!showPoints)}><i /> 보정치</button><button className={showZero ? 'active green' : ''} onClick={() => setShowZero(!showZero)} disabled={!valleyLines.length}><i /> 제로라인</button></div></div><SheetTitleBlock scan={scan} /><div className="sheet-stage sheet-stage--light"><Heatmap key={`${scan.id}-service`} imageUrl={baseImage} width={result.source.width} height={result.source.height} lightBackground>{showPoints && <CorrectionPoints coefficient={coefficient} points={points} visibleLabelIds={visiblePointIds} onLabelToggle={onPointToggle} />}{showZero && <ValleyLineOverlay lines={valleyLines} width={result.source.width} height={result.source.height} />}</Heatmap><div className="sheet-stamp"><span>AJIN INDUSTRIAL</span><b>DIE CORRECTION SHEET</b><small>{scan.partNo} · REV.01</small></div></div><div className="sheet-note"><ShieldCheck size={17} /><span><b>검토용 가상 보정치입니다.</b> 실제 가공 전 담당자 승인과 현장 검증이 필요합니다.</span></div></div><aside className="control-panel"><div className="card coefficient-card"><div className="card-title"><div><h3>보정 계수</h3><p>편차값에 곱할 비율을 조절합니다.</p></div><span>{coefficient.toFixed(2)}×</span></div><div className="coefficient-input"><input aria-label="보정 계수 직접 입력" type="number" min="0.5" max="1.5" step="0.01" value={coefficient} onChange={(e) => { const value = e.target.valueAsNumber; if (!Number.isNaN(value)) setCoefficient(Math.max(0.5, Math.min(1.5, value))); }} /><span>×</span></div><input aria-label="보정 계수" type="range" min="0.5" max="1.5" step="0.05" value={coefficient} onChange={(e) => setCoefficient(Number(e.target.value))} /><div className="range-labels"><span>보수적 0.50</span><span>기준 1.00</span><span>적극적 1.50</span></div><div className="formula"><span>보정치</span><b>= 편차 × {coefficient.toFixed(2)} × (−1)</b></div></div><div className="card correction-summary"><h3>실제 엔진 요약</h3><div><span>보정 포인트</span><b>{visiblePointIds.size}개</b></div><div><span>최대 보정량</span><b className="orange">{maxCorrection.toFixed(3)} mm</b></div><div><span>제로라인</span><b className="green">{valleyLines.length ? `선 ${valleyLines.length}개` : '없음'}</b></div><div><span>처리 품번</span><b>{scan.partNo}</b></div></div></aside></div>{folderAvailable && <Explorer />}</section>;
}

// 3D CAD/스캔 데이터 화면.
//
// 현업 자료(2026-08-25)가 정리한 제로라인 판정 4가지 방법 중 3가지
// (RPS 정렬, 수축 중심선, 단면 분석)는 3D 데이터가 있어야 한다. 지금까지
// 우리가 가진 건 2D 히트맵뿐이라 컬러맵 제로존 하나만 쓸 수 있었다.
// 여기서 STEP 을 읽으면 조립 홀 좌표가 나오므로 RPS 정렬의 입구가 열린다.
function CadWorkspace() {
  const [mesh, setMesh] = useState<CadMesh | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [showHoles, setShowHoles] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = (file: File) => {
    setStatus('loading'); setError(null);
    const body = new FormData();
    body.append('file', file);
    fetch(`${API_BASE}/api/cad`, { method: 'POST', body })
      .then(async (response) => {
        const data = await response.json() as CadMesh & { error?: string };
        if (!response.ok) throw new Error(data.error || '읽지 못했습니다.');
        return data;
      })
      .then((data) => { setMesh(data); setStatus('idle'); })
      .catch((err) => { setError(String(err.message || err)); setStatus('error'); setMesh(null); });
  };

  const onPick = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) upload(file);
    event.target.value = '';
  };
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) upload(file);
  };

  const summary = mesh?.summary;
  return <section className="page page--cad">
    <div className="page-heading page-heading--compact">
      <div>
        <span className="breadcrumb">ADC · Ajin Die Compensation</span>
        <h2>3D 데이터</h2>
        <p>STEP·STL을 읽어 형상과 조립 홀을 확인합니다. 이 PC 안에서만 처리됩니다.</p>
      </div>
      <button className="primary-button" onClick={() => inputRef.current?.click()}>
        3D 파일 열기 <UploadCloud size={17} />
      </button>
      <input ref={inputRef} type="file" hidden accept=".step,.stp,.stl,.ply,.obj,.glb,.gltf,.3mf" onChange={onPick} />
    </div>

    <div className="cad-layout">
      <div className="card cad-stage" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
        {mesh
          ? <>
              <div className="viewer-toolbar">
                <div>
                  <span className="status status--done"><Check size={13} /> {summary?.source_format.toUpperCase()} 읽기 완료</span>
                  <b>{summary?.name}</b>
                </div>
                {mesh.holes.length > 0 && <button className="tool-button" onClick={() => setShowHoles(!showHoles)}>
                  {showHoles ? <EyeOff size={14} /> : <Eye size={14} />} 홀 표시 {showHoles ? 'OFF' : 'ON'}
                </button>}
              </div>
              <div className="cad-viewer"><CadViewer mesh={mesh} showHoles={showHoles} /></div>
              <div className="viewer-legend">
                <span>드래그: 회전 · 휠: 확대 · 오른쪽 드래그(또는 Shift+드래그): 이동</span>
                <span>{summary?.n_faces.toLocaleString()} 삼각형</span>
              </div>
            </>
          : <div className="cad-drop">
              {status === 'loading'
                ? <><Play size={30} /><b>읽는 중…</b><span>큰 파일은 시간이 걸립니다.</span></>
                : <><Box size={30} /><b>3D 파일을 여기에 놓으세요</b>
                    <span>STEP · STL · PLY · OBJ · GLB · 3MF</span>
                    <small>CATIA 네이티브(.CATPart)는 독자 포맷이라 읽을 수 없습니다 — STEP(AP214)으로 내보내 주세요.</small></>}
            </div>}
      </div>

      <aside className="inspection-panel">
        {error && <div className="card cad-error"><b>읽지 못했습니다</b><p>{error}</p></div>}
        {summary && <>
          <div className="card plain-summary">
            <h3>부품 정보</h3>
            <div className="summary-line"><Check size={16} /><div><b>크기 ({summary.units})</b>
              <span>{summary.bounds.size.map((v) => v.toFixed(1)).join(' × ')}</span></div></div>
            <div className="summary-line"><Check size={16} /><div><b>원래 위치 중심</b>
              <span>{summary.bounds.center.map((v) => v.toFixed(1)).join(', ')}</span></div></div>
            <div className="summary-line"><Check size={16} /><div><b>메시</b>
              <span>삼각형 {summary.n_faces.toLocaleString()} · 정점 {summary.n_vertices.toLocaleString()}</span></div></div>
          </div>

          <div className="card mini-table">
            <div className="card-title"><h3>조립 홀 (RPS 후보)</h3><span>{mesh.holes.length}개</span></div>
            {mesh.note && <p className="anchor-panel__hint">{mesh.note}</p>}
            {mesh.holes.map((hole, index) => <div className="point-list-row" key={index}>
              <span>ø{hole.diameter.toFixed(1)}</span>
              <b className="positive">{hole.center.map((v) => v.toFixed(0)).join(', ')}</b>
              <small>깊이 {hole.height.toFixed(1)}</small>
              <span />
            </div>)}
            {!mesh.holes.length && !mesh.note && <p className="empty-mini">홀을 찾지 못했습니다.</p>}
          </div>

          {mesh.planes.length > 0 && <div className="card mini-table">
            <div className="card-title"><h3>기준면 후보</h3><span>{mesh.counts.planes}개</span></div>
            <p className="anchor-panel__hint">넓은 평면 순입니다. 어느 면이 실제 기준면인지는 도면에 따라 사람이 지정합니다.</p>
            {mesh.planes.slice(0, 6).map((plane, index) => <div className="point-list-row" key={index}>
              <span>#{index + 1}</span>
              <b className="positive">{(plane.area / 100).toFixed(0)} cm²</b>
              <small>법선 {plane.normal.map((v) => v.toFixed(2)).join(', ')}</small>
              <span />
            </div>)}
          </div>}
        </>}
      </aside>
    </div>
  </section>;
}

export default function Home() {
  const [view, setView] = useState<View>('workspace'); const [scans, setScans] = useState<ScanItem[]>([]); const [activeId, setActiveId] = useState<string>(); const [collapsed, setCollapsed] = useState(false); const [backendOnline, setBackendOnline] = useState<boolean | null>(null); const [folderAvailable, setFolderAvailable] = useState(false); const [hiddenPointIdsByScan, setHiddenPointIdsByScan] = useState<Record<string, Set<string>>>({});
  const [valleyLinesByScan, setValleyLinesByScan] = useState<Record<string, ValleyLine[]>>({});
  useEffect(() => { fetch(`${API_BASE}/api/health`).then((response) => response.json()).then((data) => { setBackendOnline(Boolean(data.ok)); setFolderAvailable(Boolean(data.folderAvailable)); }).catch(() => setBackendOnline(false)); }, []);
  const resolvedActiveId = activeId || scans[0]?.id;
  const activeScan = scans.find((scan) => scan.id === resolvedActiveId); const completedScan = activeScan?.result ? activeScan : scans.find((scan) => scan.result); const hasResult = Boolean(completedScan?.result);
  const hiddenPointIds = completedScan ? hiddenPointIdsByScan[completedScan.id] || new Set<string>() : new Set<string>();
  const togglePoint = (id: string) => completedScan && setHiddenPointIdsByScan((current) => { const next = new Set(current[completedScan.id] || []); if (next.has(id)) next.delete(id); else next.add(id); return { ...current, [completedScan.id]: next }; });
  const setAllPointsVisible = (visible: boolean) => completedScan && setHiddenPointIdsByScan((current) => ({ ...current, [completedScan.id]: visible ? new Set() : new Set(completedScan.result!.points.map((point) => point.id)) }));
  // 스캔이 새로 분석되면 제로라인을 미리 채워둔다. 우선순위는 "실제
  // 검출"이 먼저다 — referenceLine(보정시트에서 그대로 베낀 것)은 정답
  // 카피일 뿐 검출이 아니라서 기본 화면에 자동으로 깔지 않는다. 전에는
  // referenceLine 이 먼저 낚아채서, 오늘 새로 만든 라벨 0포인트 기반
  // 검출(labelZeroLine)이 있어도 화면엔 항상 시트 카피만 보이는 버그가
  // 있었다 — 등록된 품번(64XX2, 67XX6)마다 "바뀐 게 없다"고 느껴진 이유.
  useEffect(() => {
    if (!completedScan || valleyLinesByScan[completedScan.id] !== undefined) return;
    const labelLine = completedScan.result?.labelZeroLine;
    if (labelLine && labelLine.points.length >= 2) {
      setValleyLinesByScan((current) => ({ ...current, [completedScan.id]: [{
        id: 'label-zero-line', anchorStartId: null, anchorEndId: null,
        points: labelLine.points, length_px: labelLine.length_px,
        mean_abs_deviation: labelLine.mean_abs_deviation, source: 'ai',
      }] }));
      return;
    }
    const best = (completedScan.result?.zeroLineCandidates || [])[0];
    const advance = completedScan.result?.advanceLine;
    const initial: ValleyLine[] = best
      ? [{ id: `cand-${best.rank}`, anchorStartId: best.anchor_start_id, anchorEndId: best.anchor_end_id, points: best.points, length_px: best.length_px, mean_abs_deviation: best.mean_abs_deviation, source: 'ai' }]
      : advance && advance.points.length >= 2
        ? [{ id: 'ai-suggestion', anchorStartId: null, anchorEndId: null, points: advance.points, length_px: 0, mean_abs_deviation: 0, source: 'ai' }]
        : [];
    setValleyLinesByScan((current) => ({ ...current, [completedScan.id]: initial }));
  }, [completedScan, valleyLinesByScan]);
  const valleyLines = completedScan ? valleyLinesByScan[completedScan.id] || [] : [];
  const setValleyLines = (updater: ValleyLine[] | ((current: ValleyLine[]) => ValleyLine[])) => completedScan && setValleyLinesByScan((current) => {
    const previous = current[completedScan.id] || [];
    const next = typeof updater === 'function' ? (updater as (value: ValleyLine[]) => ValleyLine[])(previous) : updater;
    return { ...current, [completedScan.id]: next };
  });
  const openResults = (id: string) => { setActiveId(id); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const selectView = (next: View) => { if (next === 'workspace' || next === 'cad' || hasResult) setView(next); };
  return <main className={`app-shell ${collapsed ? 'app-shell--collapsed' : ''}`}><Sidebar view={view} setView={selectView} collapsed={collapsed} setCollapsed={setCollapsed} hasResult={hasResult} /><div className="app-main"><Header scans={scans} activeId={resolvedActiveId} setActiveId={setActiveId} />{view === 'workspace' && <Workspace scans={scans} setScans={setScans} onOpenResults={openResults} backendOnline={backendOnline} />}{view === 'results' && completedScan?.result && <Results scan={completedScan} onService={() => setView('service')} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} onAllPointsToggle={setAllPointsVisible} valleyLines={valleyLines} setValleyLines={setValleyLines} />}{view === 'service' && completedScan?.result && <ServicePreview scan={completedScan} folderAvailable={folderAvailable} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} valleyLines={valleyLines} />}{view === 'cad' && <CadWorkspace />}</div></main>;
}
