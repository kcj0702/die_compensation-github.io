'use client';

/* Blob/data URLs are local engine outputs and cannot use Next's remote image loader. */
/* eslint-disable @next/next/no-img-element */

import {
  Activity, ArrowLeft, ArrowRight, ArrowUpRight, BarChart3, Check, ChevronDown, ChevronRight,
  Circle, CircleHelp, Eye, EyeOff, File, Folder, FolderOpen, Gauge, Grid2X2, Image as ImageIcon,
  Layers3, ListFilter, Maximize2, MousePointer2, MoveRight, PanelLeftClose, Play, Settings2,
  ShieldCheck, Sparkles, Square, Trash2, Type, UploadCloud, X, ZoomIn, ZoomOut,
} from 'lucide-react';
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

type View = 'workspace' | 'results' | 'service';
type Engine = 'label' | 'deviation' | 'zero';
type ScanStatus = 'ready' | 'analyzing' | 'done' | 'error';
type PointResult = { id: string; xPx: number; yPx: number; x: number; y: number; value: number; labelColor: string; confidence: string };
type AnalysisResult = {
  source: { name: string; width: number; height: number };
  cleanImage: string | null;
  zeroOverlay: string | null;
  zeroMask: string | null;
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

/* 보정 시트 주석 — 좌표와 크기는 모두 이미지 대비 %라 확대/축소와 창 크기에 영향받지 않는다. */
type AnnotationKind = 'rect' | 'ellipse' | 'text' | 'arrow';
type AnnotationTool = 'select' | AnnotationKind;
/* 사각형·타원·텍스트는 x,y 가 좌상단이고 w,h 가 크기다. 화살표는 x,y 가 시작점이고 w,h 가 끝점까지의 변위라 음수가 될 수 있다. */
type Annotation = { id: string; kind: AnnotationKind; x: number; y: number; w: number; h: number; text?: string; fontSize?: number; color?: string };

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
    onDoubleClick={(event) => { if ((event.target as Element).closest('.measure-point, .zoom-controls, .annotation-layer--armed, .annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-arrow__hit')) return; resetView(); }}
    onPointerDown={(event) => {
      if (scale <= 1 || (event.target as Element).closest('.measure-point, .zoom-controls, .annotation-layer--armed, .annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-arrow__hit')) return;
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

const MIN_ANNOTATION_SIZE = 2;
const DEFAULT_ANNOTATION_SIZE: Record<AnnotationKind, { w: number; h: number }> = {
  rect: { w: 14, h: 10 }, ellipse: { w: 14, h: 12 }, text: { w: 16, h: 7 }, arrow: { w: 12, h: -8 },
};
/* 주석 색상. 도면에서 구분이 잘 되는 색만 골랐다. */
const ANNOTATION_COLORS = [
  { hex: '#e8802f', label: '주황' },
  { hex: '#d33f3f', label: '빨강' },
  { hex: '#2f7fd6', label: '파랑' },
  { hex: '#17a06f', label: '초록' },
  { hex: '#8b5cd6', label: '보라' },
  { hex: '#3d4550', label: '먹색' },
];
const DEFAULT_ANNOTATION_COLOR = ANNOTATION_COLORS[0].hex;

/* 받침이 있으면 '으로', 없거나 ㄹ 받침이면 '로'. (보라 → 보라로, 주황 → 주황으로) */
function withRo(word: string) {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  if (code < 0 || code > 11171) return `${word}로`;
  const jong = code % 28;
  return `${word}${jong === 0 || jong === 8 ? '' : '으'}로`;
}

/* 도형 채움은 같은 색을 옅게 깐다. CSS 만으로는 색을 반투명하게 못 만들어 여기서 계산한다. */
function withAlpha(hex: string, alpha: number) {
  const value = hex.replace('#', '');
  const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
  const int = parseInt(full, 16);
  if (!Number.isFinite(int)) return hex;
  return `rgba(${(int >> 16) & 255}, ${(int >> 8) & 255}, ${int & 255}, ${alpha})`;
}

/* 글자 크기는 이미지와 함께 확대·축소되도록 레이어 안의 px 로 다룬다. */
const DEFAULT_TEXT_SIZE = 10;
const TEXT_SIZE_MIN = 6;
const TEXT_SIZE_MAX = 48;
const TEXT_SIZE_STEP = 2;
const BOX_HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const;
const HANDLE_OFFSET: Record<string, { x: number; y: number }> = {
  nw: { x: 0, y: 0 }, n: { x: 0.5, y: 0 }, ne: { x: 1, y: 0 }, e: { x: 1, y: 0.5 },
  se: { x: 1, y: 1 }, s: { x: 0.5, y: 1 }, sw: { x: 0, y: 1 }, w: { x: 0, y: 0.5 },
};

let annotationSeq = 0;
const nextAnnotationId = () => `ann-${Date.now().toString(36)}-${++annotationSeq}`;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/* 포인터가 이미 놓였거나 취소된 경우 setPointerCapture 는 예외를 던진다.
   캡처는 커서가 도형 밖으로 나가도 이벤트를 계속 받기 위한 편의일 뿐이라,
   실패해도 조작 자체는 이어져야 한다. */
function capturePointer(element: Element, pointerId: number) {
  try { element.setPointerCapture(pointerId); } catch { /* 캡처 없이 진행 */ }
}

/* 드래그가 끝나면 음수 크기를 뒤집고 최소 크기를 보장한다. 화살표는 방향이 의미를 가지므로 그대로 둔다. */
function normalizeAnnotation(annotation: Annotation): Annotation {
  if (annotation.kind === 'arrow') return annotation;
  let { x, y, w, h } = annotation;
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  w = Math.max(w, MIN_ANNOTATION_SIZE); h = Math.max(h, MIN_ANNOTATION_SIZE);
  return { ...annotation, x: clamp(x, 0, 100 - w), y: clamp(y, 0, 100 - h), w, h };
}

function AnnotationToolbar({ tool, setTool, hasAnnotations, onClearAll, selectedColor, onColorChange }: { tool: AnnotationTool; setTool: (tool: AnnotationTool) => void; hasAnnotations: boolean; onClearAll: () => void; selectedColor: string | null; onColorChange: (hex: string) => void }) {
  const tools: { id: AnnotationTool; icon: typeof Square; label: string }[] = [
    { id: 'select', icon: MousePointer2, label: '선택 · 이동' },
    { id: 'rect', icon: Square, label: '사각형 강조' },
    { id: 'ellipse', icon: Circle, label: '원형 강조' },
    { id: 'arrow', icon: ArrowUpRight, label: '화살표' },
    { id: 'text', icon: Type, label: '텍스트 상자' },
  ];
  return <div className="annotation-toolbar" role="toolbar" aria-label="주석 도구">
    {tools.map((item) => { const Icon = item.icon; return <button key={item.id} type="button" className={tool === item.id ? 'active' : ''} onClick={() => setTool(item.id)} aria-pressed={tool === item.id} aria-label={item.label} title={item.label}><Icon size={14} /></button>; })}
    <span className="annotation-toolbar__divider" />
    {/* 팔레트는 모드가 아니라 동작이다. 선택한 주석의 색을 그대로 비추므로 화면과 어긋나지 않는다. */}
    <div className={`annotation-palette ${selectedColor ? '' : 'annotation-palette--idle'}`} role="group" aria-label="주석 색상">
      {ANNOTATION_COLORS.map((item) => <button key={item.hex} type="button" className={`annotation-swatch ${selectedColor === item.hex ? 'annotation-swatch--active' : ''}`}
        style={{ ['--swatch' as string]: item.hex }} onClick={() => onColorChange(item.hex)} disabled={!selectedColor} aria-pressed={selectedColor === item.hex}
        aria-label={item.label} title={selectedColor ? `${withRo(item.label)} 변경` : '주석을 먼저 선택하세요'} />)}
    </div>
    <span className="annotation-toolbar__divider" />
    <button type="button" onClick={onClearAll} disabled={!hasAnnotations} aria-label="주석 전체 삭제" title="주석 전체 삭제"><Trash2 size={14} /></button>
  </div>;
}

function AnnotationLayer({ annotations, tool, setTool, selectedId, onSelect, onCommit, onCreate, onDelete }: { annotations: Annotation[]; tool: AnnotationTool; setTool: (tool: AnnotationTool) => void; selectedId: string | null; onSelect: (id: string | null) => void; onCommit: (annotation: Annotation) => void; onCreate: (annotation: Annotation) => void; onDelete: (id: string) => void }) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [layerSize, setLayerSize] = useState({ width: 0, height: 0 });
  /* 드래그 중에는 부모 state 를 건드리지 않고 여기서만 갱신한다. 포인트가 수십 개일 때 시트 전체가 매 프레임 다시 그려지는 걸 막는다. */
  const [draft, setDraftState] = useState<Annotation | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const opRef = useRef<{ mode: 'draw' | 'move' | 'resize'; handle?: string; startX: number; startY: number; origin: Annotation; moved: boolean } | null>(null);
  /* pointerup 이 리렌더 전에 도착해도 마지막 값을 잃지 않도록 ref 로도 들고 있는다. */
  const draftRef = useRef<Annotation | null>(null);
  const setDraft = (next: Annotation | null) => { draftRef.current = next; setDraftState(next); };

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
    if (!selectedId || editingId) return;
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); onDelete(selectedId); onSelect(null); }
      if (event.key === 'Escape') onSelect(null);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [selectedId, editingId, onDelete, onSelect]);

  /* 주석 바깥을 누르면 선택을 푼다. 레이어 자체는 pointer-events 를 받지 않아서 문서 단위로 감시해야 한다. */
  useEffect(() => {
    if (!selectedId) return;
    const handleOutside = (event: PointerEvent) => {
      /* document 나 window 가 target 인 이벤트도 들어올 수 있어 Element 인지 먼저 확인한다. */
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest('.annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-toolbar, .annotation-arrow__hit')) return;
      onSelect(null);
    };
    document.addEventListener('pointerdown', handleOutside);
    return () => document.removeEventListener('pointerdown', handleOutside);
  }, [selectedId, onSelect]);

  /* getBoundingClientRect 는 확대/이동 변형이 반영된 값이라 어떤 배율에서도 같은 % 가 나온다. */
  const toPercent = (clientX: number, clientY: number) => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return { x: 0, y: 0 };
    return { x: (clientX - rect.left) / rect.width * 100, y: (clientY - rect.top) / rect.height * 100 };
  };

  const beginDraw = (event: React.PointerEvent<HTMLDivElement>) => {
    if (tool === 'select' || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    const seed: Annotation = { id: nextAnnotationId(), kind: tool, x: start.x, y: start.y, w: 0, h: 0, color: DEFAULT_ANNOTATION_COLOR, ...(tool === 'text' ? { text: '' } : {}) };
    opRef.current = { mode: 'draw', startX: start.x, startY: start.y, origin: seed, moved: false };
    setDraft(seed);
    capturePointer(event.currentTarget, event.pointerId);
  };

  const beginMove = (event: React.PointerEvent<Element>, annotation: Annotation) => {
    if (tool !== 'select' || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    opRef.current = { mode: 'move', startX: start.x, startY: start.y, origin: annotation, moved: false };
    onSelect(annotation.id);
    setDraft(annotation);
    capturePointer(event.currentTarget, event.pointerId);
  };

  const beginResize = (event: React.PointerEvent<Element>, annotation: Annotation, handle: string) => {
    if (event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    opRef.current = { mode: 'resize', handle, startX: start.x, startY: start.y, origin: annotation, moved: false };
    setDraft(annotation);
    capturePointer(event.currentTarget, event.pointerId);
  };

  const handleMove = (event: React.PointerEvent<Element>) => {
    const op = opRef.current;
    if (!op) return;
    const now = toPercent(event.clientX, event.clientY);
    const dx = now.x - op.startX;
    const dy = now.y - op.startY;
    if (Math.abs(dx) > 0.3 || Math.abs(dy) > 0.3) op.moved = true;
    const origin = op.origin;

    if (op.mode === 'draw') {
      setDraft({ ...origin, w: clamp(now.x, 0, 100) - origin.x, h: clamp(now.y, 0, 100) - origin.y });
      return;
    }
    if (op.mode === 'move') {
      if (origin.kind === 'arrow') {
        const minX = Math.min(0, -origin.w); const maxX = 100 - Math.max(0, origin.w);
        const minY = Math.min(0, -origin.h); const maxY = 100 - Math.max(0, origin.h);
        setDraft({ ...origin, x: clamp(origin.x + dx, minX, maxX), y: clamp(origin.y + dy, minY, maxY) });
      } else {
        setDraft({ ...origin, x: clamp(origin.x + dx, 0, 100 - origin.w), y: clamp(origin.y + dy, 0, 100 - origin.h) });
      }
      return;
    }
    if (origin.kind === 'arrow') {
      /* 화살표는 잡은 쪽 끝점만 따라오고 반대쪽은 제자리를 지킨다. */
      if (op.handle === 'start') {
        const nx = clamp(now.x, 0, 100); const ny = clamp(now.y, 0, 100);
        setDraft({ ...origin, x: nx, y: ny, w: origin.x + origin.w - nx, h: origin.y + origin.h - ny });
      } else {
        setDraft({ ...origin, w: clamp(now.x, 0, 100) - origin.x, h: clamp(now.y, 0, 100) - origin.y });
      }
      return;
    }
    const handle = op.handle || 'se';
    let { x, y, w, h } = origin;
    if (handle.includes('n')) { y = origin.y + dy; h = origin.h - dy; }
    if (handle.includes('s')) { h = origin.h + dy; }
    if (handle.includes('w')) { x = origin.x + dx; w = origin.w - dx; }
    if (handle.includes('e')) { w = origin.w + dx; }
    setDraft({ ...origin, x, y, w, h });
  };

  const endOperation = (event: React.PointerEvent<Element>) => {
    const op = opRef.current;
    const current = draftRef.current;
    opRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (!op || !current) { setDraft(null); return; }
    setDraft(null);

    if (op.mode === 'draw') {
      /* 끌지 않고 툭 찍기만 해도 기본 크기 도형이 생기게 한다. */
      const tiny = Math.abs(current.w) < MIN_ANNOTATION_SIZE && Math.abs(current.h) < MIN_ANNOTATION_SIZE;
      const sized = tiny ? { ...current, ...DEFAULT_ANNOTATION_SIZE[current.kind] } : current;
      const created = normalizeAnnotation(sized);
      onCreate(created);
      onSelect(created.id);
      setTool('select');
      if (created.kind === 'text') { setEditingId(created.id); setEditText(''); }
      return;
    }
    if (!op.moved) return;
    onCommit(normalizeAnnotation(current));
  };

  const commitText = () => {
    if (!editingId) return;
    const target = annotations.find((item) => item.id === editingId);
    if (target) onCommit({ ...target, text: editText });
    setEditingId(null);
  };

  const armed = tool !== 'select';
  const rendered = annotations.map((item) => draft && draft.id === item.id ? draft : item);
  const drawing = draft && !annotations.some((item) => item.id === draft.id) ? draft : null;
  const arrows = [...rendered, ...(drawing ? [drawing] : [])].filter((item) => item.kind === 'arrow');

  const renderHandles = (annotation: Annotation) => {
    if (annotation.kind === 'arrow') {
      return (['start', 'end'] as const).map((handle) => <span key={handle} className="annotation-handle annotation-handle--endpoint"
        style={{ left: `${handle === 'start' ? annotation.x : annotation.x + annotation.w}%`, top: `${handle === 'start' ? annotation.y : annotation.y + annotation.h}%` }}
        onPointerDown={(event) => beginResize(event, annotation, handle)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />);
    }
    /* 끌어서 뒤집히는 중에도 핸들이 실제 테두리에 붙어 있도록 좌상단 기준으로 편다. */
    const left = Math.min(annotation.x, annotation.x + annotation.w);
    const top = Math.min(annotation.y, annotation.y + annotation.h);
    const width = Math.abs(annotation.w);
    const height = Math.abs(annotation.h);
    return BOX_HANDLES.map((handle) => <span key={handle} className={`annotation-handle annotation-handle--${handle}`}
      style={{ left: `${left + width * HANDLE_OFFSET[handle].x}%`, top: `${top + height * HANDLE_OFFSET[handle].y}%` }}
      onPointerDown={(event) => beginResize(event, annotation, handle)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />);
  };

  return <div ref={layerRef} className={`annotation-layer ${armed ? 'annotation-layer--armed' : ''} annotation-layer--${tool}`}
    onPointerDown={beginDraw} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}>

    {layerSize.width > 0 && arrows.length > 0 && <svg className="annotation-arrows" viewBox={`0 0 ${layerSize.width} ${layerSize.height}`} aria-hidden="true">
      {/* 화살촉은 marker 안에서 색을 물려받지 못해 쓰이는 색마다 하나씩 만든다. */}
      <defs>{[...new Set(arrows.map((arrow) => arrow.color || DEFAULT_ANNOTATION_COLOR))].map((hex) => (
        <marker key={hex} id={`adc-arrowhead-${hex.replace('#', '')}`} markerWidth="9" markerHeight="7" refX="8.2" refY="3.5" orient="auto">
          <polygon points="0 0, 9 3.5, 0 7" fill={hex} />
        </marker>
      ))}</defs>
      {arrows.map((arrow) => {
        const x1 = layerSize.width * arrow.x / 100; const y1 = layerSize.height * arrow.y / 100;
        const x2 = layerSize.width * (arrow.x + arrow.w) / 100; const y2 = layerSize.height * (arrow.y + arrow.h) / 100;
        const hex = arrow.color || DEFAULT_ANNOTATION_COLOR;
        return <g key={arrow.id}>
          <line className="annotation-arrow__line" style={{ stroke: hex }} x1={x1} y1={y1} x2={x2} y2={y2} markerEnd={`url(#adc-arrowhead-${hex.replace('#', '')})`} />
          {!armed && <line className="annotation-arrow__hit" x1={x1} y1={y1} x2={x2} y2={y2}
            onPointerDown={(event) => beginMove(event, arrow)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />}
        </g>;
      })}
    </svg>}

    {[...rendered, ...(drawing ? [drawing] : [])].map((annotation) => {
      if (annotation.kind === 'arrow') return null;
      const selected = selectedId === annotation.id && !drawing;
      const editing = editingId === annotation.id;
      const hex = annotation.color || DEFAULT_ANNOTATION_COLOR;
      const box = {
        left: `${Math.min(annotation.x, annotation.x + annotation.w)}%`, top: `${Math.min(annotation.y, annotation.y + annotation.h)}%`,
        width: `${Math.abs(annotation.w)}%`, height: `${Math.abs(annotation.h)}%`,
        ['--annot' as string]: hex, ['--annot-fill' as string]: withAlpha(hex, 0.22), ['--annot-glow' as string]: withAlpha(hex, 0.3),
      };
      return <div key={annotation.id} className={`annotation-shape annotation-shape--${annotation.kind} ${selected ? 'annotation-shape--selected' : ''}`} style={box}
        onPointerDown={(event) => beginMove(event, annotation)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}
        onDoubleClick={(event) => { if (annotation.kind !== 'text') return; event.stopPropagation(); setEditingId(annotation.id); setEditText(annotation.text || ''); }}>
        {annotation.kind === 'text' && (editing
          ? <textarea className="annotation-text__input" style={{ fontSize: `${annotation.fontSize ?? DEFAULT_TEXT_SIZE}px` }} value={editText} autoFocus onChange={(event) => setEditText(event.target.value)} onPointerDown={(event) => event.stopPropagation()}
              onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); commitText(); } else if (event.key === 'Escape') { event.preventDefault(); setEditingId(null); } }}
              onBlur={commitText} placeholder="가공 내용 입력" aria-label="주석 텍스트" />
          : <span className="annotation-text__value" style={{ fontSize: `${annotation.fontSize ?? DEFAULT_TEXT_SIZE}px` }}>{annotation.text || <em>더블클릭해 입력</em>}</span>)}
      </div>;
    })}

    {!armed && selectedId && !drawing && (() => {
      const target = rendered.find((item) => item.id === selectedId);
      if (!target) return null;
      const anchorX = target.kind === 'arrow' ? Math.max(target.x, target.x + target.w) : Math.min(target.x, target.x + target.w) + Math.abs(target.w);
      const anchorY = target.kind === 'arrow' ? Math.min(target.y, target.y + target.h) : Math.min(target.y, target.y + target.h);
      const size = target.fontSize ?? DEFAULT_TEXT_SIZE;
      /* 편집 중에도 크기를 바로 확인할 수 있도록 preventDefault 로 textarea 포커스를 지킨다. */
      const resize = (next: number) => onCommit({ ...target, fontSize: clamp(next, TEXT_SIZE_MIN, TEXT_SIZE_MAX) });
      /* 핸들은 도형의 자식이 아니라 변수가 상속되지 않으므로 여기서 직접 씌운다. */
      const selectedHex = target.color || DEFAULT_ANNOTATION_COLOR;
      return <div className="annotation-selection" style={{ ['--annot' as string]: selectedHex }}>
        {renderHandles(target)}
        <button type="button" className="annotation-delete" style={{ left: `${anchorX}%`, top: `${anchorY}%` }} onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => { event.stopPropagation(); onDelete(selectedId); onSelect(null); }} aria-label="이 주석 삭제" title="삭제 (Delete)"><X size={11} /></button>
        {target.kind === 'text' && <div className="annotation-fontsize" style={{ left: `${Math.min(target.x, target.x + target.w)}%`, top: `${Math.min(target.y, target.y + target.h) + Math.abs(target.h)}%` }}
          onPointerDown={(event) => { event.stopPropagation(); event.preventDefault(); }}>
          <button type="button" onClick={() => resize(size - TEXT_SIZE_STEP)} disabled={size <= TEXT_SIZE_MIN} aria-label="글자 작게" title="글자 작게">A<span>−</span></button>
          <span className="annotation-fontsize__value" aria-live="polite">{size}</span>
          <button type="button" onClick={() => resize(size + TEXT_SIZE_STEP)} disabled={size >= TEXT_SIZE_MAX} aria-label="글자 크게" title="글자 크게">A<span>+</span></button>
        </div>}
      </div>;
    })()}
  </div>;
}

function CorrectionPoints({ coefficient, points, labels = true, visibleLabelIds, onLabelToggle, overrides, onOverrideChange }: { coefficient: number; points: PointResult[]; labels?: boolean; visibleLabelIds?: Set<string>; onLabelToggle?: (id: string) => void; overrides?: Record<string, number>; onOverrideChange?: (id: string, value: number | null) => void }) {
  const labelHeight = 17;
  const displayFor = (point: PointResult) => overrides?.[point.id] !== undefined ? overrides[point.id]! : -(point.value * coefficient);
  const formatCorrection = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(1)}`;
  const getLabelWidth = (point: PointResult) => Math.max(24, formatCorrection(displayFor(point)).length * 5.2 + 8);
  const layerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: string; x: number; y: number; clientX: number; clientY: number; moved: boolean } | null>(null);
  const ignoreClickRef = useRef(false);
  const layoutKeyRef = useRef('');
  const [layerSize, setLayerSize] = useState({ width: 0, height: 0 });
  const [labelPositions, setLabelPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
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
    const layoutKey = `${labelHeight}:${points.map((point) => `${point.id}:${formatCorrection(displayFor(point)).length}`).join('|')}`;
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
  }, [layerSize, points, coefficient, labelHeight, overrides]);
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
  const startEdit = (id: string, currentValue: number) => {
    setEditingId(id);
    setEditValue(currentValue.toFixed(1));
  };
  const commitEdit = () => {
    if (!editingId || !onOverrideChange) { setEditingId(null); return; }
    const parsed = parseFloat(editValue);
    if (Number.isFinite(parsed)) onOverrideChange(editingId, parsed);
    setEditingId(null);
  };
  const cancelEdit = () => setEditingId(null);
  const resetOverride = () => {
    if (!editingId || !onOverrideChange) return;
    onOverrideChange(editingId, null);
    setEditingId(null);
  };
  return <div className="point-layer" ref={layerRef}>
    <svg className="point-leaders" aria-hidden="true" viewBox={`0 0 ${layerSize.width} ${layerSize.height}`}>{points.map((point) => {
       const position = labelPositions[point.id]; const visible = !visibleLabelIds || visibleLabelIds.has(point.id);
       if (!position || !visible) return null;
       const labelWidth = getLabelWidth(point);
       const isOverridden = overrides?.[point.id] !== undefined;
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
      return <line key={point.id} className={isOverridden ? 'point-leader--overridden' : undefined} x1={x1} y1={y1} x2={x2} y2={y2} />;
    })}</svg>{points.map((point) => {
    const display = displayFor(point);
    const labelVisible = !visibleLabelIds || visibleLabelIds.has(point.id);
    const position = labelPositions[point.id];
    const isOverridden = overrides?.[point.id] !== undefined;
    const isEditing = editingId === point.id;
    const editable = Boolean(onOverrideChange);
    const labelStyle = position ? { left: `${position.x - layerSize.width * point.x / 100}px`, top: `${position.y - layerSize.height * point.y / 100}px` } : undefined;
    const labelClasses = ['measure-point__label'];
    if (editable) labelClasses.push('measure-point__label--editable');
    if (isOverridden) labelClasses.push('measure-point__label--overridden');
    if (isEditing) labelClasses.push('measure-point__label--editing');
    return <div className={`measure-point ${display >= 0 ? 'measure-point--plus' : 'measure-point--minus'} ${onLabelToggle ? 'measure-point--interactive' : ''}`} style={{ left: `${point.x}%`, top: `${point.y}%` }} key={point.id}>
      <button type="button" className="measure-point__dot" onClick={() => onLabelToggle?.(point.id)} aria-label={`${point.id} 라벨 ${labelVisible ? '숨기기' : '표시하기'}`} aria-pressed={labelVisible} title={`${point.id} 편차 ${point.value > 0 ? '+' : ''}${point.value.toFixed(3)} · 점 클릭으로 표시 전환`} />
      {labels && labelVisible && position && (isEditing ? <span className={labelClasses.join(' ')} style={labelStyle}>
        <input type="text" inputMode="decimal" className="measure-point__label__input" value={editValue} onChange={(e) => setEditValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commitEdit(); } else if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); } }} onBlur={commitEdit} autoFocus onFocus={(e) => e.currentTarget.select()} aria-label={`${point.id} 보정치 편집`} />
        {isOverridden && <button type="button" className="measure-point__label__reset" onMouseDown={(e) => e.preventDefault()} onClick={resetOverride} aria-label="자동값으로 되돌리기" title="자동값으로 되돌리기">↺</button>}
      </span> : <span className={labelClasses.join(' ')} data-point-id={point.id} style={labelStyle} onPointerDown={(event) => beginLabelDrag(event, point.id)} onPointerMove={moveLabel} onPointerUp={endLabelDrag} onPointerCancel={endLabelDrag} onClick={() => { if (ignoreClickRef.current) { ignoreClickRef.current = false; return; } if (editable) startEdit(point.id, display); }} title={isOverridden ? `수정된 값 (계수 영향 없음) · 클릭하여 편집` : (editable ? '클릭하여 값 편집 · 드래그로 이동' : undefined)}>{formatCorrection(display)}</span>)}
    </div>;
  })}</div>;
}

function Sidebar({ view, setView, collapsed, setCollapsed, hasResult }: { view: View; setView: (view: View) => void; collapsed: boolean; setCollapsed: (value: boolean) => void; hasResult: boolean }) {
  const items = [
    { id: 'workspace' as const, label: '분석 작업실', icon: Grid2X2 },
    { id: 'results' as const, label: '엔진 결과', icon: BarChart3 },
    { id: 'service' as const, label: 'ADC 보정 시트', icon: Layers3 },
  ];
  return <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
    <div className="brand"><img className="brand__logo" src="/ajin-industrial-logo.png" alt="아진산업" /><div className="brand__copy"><strong>ADC</strong><span>Ajin Die Compensation</span></div></div>
    <nav className="sidebar__nav" aria-label="주 메뉴"><span className="sidebar__eyebrow">ADC WORKSPACE</span>{items.map((item) => { const Icon = item.icon; const disabled = item.id !== 'workspace' && !hasResult; return <button key={item.id} disabled={disabled} onClick={() => !disabled && setView(item.id)} className={view === item.id ? 'active' : ''}><Icon size={19} /><span>{item.label}</span></button>; })}</nav>
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

function Results({ scan, onService, hiddenPointIds, onPointToggle, onAllPointsToggle }: { scan: ScanItem; onService: () => void; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; onAllPointsToggle: (visible: boolean) => void }) {
  const [engine, setEngine] = useState<Engine>('label');
  const result = scan.result!; const meta = engineMeta[engine]; const summary = engineSummary(engine, result);
  const engineWarnings = result.warningsByEngine?.[engine] ?? (engine === 'zero' ? result.warnings : []);
  const visibleLabelIds = new Set(result.points.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id));
  const image = engine === 'zero' ? result.zeroOverlay : result.cleanImage || scan.url;
  const toggleLabel = onPointToggle;
  const allLabelsVisible = result.points.length > 0 && visibleLabelIds.size === result.points.length;
  return <section className="page page--results"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">분석 작업실 <ChevronRight size={14} /> {scan.partNo}</span><h2>엔진별 실제 분석 결과</h2><p>{scan.name} · {result.source.width} × {result.source.height}px</p></div><button className="primary-button" onClick={onService}>보정 시트 만들기 <ArrowRight size={17} /></button></div>
    <div className="result-tabs" role="tablist">{(Object.keys(engineMeta) as Engine[]).map((key, index) => { const item = engineMeta[key]; const failed = Boolean(result.errors[key]); return <button role="tab" aria-selected={engine === key} className={engine === key ? 'active' : ''} onClick={() => setEngine(key)} key={key}><span style={{ color: failed ? '#bd4650' : item.color }}>0{index + 1}</span><div><b>{item.name}</b><small>{failed ? '실행 오류' : item.short}</small></div>{!failed && <Check size={17} />}</button>; })}</div>
    <div className="results-layout"><div className="viewer-card card"><div className="viewer-toolbar"><div><span className={`status ${result.errors[engine] ? 'status--error' : 'status--done'}`}>{result.errors[engine] ? <><X size={13} /> 실행 실패</> : <><Check size={13} /> 실제 분석 완료</>}</span><b>{meta.name}</b></div>{engine === 'deviation' && <button className="tool-button" onClick={() => onAllPointsToggle(!allLabelsVisible)}>{allLabelsVisible ? <EyeOff size={14} /> : <Eye size={14} />} 라벨 전체 {allLabelsVisible ? 'OFF' : 'ON'}</button>}</div><div className={`viewer-stage ${engine === 'deviation' ? 'viewer-stage--light' : ''}`}><Heatmap key={`${scan.id}-${engine}`} imageUrl={image} width={result.source.width} height={result.source.height} lightBackground={engine === 'deviation'}>{engine === 'deviation' && <CorrectionPoints coefficient={-1} points={result.points} visibleLabelIds={visibleLabelIds} onLabelToggle={toggleLabel} />}</Heatmap></div><div className="viewer-legend"><span><i className="legend-dot" style={{ background: meta.color }} /> 현재 표시: {meta.name}</span><span>{engine === 'deviation' ? '라벨이나 포인트 점을 누르면 개별 표시를 켜고 끌 수 있습니다.' : '표시된 값과 위치는 업로드 이미지의 실제 엔진 결과입니다.'}</span></div></div>
      <aside className="inspection-panel"><div className="score-card card"><span className="score-card__icon" style={{ color: meta.color, background: `${meta.color}12` }}>{engine === 'label' ? <Sparkles /> : engine === 'deviation' ? <Activity /> : <Gauge />}</span><span>핵심 결과</span><strong style={{ color: result.errors[engine] ? '#bd4650' : meta.color }}>{summary.stat}</strong><p>{summary.detail}</p></div><div className="card plain-summary"><h3>쉽게 보는 결과</h3><div className="summary-line"><Check size={16} /><div><b>처리 방식</b><span>{engine === 'label' ? 'label_removal의 인페인팅 결과입니다.' : engine === 'deviation' ? '라벨 제거 이미지에 deviation_extraction의 지시선 끝점과 판독값을 겹쳐 표시합니다.' : 'zero_line_detection의 컬러바 기반 결과입니다.'}</span></div></div>{engineWarnings.length > 0 && <div className="summary-line warning"><MoveRight size={16} /><div><b>확인 필요</b><span>{engineWarnings[0]}</span></div></div>}</div><div className="card mini-table"><div className="card-title"><h3>검출 포인트</h3><span>라벨 {visibleLabelIds.size}/{result.points.length}</span></div>{result.points.map((point) => { const visible = visibleLabelIds.has(point.id); return <div className="point-list-row" key={point.id}><span>{point.id}</span><b className={point.value > 0 ? 'positive' : 'negative'}>{point.value > 0 ? '+' : ''}{point.value.toFixed(3)} mm</b><small>{point.xPx}, {point.yPx}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleLabel(point.id)} aria-label={`${point.id} 라벨 ${visible ? '숨기기' : '표시하기'}`} title={`라벨 ${visible ? 'OFF' : 'ON'}`}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}{!result.points.length && <p className="empty-mini">검출된 포인트가 없습니다.</p>}</div></aside>
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

function ServicePreview({ scan, folderAvailable, hiddenPointIds, onPointToggle, pointOverrides, onOverrideChange, onClearAllOverrides, annotations = [], setAnnotations }: { scan: ScanItem; folderAvailable: boolean; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; onClearAllOverrides: () => void; annotations: Annotation[]; setAnnotations: (updater: (current: Annotation[]) => Annotation[]) => void }) {
  const result = scan.result!; const points = result.points; const visiblePointIds = new Set(points.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id)); const [coefficient, setCoefficient] = useState(1); const [showPoints, setShowPoints] = useState(true); const [showZero, setShowZero] = useState(true);
  const [tool, setTool] = useState<AnnotationTool>('select'); const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null); const [showAnnotations, setShowAnnotations] = useState(true);
  const createAnnotation = (annotation: Annotation) => setAnnotations((current) => [...current, annotation]);
  const commitAnnotation = (annotation: Annotation) => setAnnotations((current) => current.map((item) => item.id === annotation.id ? annotation : item));
  const deleteAnnotation = (id: string) => setAnnotations((current) => current.filter((item) => item.id !== id));
  const clearAnnotations = () => { setAnnotations(() => []); setSelectedAnnotationId(null); setTool('select'); };
  /* 새 주석은 늘 기본색으로 그려지고, 색 변경은 주석을 고른 뒤 팔레트를 누르는 동작으로만 일어난다. */
  const selectedColor = selectedAnnotationId ? (annotations.find((item) => item.id === selectedAnnotationId)?.color ?? DEFAULT_ANNOTATION_COLOR) : null;
  const changeColor = (hex: string) => {
    if (!selectedAnnotationId) return;
    setAnnotations((current) => current.map((item) => item.id === selectedAnnotationId ? { ...item, color: hex } : item));
  };
  const displayFor = (point: PointResult) => pointOverrides[point.id] !== undefined ? pointOverrides[point.id] : -(point.value * coefficient);
  const maxCorrection = useMemo(() => points.length ? Math.max(...points.map((point) => Math.abs(displayFor(point)))) : 0, [coefficient, points, pointOverrides]);
  const overrideCount = useMemo(() => points.filter((point) => pointOverrides[point.id] !== undefined).length, [points, pointOverrides]);
  const baseImage = showZero && result.zeroOverlay ? result.zeroOverlay : result.cleanImage || scan.url;
  return <section className="page page--service"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">ADC · Ajin Die Compensation <span className="demo-badge">DEMO</span></span><h2>ADC 금형 보정 시트</h2><p>실제 라벨 제거 이미지에 검출된 편차값과 제로라인을 합성합니다.</p></div></div><div className="service-grid"><div className="correction-card card"><div className="viewer-toolbar"><div><span className="status status--done"><Check size={13} /> 실제 결과 합성</span><b>{scan.partNo} · 보정 작업 지시도</b></div><div className="layer-toggles"><button className={showPoints ? 'active orange' : ''} onClick={() => setShowPoints(!showPoints)}><i /> 보정치</button><button className={showZero ? 'active green' : ''} onClick={() => setShowZero(!showZero)} disabled={!result.zeroOverlay}><i /> 제로라인</button><button className={showAnnotations ? 'active amber' : ''} onClick={() => { setShowAnnotations(!showAnnotations); setTool('select'); setSelectedAnnotationId(null); }}><i /> 주석</button></div></div><SheetTitleBlock scan={scan} />{showAnnotations && <AnnotationToolbar tool={tool} setTool={(next) => { setTool(next); if (next !== 'select') setSelectedAnnotationId(null); }} hasAnnotations={annotations.length > 0} onClearAll={clearAnnotations} selectedColor={selectedColor} onColorChange={changeColor} />}<div className="sheet-stage sheet-stage--light"><Heatmap key={`${scan.id}-${showZero ? 'zero' : 'clean'}`} imageUrl={baseImage} width={result.source.width} height={result.source.height} lightBackground>{showAnnotations && <AnnotationLayer annotations={annotations} tool={tool} setTool={setTool} selectedId={selectedAnnotationId} onSelect={setSelectedAnnotationId} onCommit={commitAnnotation} onCreate={createAnnotation} onDelete={deleteAnnotation} />}{showPoints && <CorrectionPoints coefficient={coefficient} points={points} visibleLabelIds={visiblePointIds} onLabelToggle={onPointToggle} overrides={pointOverrides} onOverrideChange={onOverrideChange} />}</Heatmap><div className="sheet-stamp"><span>AJIN INDUSTRIAL</span><b>DIE CORRECTION SHEET</b><small>{scan.partNo} · REV.01</small></div></div><div className="sheet-note"><ShieldCheck size={17} /><span><b>검토용 가상 보정치입니다.</b> 실제 가공 전 담당자 승인과 현장 검증이 필요합니다.</span></div></div><aside className="control-panel"><div className="card coefficient-card"><div className="card-title"><div><h3>보정 계수</h3><p>편차값에 곱할 비율을 조절합니다.</p></div><span>{coefficient.toFixed(2)}×</span></div><div className="coefficient-input"><input aria-label="보정 계수 직접 입력" type="number" min="0.5" max="1.5" step="0.01" value={coefficient} onChange={(e) => { const value = e.target.valueAsNumber; if (!Number.isNaN(value)) setCoefficient(Math.max(0.5, Math.min(1.5, value))); }} /><span>×</span></div><input aria-label="보정 계수" type="range" min="0.5" max="1.5" step="0.05" value={coefficient} onChange={(e) => setCoefficient(Number(e.target.value))} /><div className="range-labels"><span>보수적 0.50</span><span>기준 1.00</span><span>적극적 1.50</span></div><div className="formula"><span>보정치</span><b>= 편차 × {coefficient.toFixed(2)} × (−1)</b></div>{overrideCount > 0 && <p className="coefficient-note">수정된 {overrideCount}개 포인트는 계수 영향을 받지 않습니다.</p>}</div><div className="card correction-summary"><h3>실제 엔진 요약</h3><div><span>보정 포인트</span><b>{visiblePointIds.size}개</b></div>{overrideCount > 0 && <div><span>수정된 포인트</span><b className="blue">{overrideCount}개</b></div>}<div><span>최대 보정량</span><b className="orange">{maxCorrection.toFixed(3)} mm</b></div><div><span>제로라인</span><b className="green">{result.stats.zeroRegions}개 영역</b></div><div><span>처리 품번</span><b>{scan.partNo}</b></div>{overrideCount > 0 && <button type="button" className="reset-all-overrides" onClick={onClearAllOverrides}>모든 수정 취소</button>}</div></aside></div>{folderAvailable && <Explorer />}</section>;
}

export default function Home() {
  const [view, setView] = useState<View>('workspace'); const [scans, setScans] = useState<ScanItem[]>([]); const [activeId, setActiveId] = useState<string>(); const [collapsed, setCollapsed] = useState(false); const [backendOnline, setBackendOnline] = useState<boolean | null>(null); const [folderAvailable, setFolderAvailable] = useState(false); const [hiddenPointIdsByScan, setHiddenPointIdsByScan] = useState<Record<string, Set<string>>>({}); const [pointOverridesByScan, setPointOverridesByScan] = useState<Record<string, Record<string, number>>>({}); const [annotationsByScan, setAnnotationsByScan] = useState<Record<string, Annotation[]>>({});
  useEffect(() => { fetch(`${API_BASE}/api/health`).then((response) => response.json()).then((data) => { setBackendOnline(Boolean(data.ok)); setFolderAvailable(Boolean(data.folderAvailable)); }).catch(() => setBackendOnline(false)); }, []);
  const resolvedActiveId = activeId || scans[0]?.id;
  const activeScan = scans.find((scan) => scan.id === resolvedActiveId); const completedScan = activeScan?.result ? activeScan : scans.find((scan) => scan.result); const hasResult = Boolean(completedScan?.result);
  const hiddenPointIds = completedScan ? hiddenPointIdsByScan[completedScan.id] || new Set<string>() : new Set<string>();
  const pointOverrides = completedScan ? pointOverridesByScan[completedScan.id] || {} : {};
  const togglePoint = (id: string) => completedScan && setHiddenPointIdsByScan((current) => { const next = new Set(current[completedScan.id] || []); if (next.has(id)) next.delete(id); else next.add(id); return { ...current, [completedScan.id]: next }; });
  const setAllPointsVisible = (visible: boolean) => completedScan && setHiddenPointIdsByScan((current) => ({ ...current, [completedScan.id]: visible ? new Set() : new Set(completedScan.result!.points.map((point) => point.id)) }));
  const setPointOverride = (id: string, value: number | null) => completedScan && setPointOverridesByScan((current) => { const next = { ...(current[completedScan.id] || {}) }; if (value === null) delete next[id]; else next[id] = value; return { ...current, [completedScan.id]: next }; });
  const clearAllOverrides = () => completedScan && setPointOverridesByScan((current) => ({ ...current, [completedScan.id]: {} }));
  const annotations = completedScan ? annotationsByScan[completedScan.id] || [] : [];
  const setAnnotations = (updater: (current: Annotation[]) => Annotation[]) => completedScan && setAnnotationsByScan((current) => ({ ...current, [completedScan.id]: updater(current[completedScan.id] || []) }));
  const openResults = (id: string) => { setActiveId(id); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const selectView = (next: View) => { if (next === 'workspace' || hasResult) setView(next); };
  return <main className={`app-shell ${collapsed ? 'app-shell--collapsed' : ''}`}><Sidebar view={view} setView={selectView} collapsed={collapsed} setCollapsed={setCollapsed} hasResult={hasResult} /><div className="app-main"><Header scans={scans} activeId={resolvedActiveId} setActiveId={setActiveId} />{view === 'workspace' && <Workspace scans={scans} setScans={setScans} onOpenResults={openResults} backendOnline={backendOnline} />}{view === 'results' && completedScan?.result && <Results scan={completedScan} onService={() => setView('service')} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} onAllPointsToggle={setAllPointsVisible} />}{view === 'service' && completedScan?.result && <ServicePreview scan={completedScan} folderAvailable={folderAvailable} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} pointOverrides={pointOverrides} onOverrideChange={setPointOverride} onClearAllOverrides={clearAllOverrides} annotations={annotations} setAnnotations={setAnnotations} />}</div></main>;
}
