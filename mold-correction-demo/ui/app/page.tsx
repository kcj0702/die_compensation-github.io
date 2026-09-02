'use client';

/* Blob/data URLs are local engine outputs and cannot use Next's remote image loader. */
/* eslint-disable @next/next/no-img-element */

import { Activity, ArrowLeft, ArrowRight, ArrowUpRight, BarChart3, Box, Check, ChevronDown, ChevronRight, Circle, CircleHelp, Crosshair, Download, Eye, EyeOff, File, Folder, FolderOpen, Gauge, Grid2X2, Image as ImageIcon, Layers3, ListFilter, Maximize2, MousePointer2, MoveRight, PanelLeftClose, Play, Printer, Settings2, ShieldCheck, Sparkles, Square, Trash2, Type, UploadCloud, X, ZoomIn, ZoomOut } from 'lucide-react';
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react';

import { clearSession, downloadSession, emptySession, loadSession, readSessionFile, saveSession, type SessionSnapshot } from './session-store';
import { CIRCLED, CadViewer, type CadMesh, type CadMorph, type CadNote, type CadOverlay, type CadRegion, type CadSection } from './cad-viewer';

const API_BASE = 'http://127.0.0.1:8000';

type View = 'workspace' | 'results' | 'service' | 'cad';
type Engine = 'label' | 'deviation' | 'zero';
type ScanStatus = 'ready' | 'analyzing' | 'done' | 'error';
/* source 가 'colormap' 이면 작업자가 찍은 추정 포인트다. 라벨을 읽어 얻은 실측값과
   섞이지 않도록 화면에서도 구분해 보여준다. */
type PointResult = { id: string; xPx: number; yPx: number; x: number; y: number; value: number; labelColor: string; confidence: string; source?: 'colormap' };
type ZeroAnchor = { anchor_id: number; x: number; y: number; boundary_arclen: number; source?: string; kind?: 'point' | 'zone'; strength?: number };
type ValleyLine = { id: string; anchorStartId: number | null; anchorEndId: number | null; points: [number, number][]; length_px: number; mean_abs_deviation: number; source: 'ai' | 'manual' };
// /api/zero-valley-line 응답 — 백엔드는 snake_case 로 준다
type ValleyLineResponse = { anchor_start_id: number; anchor_end_id: number; points: [number, number][]; length_px: number; mean_abs_deviation: number };
type AdvanceLine = { points: [number, number][]; warnings: string[]; confidence: 'high' | 'low' };
type ReferenceLine = { kind: 'line' | 'areas'; points: [number, number][]; contours: [number, number][][]; partNo: string; sourceSheet: string; mirrored: boolean };
// 현업 방식(녹색 영역 x 부호 전환대)으로 찾은 영라인 후보 구간
// 주요 0포인트를 직선으로 이은 영라인 — 곡선 없이 꺾임 최대 1개
type SimpleZeroLine = { line_id: number; points: [number, number][]; route_type: string; bend_count: number; combined_coverage: number; tolerance_coverage: number; product_coverage: number; support_count: number; length_px: number };
// my_lab 파이프라인이 그린 영라인 — 데모 화면의 기본 표시다
type LabShape = { shape_id: number; points: [number, number][]; is_closed: boolean };
type LabDistance = { to_lab_pct: number; to_predicted_pct: number; diagonal_px: number };
type GreenBelt = { belt_id: number; contour: [number, number][]; center: [number, number]; length_px: number; area_px: number; mean_abs_deviation: number };
type ZeroPointCluster = { cluster_id: number; loop: string; kind: 'point' | 'zone'; center: [number, number]; members: [number, number][]; contour: [number, number][]; strength: number; span: number };
type LabelZeroLine = { points: [number, number][]; length_px: number; mean_abs_deviation: number };
/* 보정시트에서 제로라인을 손본 내역. 3D 는 이걸 그대로 따른다 —
   시트가 제로라인의 주인이고 3D 는 그 결과를 보여 주는 자리다.
   옮김은 **그림 좌표(px)** 다. 3D 좌표로 밀면 표면에서 뜬다. */
export type ZeroEdit = { index: number; dx: number; dy: number; hidden?: boolean };
/* 작업자가 손으로 맞춘 정렬. 자동 정합은 실루엣만 보므로 몇 퍼센트가
   모자랄 수 있다 — 그때 조금 돌리고 옮겨 맞춘다. */
export type FitAdjust = { angle: number; dx: number; dy: number; scale: number };
const NO_ADJUST: FitAdjust = { angle: 0, dx: 0, dy: 0, scale: 1 };
type ZeroLineCandidate = { rank: number; anchor_start_id: number; anchor_end_id: number; points: [number, number][]; length_px: number; mean_abs_deviation: number; separation: number; balance: number; score: number };
type AnalysisResult = {
  analysisId: string | null;
  partNo?: string;
  knownParts?: string[];
  /* 현업 파일명 규칙에서 읽어낸 것들 — 차종_품번_품명_공정_날짜.
     보정시트 머리말을 이걸로 채운다. 못 읽은 칸은 null 이다. */
  naming?: {
    part_no: string | null; maker: string | null; part_name: string | null;
    process: string | null; applied_at: string | null; control_no: string | null;
  };
  /* 보정시트에 적을 만한 포인트만 골라낸 것. 스캔에는 백여 개가 찍히지만
     현업 시트에 적히는 건 열몇 개다. */
  keyPoints?: { point_id: string; x_px: number; y_px: number; value: number; score: number; reason: string }[];
  /* 현업 파이프라인(lab_pipeline)이 만든 제로라인. 허용범위 밖 영역을
     윤곽 위 제로포인트 둘로 닫은 선이라 근거가 가장 분명하다. */
  labZeroLines?: [number, number][][];
  /* 제로 **영역**(67XX6). 서버가 이미 네모로 다듬어 보낸다 —
     시트와 3D 가 같은 도형을 그리려고 한 군데서 만든다. */
  labZeroAreas?: [number, number][][];
  labZeroRegions?: { label: string; area: number; status: string;
                     zeroPoints: string[]; attempts: number; coverage: number }[];
  keyPointsRejected?: { id: string; value: number }[];
  source: { name: string; width: number; height: number };
  cleanImage: string | null;
  zeroOverlay: string | null;
  zeroMask: string | null;
  zeroAnchors: ZeroAnchor[];
  advanceLine: AdvanceLine | null;
  zeroLineCandidates: ZeroLineCandidate[];
  zeroPointClusters: ZeroPointCluster[];
  greenBelts: GreenBelt[];
  simpleZeroLines: SimpleZeroLine[];
  labProfile: LabShape[];
  labDistance: LabDistance | null;
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
type FolderResponse = { available?: boolean; rootName?: string; path?: string; entries?: FolderEntry[]; error?: string };
type HealthResponse = { ok?: boolean; folderAvailable?: boolean };

/* 보정 시트 주석 — 좌표와 크기는 모두 이미지 대비 %라 확대/축소와 창 크기에 영향받지 않는다. */
type AnnotationKind = 'rect' | 'ellipse' | 'text' | 'arrow';
type AnnotationTool = 'select' | AnnotationKind;
/* 사각형·타원·텍스트는 x,y 가 좌상단이고 w,h 가 크기다. 화살표는 x,y 가 시작점이고 w,h 가 끝점까지의 변위라 음수가 될 수 있다. */
type Annotation = { id: string; kind: AnnotationKind; x: number; y: number; w: number; h: number; text?: string; fontSize?: number; color?: string };
type DetailRegion = { id: string; x: number; y: number; w: number; h: number; label: string };
type SheetLayout = { id: string; kind: 'front' | 'detail'; x: number; y: number; w: number; h: number; regionId?: string };

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
      if (scale <= 1 || (event.target as Element).closest('.measure-point, .anchor-point, .zoom-controls, .annotation-layer--armed, .annotation-shape, .annotation-handle, .annotation-delete, .annotation-fontsize, .annotation-arrow__hit')) return;
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
let detailSeq = 0;
const nextDetailId = () => `detail-${Date.now().toString(36)}-${++detailSeq}`;

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

function AnnotationToolbar({ tool, setTool, hasAnnotations, onClearAll, selectedColor, onColorChange, detailMode, onDetailMode, labelAreaMode, onLabelAreaMode, addPointMode, onAddPointMode }: { tool: AnnotationTool; setTool: (tool: AnnotationTool) => void; hasAnnotations: boolean; onClearAll: () => void; selectedColor: string | null; onColorChange: (hex: string) => void; detailMode?: boolean; onDetailMode?: () => void; labelAreaMode?: 'hide' | 'show' | null; onLabelAreaMode?: (mode: 'hide' | 'show') => void; addPointMode?: boolean; onAddPointMode?: () => void }) {
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
    {onDetailMode && <button type="button" className={detailMode ? 'active detail' : ''} onClick={onDetailMode} aria-pressed={detailMode} aria-label="Detail View 영역 만들기" title="Detail View 영역 만들기"><ZoomIn size={14} /></button>}
    {onLabelAreaMode && <button type="button" className={labelAreaMode === 'hide' ? 'active hide-area' : ''} onClick={() => onLabelAreaMode('hide')} aria-pressed={labelAreaMode === 'hide'} aria-label="영역 내 라벨 숨기기" title="영역 내 라벨 숨기기"><EyeOff size={14} /></button>}
    {onLabelAreaMode && <button type="button" className={labelAreaMode === 'show' ? 'active show-area' : ''} onClick={() => onLabelAreaMode('show')} aria-pressed={labelAreaMode === 'show'} aria-label="영역 내 라벨 보이기" title="영역 내 라벨 보이기"><Eye size={14} /></button>}
    {onAddPointMode && <button type="button" className={addPointMode ? 'active add-point' : ''} onClick={onAddPointMode} aria-pressed={addPointMode} aria-label="보정 포인트 추가" title="보정 포인트 추가 — 부품 위를 누르면 그 자리의 편차값을 색에서 추정합니다"><Crosshair size={14} /></button>}
    {onDetailMode && <span className="annotation-toolbar__divider" />}
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

const MIN_DETAIL_SIZE = 5;
const MIN_LAYOUT_SIZE = 14;
const SHEET_ASPECT = 1.414;

function normalizeBox<T extends { x: number; y: number; w: number; h: number }>(box: T, minimum: number): T {
  let { x, y, w, h } = box;
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  w = clamp(w, minimum, 100); h = clamp(h, minimum, 100);
  x = clamp(x, 0, 100 - w); y = clamp(y, 0, 100 - h);
  return { ...box, x, y, w, h };
}

function fitAspectSize(imageAspect: number, maxW: number, maxH: number) {
  let w = maxW; let h = w * SHEET_ASPECT / imageAspect;
  if (h > maxH) { h = maxH; w = h * imageAspect / SHEET_ASPECT; }
  return { w, h };
}

function DetailRegionLayer({ regions, active, selectedId, onSelect, onCreate, onChange, onDelete }: { regions: DetailRegion[]; active: boolean; selectedId: string | null; onSelect: (id: string | null) => void; onCreate: (region: DetailRegion) => void; onChange: (region: DetailRegion) => void; onDelete: (id: string) => void }) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [draft, setDraftState] = useState<DetailRegion | null>(null);
  const draftRef = useRef<DetailRegion | null>(null);
  const opRef = useRef<{ mode: 'draw' | 'move' | 'resize'; startX: number; startY: number; origin: DetailRegion; handle?: string } | null>(null);
  const setDraft = (value: DetailRegion | null) => { draftRef.current = value; setDraftState(value); };
  const toPercent = (clientX: number, clientY: number) => {
    const bounds = layerRef.current?.getBoundingClientRect();
    if (!bounds?.width || !bounds.height) return { x: 0, y: 0 };
    return { x: (clientX - bounds.left) / bounds.width * 100, y: (clientY - bounds.top) / bounds.height * 100 };
  };
  const beginDraw = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!active || event.button !== 0 || (event.target as Element).closest('.detail-region')) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    const region: DetailRegion = { id: nextDetailId(), x: start.x, y: start.y, w: 0, h: 0, label: `DETAIL ${String.fromCharCode(65 + regions.length)}` };
    opRef.current = { mode: 'draw', startX: start.x, startY: start.y, origin: region };
    setDraft(region); capturePointer(event.currentTarget, event.pointerId);
  };
  const beginEdit = (event: React.PointerEvent<Element>, region: DetailRegion, mode: 'move' | 'resize', handle?: string) => {
    if (active || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const start = toPercent(event.clientX, event.clientY);
    opRef.current = { mode, startX: start.x, startY: start.y, origin: region, handle };
    onSelect(region.id); setDraft(region); capturePointer(event.currentTarget, event.pointerId);
  };
  const handleMove = (event: React.PointerEvent<Element>) => {
    const op = opRef.current;
    if (!op) return;
    const now = toPercent(event.clientX, event.clientY); const dx = now.x - op.startX; const dy = now.y - op.startY;
    if (op.mode === 'draw') { setDraft({ ...op.origin, w: clamp(now.x, 0, 100) - op.origin.x, h: clamp(now.y, 0, 100) - op.origin.y }); return; }
    if (op.mode === 'move') { setDraft({ ...op.origin, x: clamp(op.origin.x + dx, 0, 100 - op.origin.w), y: clamp(op.origin.y + dy, 0, 100 - op.origin.h) }); return; }
    const handle = op.handle || 'se'; let { x, y, w, h } = op.origin;
    if (handle.includes('n')) { y += dy; h -= dy; } if (handle.includes('s')) h += dy;
    if (handle.includes('w')) { x += dx; w -= dx; } if (handle.includes('e')) w += dx;
    setDraft({ ...op.origin, x, y, w, h });
  };
  const endOperation = (event: React.PointerEvent<Element>) => {
    const op = opRef.current; const current = draftRef.current; opRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDraft(null); if (!op || !current) return;
    const normalized = normalizeBox(op.mode === 'draw' && Math.abs(current.w) < MIN_DETAIL_SIZE && Math.abs(current.h) < MIN_DETAIL_SIZE ? { ...current, w: 18, h: 18 } : current, MIN_DETAIL_SIZE);
    if (op.mode === 'draw') onCreate(normalized); else onChange(normalized);
  };
  const rendered = regions.map((region) => draft?.id === region.id ? draft : region);
  const drawing = draft && !regions.some((region) => region.id === draft.id) ? draft : null;
  return <div ref={layerRef} className={`detail-region-layer ${active ? 'detail-region-layer--active' : ''}`} onPointerDown={beginDraw} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}>
    {[...rendered, ...(drawing ? [drawing] : [])].map((region) => {
      const box = normalizeBox(region, draft?.id === region.id ? 0 : MIN_DETAIL_SIZE); const selected = selectedId === region.id && !active;
      return <div key={region.id} className={`detail-region ${selected ? 'selected' : ''}`} style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.w}%`, height: `${box.h}%` }} onPointerDown={(event) => beginEdit(event, region, 'move')} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation}>
        <span>{region.label}</span>
        {selected && BOX_HANDLES.map((handle) => <i key={handle} className={`layout-resize-handle layout-resize-handle--${handle}`} style={{ left: `${HANDLE_OFFSET[handle].x * 100}%`, top: `${HANDLE_OFFSET[handle].y * 100}%` }} onPointerDown={(event) => beginEdit(event, region, 'resize', handle)} onPointerMove={handleMove} onPointerUp={endOperation} onPointerCancel={endOperation} />)}
        {selected && <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onDelete(region.id); }} aria-label={`${region.label} 삭제`}><X size={10} /></button>}
      </div>;
    })}
  </div>;
}

function LabelAreaSelector({ mode, points, onApply, onComplete }: { mode: 'hide' | 'show' | null; points: PointResult[]; onApply: (ids: string[], mode: 'hide' | 'show') => void; onComplete: () => void }) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [draft, setDraftState] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const draftRef = useRef<{ x: number; y: number; w: number; h: number } | null>(null);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const setDraft = (next: { x: number; y: number; w: number; h: number } | null) => { draftRef.current = next; setDraftState(next); };
  const toPercent = (clientX: number, clientY: number) => {
    const bounds = layerRef.current?.getBoundingClientRect();
    if (!bounds?.width || !bounds.height) return { x: 0, y: 0 };
    return { x: clamp((clientX - bounds.left) / bounds.width * 100, 0, 100), y: clamp((clientY - bounds.top) / bounds.height * 100, 0, 100) };
  };
  const begin = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!mode || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation(); const start = toPercent(event.clientX, event.clientY);
    startRef.current = start; setDraft({ ...start, w: 0, h: 0 }); capturePointer(event.currentTarget, event.pointerId);
  };
  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = startRef.current; if (!start) return; const now = toPercent(event.clientX, event.clientY);
    setDraft({ ...start, w: now.x - start.x, h: now.y - start.y });
  };
  const end = (event: React.PointerEvent<HTMLDivElement>) => {
    const current = draftRef.current; startRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDraft(null); if (!current) return;
    const box = normalizeBox(current, 0);
    if (mode && box.w >= 1 && box.h >= 1) onApply(points.filter((point) => point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h).map((point) => point.id), mode);
    onComplete();
  };
  const box = draft ? normalizeBox(draft, 0) : null;
  return <div ref={layerRef} className={`label-area-selector ${mode ? `active label-area-selector--${mode}` : ''}`} onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={end}>
    {box && <div className="label-area-selector__box" style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.w}%`, height: `${box.h}%` }}><span>라벨 {mode === 'show' ? '표시' : '숨김'} 영역</span></div>}
  </div>;
}

function SheetLayoutFrame({ layout, imageAspect, selected, onSelect, onChange, onDelete, title, children }: { layout: SheetLayout; imageAspect: number; selected: boolean; onSelect: () => void; onChange: (layout: SheetLayout) => void; onDelete?: () => void; title: string; children: React.ReactNode }) {
  const frameRef = useRef<HTMLDivElement>(null);
  const draftRef = useRef<SheetLayout | null>(null);
  const [draft, setDraftState] = useState<SheetLayout | null>(null);
  const opRef = useRef<{ mode: 'move' | 'resize'; pointerId: number; clientX: number; clientY: number; origin: SheetLayout; handle?: string } | null>(null);
  const setDraft = (value: SheetLayout | null) => { draftRef.current = value; setDraftState(value); };
  const begin = (event: React.PointerEvent<Element>, mode: 'move' | 'resize', handle?: string) => {
    if (event.button !== 0) return; event.preventDefault(); event.stopPropagation(); onSelect();
    opRef.current = { mode, pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, origin: layout, handle };
    setDraft(layout); capturePointer(event.currentTarget, event.pointerId);
  };
  const move = (event: React.PointerEvent<Element>) => {
    const op = opRef.current; const canvas = frameRef.current?.parentElement?.getBoundingClientRect(); if (!op || !canvas) return;
    const dx = (event.clientX - op.clientX) / canvas.width * 100; const dy = (event.clientY - op.clientY) / canvas.height * 100;
    if (op.mode === 'move') { setDraft({ ...op.origin, x: clamp(op.origin.x + dx, 0, 100 - op.origin.w), y: clamp(op.origin.y + dy, 0, 100 - op.origin.h) }); return; }
    const handle = op.handle || 'se'; let { x, y, w, h } = op.origin;
    const horizontal = handle.includes('e') || handle.includes('w');
    const vertical = handle.includes('n') || handle.includes('s');
    const rawW = handle.includes('w') ? op.origin.w - dx : handle.includes('e') ? op.origin.w + dx : op.origin.w;
    const rawH = handle.includes('n') ? op.origin.h - dy : handle.includes('s') ? op.origin.h + dy : op.origin.h;
    if (horizontal && (!vertical || Math.abs(dx) >= Math.abs(dy))) {
      const sign = Math.sign(rawW || 1); const minW = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * imageAspect / (canvas.width / canvas.height)); const maxW = Math.min(100, 100 * imageAspect / (canvas.width / canvas.height));
      w = clamp(Math.abs(rawW), minW, maxW) * sign; h = Math.abs(w) * canvas.width / canvas.height / imageAspect * sign;
    } else {
      const sign = Math.sign(rawH || 1); const minH = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * (canvas.width / canvas.height) / imageAspect); const maxH = Math.min(100, 100 * (canvas.width / canvas.height) / imageAspect);
      h = clamp(Math.abs(rawH), minH, maxH) * sign; w = Math.abs(h) * canvas.height / canvas.width * imageAspect * sign;
    }
    if (handle.includes('w')) x = op.origin.x + op.origin.w - w;
    if (handle.includes('n')) y = op.origin.y + op.origin.h - h;
    setDraft({ ...op.origin, x, y, w, h });
  };
  const end = (event: React.PointerEvent<Element>) => {
    const current = draftRef.current; opRef.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDraft(null); if (current) onChange(normalizeBox(current, MIN_LAYOUT_SIZE));
  };
  const shown = draft ? normalizeBox(draft, 0) : layout;
  return <article ref={frameRef} className={`sheet-layout sheet-layout--${layout.kind} ${selected ? 'selected' : ''}`} style={{ left: `${shown.x}%`, top: `${shown.y}%`, width: `${shown.w}%`, height: `${shown.h}%` }} onPointerDown={onSelect}>
    <div className="sheet-layout__bar" onPointerDown={(event) => begin(event, 'move')} onPointerMove={move} onPointerUp={end} onPointerCancel={end}><span>{title}</span><small>비율 고정 · 드래그 이동</small>{onDelete && <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onDelete(); }} aria-label={`${title} 삭제`}><X size={11} /></button>}</div>
    <div className="sheet-layout__content">{children}</div>
    {selected && BOX_HANDLES.map((handle) => <i key={handle} className={`layout-resize-handle layout-resize-handle--${handle}`} style={{ left: `${HANDLE_OFFSET[handle].x * 100}%`, top: `${HANDLE_OFFSET[handle].y * 100}%` }} onPointerDown={(event) => begin(event, 'resize', handle)} onPointerMove={move} onPointerUp={end} onPointerCancel={end} />)}
  </article>;
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
  const previousLayerSizeRef = useRef({ width: 0, height: 0 });
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
    const previous = previousLayerSizeRef.current;
    if (previous.width > 0 && previous.height > 0 && (previous.width !== layerSize.width || previous.height !== layerSize.height)) {
      setLabelPositions((current) => Object.fromEntries(Object.entries(current).map(([id, position]) => [id, { x: position.x * layerSize.width / previous.width, y: position.y * layerSize.height / previous.height }])));
    }
    previousLayerSizeRef.current = layerSize;
  }, [layerSize]);
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
      y: Math.max(-96, Math.min(layer.clientHeight + 64, drag.y + (event.clientY - drag.clientY) / scale)),
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
    if (point.source === 'colormap') labelClasses.push('measure-point__label--estimated');
    return <div className={`measure-point ${display >= 0 ? 'measure-point--plus' : 'measure-point--minus'} ${onLabelToggle ? 'measure-point--interactive' : ''} ${labelVisible ? '' : 'measure-point--hidden'} ${point.source === 'colormap' ? 'measure-point--estimated' : ''}`} style={{ left: `${point.x}%`, top: `${point.y}%` }} key={point.id}>
      <button type="button" className="measure-point__dot" onClick={() => onLabelToggle?.(point.id)} aria-label={`${point.id} 라벨 ${labelVisible ? '숨기기' : '표시하기'}`} aria-pressed={labelVisible} title={`${point.id} 편차 ${point.value > 0 ? '+' : ''}${point.value.toFixed(3)} · 점 클릭으로 표시 전환`} />
      {labels && labelVisible && position && (isEditing ? <span className={labelClasses.join(' ')} style={labelStyle}>
        <input type="text" inputMode="decimal" className="measure-point__label__input" value={editValue} onChange={(e) => setEditValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commitEdit(); } else if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); } }} onBlur={commitEdit} autoFocus onFocus={(e) => e.currentTarget.select()} aria-label={`${point.id} 보정치 편집`} />
        {isOverridden && <button type="button" className="measure-point__label__reset" onMouseDown={(e) => e.preventDefault()} onClick={resetOverride} aria-label="자동값으로 되돌리기" title="자동값으로 되돌리기">↺</button>}
      </span> : <span className={labelClasses.join(' ')} data-point-id={point.id} style={labelStyle} onPointerDown={(event) => beginLabelDrag(event, point.id)} onPointerMove={moveLabel} onPointerUp={endLabelDrag} onPointerCancel={endLabelDrag} onClick={() => { if (ignoreClickRef.current) { ignoreClickRef.current = false; return; } if (editable) startEdit(point.id, display); }} title={isOverridden ? `수정된 값 (계수 영향 없음) · 클릭하여 편집` : (editable ? '클릭하여 값 편집 · 드래그로 이동' : undefined)}>{formatCorrection(display)}</span>)}
    </div>;
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

function LabProfileOverlay({ shapes, width, height }: { shapes: LabShape[]; width: number; height: number }) {
  // my_lab 파이프라인이 그린 제로라인. 데모에서는 이것을 결과로 보여준다
  // (feat/product-zero-line-profiles 의 표기를 그대로 따른다 —
  //  빨간 점선 외곽선, 닫힌 면은 옅은 빨강 채움).
  if (!shapes.length) return null;
  const strokeWidth = Math.max(width, height) * 0.0028;
  const dash = `${strokeWidth * 3.5} ${strokeWidth * 2.5}`;
  return <svg className="approved-profile-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {shapes.map((shape) => {
      const points = shape.points.map(([x, y]) => `${x},${y}`).join(' ');
      const halo = { points, fill: 'none', stroke: '#ffffff', strokeWidth: strokeWidth * 2.8,
        opacity: 0.85, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
      const line = { points, fill: 'none', stroke: '#eb3737', strokeWidth, strokeDasharray: dash,
        strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
      return <g key={shape.shape_id}>
        {shape.is_closed && <polygon points={points} fill="rgba(255,45,45,0.20)" stroke="none" />}
        {shape.is_closed ? <polygon {...halo} /> : <polyline {...halo} />}
        {shape.is_closed ? <polygon {...line} /> : <polyline {...line} />}
      </g>;
    })}
  </svg>;
}

/* 현업 파이프라인 결과를 겹침층이 아는 모양으로 바꾼다.
   그 선은 허용범위 밖 영역을 윤곽 위 제로포인트 둘로 닫은 것이라
   근거가 가장 분명하다 — 있으면 이걸 먼저 쓴다. */
/** 이 스캔이 이 CAD 의 짝인가. CAD 파일과 스캔의 품번 규칙이 다르다 —
 *  64XX1 도면이 64XX2 스캔의 짝이다. */
export function scanFitsCad(scan: ScanItem, mesh: CadMesh) {
  const name = (mesh.summary.name || '').toUpperCase().replace(/[-_]/g, '');
  const pairs: [string, string][] = [
    ['64XX1', '64XX2'], ['71XX1', '71XX2'], ['67XX6', '67XX6'],
  ];
  const wanted = pairs.find(([cad]) => name.includes(cad))?.[1];
  const part = (scan.partNo || '').toUpperCase().replace(/[-_]/g, '');
  return wanted ? part.includes(wanted) : Boolean(part && name.includes(part));
}

function zeroLinesToShow(result: AnalysisResult): SimpleZeroLine[] {
  const lab = result.labZeroLines ?? [];
  if (lab.length) {
    return lab.map((points, index) => ({
      line_id: index + 1, points,
      tolerance_coverage: 1, length_px: 0, mean_abs_deviation: 0,
    } as unknown as SimpleZeroLine));
  }
  return result.simpleZeroLines || [];
}

/** 손본 내역을 제로라인에 적용한다. 백엔드가 3D 에 하는 것과 같은 식이라
 *  시트에서 보는 자리와 3D 의 자리가 어긋나지 않는다. */
function applyZeroEdits(lines: SimpleZeroLine[], edits: ZeroEdit[]) {
  if (!edits.length) return lines;
  const out: SimpleZeroLine[] = [];
  lines.forEach((line, index) => {
    const edit = edits.find((e) => e.index === index);
    if (!edit) { out.push(line); return; }
    if (edit.hidden) return;
    out.push({ ...line, points: line.points.map(
      ([x, y]) => [x + edit.dx, y + edit.dy] as [number, number]) });
  });
  return out;
}

/** 제로 영역을 시트에 그린다 — 3D 가 표면에 얹는 것과 같은 네모다. */
function ZeroAreaOverlay({ areas, width, height }:
  { areas: [number, number][][]; width: number; height: number }) {
  if (!areas.length) return null;
  const strokeWidth = Math.max(width, height) * 0.0026;
  return <svg className="simple-zero-line-overlay" viewBox={`0 0 ${width} ${height}`}
    preserveAspectRatio="none" aria-hidden="true">
    {areas.map((box, index) => {
      const points = box.map(([x, y]) => `${x},${y}`).join(' ');
      return <g key={index}>
        <polygon points={points} fill="#ff5a1f" fillOpacity={0.16}
          stroke="#ffffff" strokeWidth={strokeWidth * 2.4} strokeLinejoin="round" />
        <polygon points={points} fill="none" stroke="#ff5a1f"
          strokeWidth={strokeWidth} strokeLinejoin="round" />
      </g>;
    })}
  </svg>;
}

function SimpleZeroLineOverlay({ lines, width, height }: { lines: SimpleZeroLine[]; width: number; height: number }) {
  // 현업 zero_line_drawing 방식 — 주요 0포인트를 직선으로 잇는다.
  // "정답지처럼 깔끔한 직선" 요구에 맞춰 곡선을 쓰지 않으므로 꼭짓점은
  // 2개(직선) 또는 3개(꺾임 1개)뿐이다. 허용범위(-0.5~0.5mm) 통과율이
  // 낮은 선은 흐리게 그려서 근거가 약하다는 걸 보이게 한다.
  if (!lines.length) return null;
  const strokeWidth = Math.max(width, height) * 0.0032;
  return <svg className="simple-zero-line-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {lines.map((line) => {
      const points = line.points.map(([x, y]) => `${x},${y}`).join(' ');
      const weak = line.tolerance_coverage < 0.55;
      return <g key={line.line_id} opacity={weak ? 0.5 : 1}>
        <polyline points={points} fill="none" stroke="#ffffff" strokeWidth={strokeWidth * 2.6} strokeLinecap="round" strokeLinejoin="round" opacity={0.9} />
        <polyline points={points} fill="none" stroke="#ff5a1f" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"
          strokeDasharray={weak ? `${strokeWidth * 3} ${strokeWidth * 2}` : undefined} />
      </g>;
    })}
  </svg>;
}

function GreenBeltOverlay({ belts, width, height }: { belts: GreenBelt[]; width: number; height: number }) {
  // 현업이 준 방법 그대로 — "녹색(오차 0 근처)" 이면서 "플러스/마이너스가
  // 전환되는" 자리만 벨트로 낸다. 근거가 없는 자리엔 아무것도 안 그린다.
  const usable = belts.filter((b) => b.contour.length >= 3);
  if (!usable.length) return null;
  const strokeWidth = Math.max(width, height) * 0.003;
  return <svg className="green-belt-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {usable.map((belt) => {
      const points = belt.contour.map(([x, y]) => `${x},${y}`).join(' ');
      return <g key={belt.belt_id}>
        <polygon points={points} fill="none" stroke="#ffffff" strokeWidth={strokeWidth * 3} strokeLinejoin="round" opacity={0.9} />
        <polygon points={points} fill="rgba(0,200,80,0.45)" stroke="#00893f" strokeWidth={strokeWidth} strokeLinejoin="round" />
      </g>;
    })}
  </svg>;
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

function SheetCanvas({ scan, imageUrl, points, coefficient, showPoints, showZero, visiblePointIds, onPointToggle, pointOverrides, onOverrideChange, annotations, showAnnotations, annotationTool, setAnnotationTool, selectedAnnotationId, setSelectedAnnotationId, onAnnotationCommit, onAnnotationCreate, onAnnotationDelete, detailMode, setDetailMode, labelAreaMode, setLabelAreaMode, addPointMode, onAddPointAt, sampling, sampleError, addedPoints, onRemoveAddedPoint, zeroEdits }: { scan: ScanItem; imageUrl: string; points: PointResult[]; coefficient: number; showPoints: boolean; showZero: boolean; visiblePointIds: Set<string>; onPointToggle: (id: string) => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; annotations: Annotation[]; showAnnotations: boolean; annotationTool: AnnotationTool; setAnnotationTool: (tool: AnnotationTool) => void; selectedAnnotationId: string | null; setSelectedAnnotationId: (id: string | null) => void; onAnnotationCommit: (annotation: Annotation) => void; onAnnotationCreate: (annotation: Annotation) => void; onAnnotationDelete: (id: string) => void; detailMode: boolean; setDetailMode: (value: boolean) => void; labelAreaMode: 'hide' | 'show' | null; setLabelAreaMode: (value: 'hide' | 'show' | null) => void; addPointMode: boolean; onAddPointAt: (x: number, y: number) => void; sampling: boolean; sampleError: string | null; addedPoints: PointResult[]; onRemoveAddedPoint: (id: string) => void;
  zeroEdits: ZeroEdit[] }) {
  const sourceAspect = scan.result!.source.width / scan.result!.source.height;
  const initialFrontSize = fitAspectSize(sourceAspect, 62, 64);
  const [regions, setRegions] = useState<DetailRegion[]>([]);
  const [layouts, setLayouts] = useState<SheetLayout[]>([{ id: 'front', kind: 'front', x: 4, y: 7, ...initialFrontSize }]);
  const [hiddenDetailPointIds, setHiddenDetailPointIds] = useState<Record<string, Set<string>>>({});
  const [selectedLayoutId, setSelectedLayoutId] = useState('front');
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const updateLayout = (next: SheetLayout) => setLayouts((current) => current.map((layout) => layout.id === next.id ? next : layout));
  const createDetail = (region: DetailRegion) => {
    const detailCount = layouts.filter((layout) => layout.kind === 'detail').length;
    const detailAspect = region.w * scan.result!.source.width / (region.h * scan.result!.source.height);
    const nextLayout: SheetLayout = { id: `layout-${region.id}`, kind: 'detail', regionId: region.id, x: 68, y: 7 + (detailCount % 3) * 29, ...fitAspectSize(detailAspect, 28, 25) };
    setRegions((current) => [...current, region]); setLayouts((current) => [...current, nextLayout]);
    setSelectedRegionId(region.id); setSelectedLayoutId(nextLayout.id); setDetailMode(false);
  };
  const deleteDetail = (regionId: string) => {
    const targetLayout = layouts.find((layout) => layout.regionId === regionId);
    setRegions((current) => current.filter((region) => region.id !== regionId));
    setLayouts((current) => current.filter((layout) => layout.regionId !== regionId));
    if (targetLayout) setHiddenDetailPointIds((current) => { const next = { ...current }; delete next[targetLayout.id]; return next; });
    setSelectedRegionId(null); setSelectedLayoutId('front');
  };
  const selectedLayout = layouts.find((layout) => layout.id === selectedLayoutId) || layouts[0];
  const aspectFor = (layout: SheetLayout) => {
    const region = layout.regionId ? regions.find((item) => item.id === layout.regionId) : undefined;
    return region ? region.w * scan.result!.source.width / (region.h * scan.result!.source.height) : sourceAspect;
  };
  const setSelectedSize = (key: 'w' | 'h', value: number) => {
    if (!selectedLayout) return;
    const aspect = aspectFor(selectedLayout);
    const minW = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * aspect / SHEET_ASPECT); const maxW = Math.min(100, 100 * aspect / SHEET_ASPECT);
    const minH = Math.max(MIN_LAYOUT_SIZE, MIN_LAYOUT_SIZE * SHEET_ASPECT / aspect); const maxH = Math.min(100, 100 * SHEET_ASPECT / aspect);
    const w = key === 'w' ? clamp(value, minW, maxW) : clamp(value, minH, maxH) * aspect / SHEET_ASPECT;
    const h = key === 'h' ? clamp(value, minH, maxH) : clamp(value, minW, maxW) * SHEET_ASPECT / aspect;
    updateLayout(normalizeBox({ ...selectedLayout, w, h }, 0));
  };
  const updateDetailRegion = (next: DetailRegion) => {
    setRegions((current) => current.map((item) => item.id === next.id ? next : item));
    const nextAspect = next.w * scan.result!.source.width / (next.h * scan.result!.source.height);
    setLayouts((current) => current.map((layout) => layout.regionId === next.id ? normalizeBox({ ...layout, ...fitAspectSize(nextAspect, layout.w, 100) }, 0) : layout));
  };

  return <div className={`sheet-canvas ${detailMode ? 'sheet-canvas--detail-mode' : ''}`} onPointerDown={(event) => { if (event.target === event.currentTarget) { setSelectedLayoutId(''); setSelectedRegionId(null); setSelectedAnnotationId(null); } }}>
    <div className="sheet-canvas__meta"><b>A3 · FRONT VIEW</b><span>{scan.partNo} / REV.01</span></div>
    {layouts.map((layout) => {
      const region = layout.regionId ? regions.find((item) => item.id === layout.regionId) : undefined;
      if (layout.kind === 'detail' && !region) return null;
      const title = layout.kind === 'front' ? '정면도 · FRONT VIEW' : region!.label;
      const imageAspect = region ? region.w * scan.result!.source.width / (region.h * scan.result!.source.height) : sourceAspect;
      const detailPoints = region ? points.filter((point) => point.x >= region.x && point.x <= region.x + region.w && point.y >= region.y && point.y <= region.y + region.h).map((point) => ({ ...point, x: (point.x - region.x) / region.w * 100, y: (point.y - region.y) / region.h * 100 })) : points;
      const layoutVisiblePointIds = layout.kind === 'front' ? visiblePointIds : new Set(detailPoints.filter((point) => !hiddenDetailPointIds[layout.id]?.has(point.id)).map((point) => point.id));
      const toggleLayoutPoint = layout.kind === 'front' ? onPointToggle : (id: string) => setHiddenDetailPointIds((current) => { const hidden = new Set(current[layout.id] || []); if (hidden.has(id)) hidden.delete(id); else hidden.add(id); return { ...current, [layout.id]: hidden }; });
      const applyAreaPoints = (ids: string[], mode: 'hide' | 'show') => {
        if (layout.kind === 'front') ids.filter((id) => mode === 'hide' ? layoutVisiblePointIds.has(id) : !layoutVisiblePointIds.has(id)).forEach(onPointToggle);
        else setHiddenDetailPointIds((current) => { const hidden = new Set(current[layout.id] || []); ids.forEach((id) => mode === 'hide' ? hidden.add(id) : hidden.delete(id)); return { ...current, [layout.id]: hidden }; });
      };
      return <SheetLayoutFrame key={layout.id} layout={layout} imageAspect={imageAspect} selected={selectedLayoutId === layout.id} onSelect={() => setSelectedLayoutId(layout.id)} onChange={updateLayout} onDelete={region ? () => deleteDetail(region.id) : undefined} title={title}>
        {region ? <div className="detail-crop"><div className="layout-image-clip"><img src={imageUrl} alt={`${region.label} 확대 정면도`} style={{ width: `${10000 / region.w}%`, height: `${10000 / region.h}%`, left: `${-region.x / region.w * 100}%`, top: `${-region.y / region.h * 100}%` }} /></div>{showPoints && <CorrectionPoints coefficient={coefficient} points={detailPoints} visibleLabelIds={layoutVisiblePointIds} onLabelToggle={toggleLayoutPoint} overrides={pointOverrides} onOverrideChange={onOverrideChange} />}</div>
          : <div className="front-view-layout"><img src={imageUrl} alt="스캔 데이터에서 추출한 정면도" />
            {/* 제로라인. 예전에는 시트의 "제로라인" 단추가 배경 그림만
                바꿨는데, 그 그림(zeroOverlay)에는 아무것도 그려져 있지
                않아서 눌러도 아무 일이 없었다. 엔진 결과 화면에서 쓰던
                것과 같은 겹침층을 여기에도 올린다. */}
            {showZero && (scan.result?.simpleZeroLines?.length ?? 0) > 0 &&
              <ZeroAreaOverlay areas={scan.result!.labZeroAreas ?? []}
                width={scan.result!.source.width} height={scan.result!.source.height} />}
            {showZero && layout.kind === 'front' &&
              <SimpleZeroLineOverlay
                lines={applyZeroEdits(zeroLinesToShow(scan.result!), zeroEdits)}
                width={scan.result!.source.width} height={scan.result!.source.height} />}{addPointMode && layout.kind === 'front' && <><div className="add-point-catcher" onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); const rect = event.currentTarget.getBoundingClientRect(); if (!rect.width || !rect.height) return; onAddPointAt((event.clientX - rect.left) / rect.width * 100, (event.clientY - rect.top) / rect.height * 100); }} />
          {/* 지우기는 거리 판정 대신 포인트 위 전용 버튼으로 받는다. 점이 작아 손으로 정확히 겨누기 어렵다. */}
          {addedPoints.map((added) => <button key={added.id} type="button" className="add-point-remove" style={{ left: `${added.x}%`, top: `${added.y}%` }}
            onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); onRemoveAddedPoint(added.id); }}
            aria-label={`${added.id} 추가 포인트 삭제`} title="이 추가 포인트 삭제"><X size={9} /></button>)}</>}{showPoints && <CorrectionPoints coefficient={coefficient} points={points} visibleLabelIds={layoutVisiblePointIds} onLabelToggle={toggleLayoutPoint} overrides={pointOverrides} onOverrideChange={onOverrideChange} />}<DetailRegionLayer regions={regions} active={detailMode} selectedId={selectedRegionId} onSelect={setSelectedRegionId} onCreate={createDetail} onChange={updateDetailRegion} onDelete={deleteDetail} /></div>}
        <LabelAreaSelector mode={labelAreaMode} points={detailPoints} onApply={applyAreaPoints} onComplete={() => setLabelAreaMode(null)} />
      </SheetLayoutFrame>;
    })}
    {showAnnotations && !detailMode && !labelAreaMode && <AnnotationLayer annotations={annotations} tool={annotationTool} setTool={setAnnotationTool} selectedId={selectedAnnotationId} onSelect={setSelectedAnnotationId} onCommit={onAnnotationCommit} onCreate={onAnnotationCreate} onDelete={onAnnotationDelete} />}
    {detailMode && <div className="detail-mode-guide"><ZoomIn size={14} /><span>정면도 위에서 확대할 영역을 드래그하세요.</span><button type="button" onClick={() => setDetailMode(false)}>취소</button></div>}
    {addPointMode && <div className="detail-mode-guide add-point-guide"><Crosshair size={14} /><span>{sampleError || (sampling ? '편차값을 읽는 중입니다…' : '정면도를 눌러 보정 포인트를 추가합니다. 값은 히트맵 색에서 추정하며, 추가한 포인트를 다시 누르면 지워집니다.')}</span></div>}
    {labelAreaMode && <div className={`detail-mode-guide label-area-guide label-area-guide--${labelAreaMode}`}>{labelAreaMode === 'hide' ? <EyeOff size={14} /> : <Eye size={14} />}<span>레이아웃 위에서 {labelAreaMode === 'hide' ? '숨길' : '표시할'} 라벨 영역을 드래그하세요.</span><button type="button" onClick={() => setLabelAreaMode(null)}>취소</button></div>}
    {selectedLayout && <div className="layout-size-control" onPointerDown={(event) => event.stopPropagation()}><b>{selectedLayout.kind === 'front' ? '정면도' : regions.find((item) => item.id === selectedLayout.regionId)?.label} 크기 · 비율 고정</b><label>W <input type="range" min="5" max="100" value={selectedLayout.w} onChange={(event) => setSelectedSize('w', Number(event.target.value))} /><span>{Math.round(selectedLayout.w)}%</span></label><label>H <input type="range" min="5" max="100" value={selectedLayout.h} onChange={(event) => setSelectedSize('h', Number(event.target.value))} /><span>{Math.round(selectedLayout.h)}%</span></label></div>}
  </div>;
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

function Header({ scans, activeId, setActiveId, onSaveFile, onLoadFile, onReset, note }: { scans: ScanItem[]; activeId?: string; setActiveId: (id: string) => void; onSaveFile: () => void; onLoadFile: (file: File) => void; onReset: () => void; note?: string | null }) {
  const fileRef = useRef<HTMLInputElement>(null);
  return <header className="topbar"><div><span className="topbar__context">AJIN INDUSTRIAL · DIE ENGINEERING</span><h1>ADC <small>Ajin Die Compensation</small></h1></div><div className="topbar__actions">
    <label className="item-select"><span>현재 품번</span><select value={activeId || ''} disabled={!scans.length} onChange={(e) => setActiveId(e.target.value)}><option value="">등록된 이미지 없음</option>{scans.map((scan) => <option value={scan.id} key={scan.id}>{scan.partNo} · {scan.name}</option>)}</select></label>
    {/* 작업 내용은 자동으로 이 PC 에 남는다. 파일로 빼두면 보관하거나
        다른 사람에게 넘길 수 있다. */}
    <div className="topbar__session">
      {note && <span className="topbar__session-note">{note}</span>}
      <button type="button" className="tool-button" onClick={onSaveFile}
        title="고친 보정값과 설정을 파일로 내려받습니다">작업 저장</button>
      <button type="button" className="tool-button"
        onClick={() => fileRef.current?.click()}
        title="저장해 둔 작업 파일을 불러옵니다">작업 불러오기</button>
      <button type="button" className="tool-button" onClick={onReset}
        title="이 PC 에 남은 작업 내용을 지웁니다">작업 비우기</button>
      <input ref={fileRef} type="file" hidden accept="application/json,.json"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onLoadFile(file);
          event.target.value = '';
        }} />
    </div>
    <button className="icon-button" aria-label="설정"><Settings2 size={19} /></button><div className="profile"><span>KJ</span><div><b>금형생산팀</b><small>관리자</small></div></div>
  </div></header>;
}

/* 파일명에서 품번을 뽑는다. 백엔드 file_naming.py 와 같은 규칙이며
   짧은 형태(64XX2)를 준다 — 컬러바 표와 제로라인 라이브러리의 열쇠다.

   순수 숫자 여섯 자리는 **날짜**라 품번으로 보지 않는다. NC 데이터가
   `260825_JDZ_DASH LWR_OP10_...ZIP` 처럼 날짜를 앞에 달고 오는데,
   예전 규칙은 이걸 품번이라고 집어냈다. */
function partNoFromName(name: string): string | null {
  for (const token of name.toUpperCase().split(/[_\s]+/)) {
    const head = token.split(/[-/]/)[0];
    if (!/^[0-9]{2}[A-Z0-9]{2,4}$/.test(head)) continue;
    if (/^[0-9]+$/.test(head) && !/[-/]/.test(token)) continue;   // 날짜
    return head;
  }
  return null;
}

/* 컬러바 범위가 등록된 품번. 백엔드 PRODUCT_COLORBAR_MM 과 같은 표이며
   분석 응답의 knownParts 로도 확인할 수 있다. */
const KNOWN_PARTS = ['64XX2', '67XX6', '71XX2'];

function Workspace({ scans, setScans, onOpenResults, backendOnline }: { scans: ScanItem[]; setScans: React.Dispatch<React.SetStateAction<ScanItem[]>>; onOpenResults: (id: string) => void; backendOnline: boolean | null }) {
  const [dragging, setDragging] = useState(false);
  const analyzingCount = scans.filter((scan) => scan.status === 'analyzing').length;
  const addFiles = (files: FileList | File[]) => {
    const accepted = Array.from(files).filter((file) => file.type.startsWith('image/'));
    const next = accepted.map((file, index): ScanItem => ({
      id: `${file.name}-${file.lastModified}-${crypto.randomUUID()}`,
      name: file.name,
      partNo: partNoFromName(file.name) || `NEW-${String(scans.length + index + 1).padStart(2, '0')}`,
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
        /* 품번은 컬러바 범위와 제로라인 파라미터를 고르는 열쇠다. 파일명에
           품번이 없으면(디버그 출력물 등) 제로라인 단계가 통째로 비어 버린다 —
           같은 그림을 파일명만 바꿔 넣어 확인했다(0개 -> 3개). 그래서
           화면에서 고른 품번을 함께 보낸다. */
        if (KNOWN_PARTS.includes(target.partNo)) form.append('partNo', target.partNo);
        const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
        const data = await response.json() as AnalysisResult & { error?: string };
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
      <div className="file-list"><div className="file-list__head"><span>등록된 이미지</span><button><ListFilter size={15} /> 상태순</button></div>{!scans.length && <div className="empty-file-list">아직 등록된 이미지가 없습니다.</div>}{scans.map((scan) => <div className="file-row" key={scan.id}><div className={`file-thumb tone-${scan.tone}`}><img src={scan.url} alt="" /></div><div className="file-row__name"><b>{scan.name}</b><span>
      <select className="part-pick" aria-label={`${scan.name} 품번`}
        value={KNOWN_PARTS.includes(scan.partNo) ? scan.partNo : ''}
        onChange={(event) => {
          const picked = event.target.value;
          if (!picked) return;
          /* 품번이 바뀌면 값의 기준(컬러바 범위)이 달라지므로 이전 결과는
             버리고 다시 분석해야 한다. */
          setScans((current) => current.map((item) => item.id === scan.id
            ? { ...item, partNo: picked, status: 'ready', result: undefined, error: undefined }
            : item));
        }}>
        <option value="">{scan.partNo} · 품번 미지정</option>
        {KNOWN_PARTS.map((part) => <option key={part} value={part}>{part}</option>)}
      </select>
      {' · '}{scan.error || scan.size}
    </span></div><span className={`status status--${scan.status}`}>{scan.status === 'done' ? <><Check size={13} /> 분석 완료</> : scan.status === 'analyzing' ? <><Activity size={13} /> 분석 중</> : scan.status === 'error' ? '오류' : '대기'}</span>{scan.status === 'done' ? <button className="text-button" onClick={() => onOpenResults(scan.id)}>결과 보기 <ArrowRight size={14} /></button> : <button className="icon-button icon-button--small" onClick={() => removeScan(scan.id)} aria-label={`${scan.name} 삭제`}><X size={15} /></button>}</div>)}</div>
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
  // 승인 도면이 있는 품번은 그것이 데모의 제로라인이다. 우리 검출 결과는
  // 기본으로 숨기고, 필요할 때만 켜서 대조한다.
  const [showDetection, setShowDetection] = useState(false);
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
      // 백엔드는 snake_case 로 준다 — 프론트 ValleyLine 과 필드명이 달라
      // 별도 타입을 둔다.
      .then((response) => response.json().then((data) => ({
        ok: response.ok,
        data: data as { error?: string; line?: ValleyLineResponse },
      })))
      .then(({ ok, data }) => {
        if (cancelled) return;
        if (!ok) { setValleyStatus('error'); setValleyError(data.error || '선을 잇지 못했습니다.'); setSelectedAnchors([]); return; }
        const line = data.line;
        if (!line) { setValleyStatus('error'); setValleyError('선을 잇지 못했습니다.'); setSelectedAnchors([]); return; }
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
        {/* 현업 파이프라인 선이 나오면 **그것만** 보여준다.
            예전 방식들(my_lab 도형·녹색 벨트·제로존·계곡선·앵커)을 같이
            깔면 화면이 선으로 덮여 무엇이 답인지 알 수 없다. 근거가
            가장 분명한 하나만 남긴다. 옛 방식은 이 파이프라인이 못 도는
            품번(SCAN_SCALES 미등록)에서만 나온다. */}
        {(result.labZeroLines?.length ?? 0) > 0 ? (
          <SimpleZeroLineOverlay lines={zeroLinesToShow(result)} width={result.source.width} height={result.source.height} />
        ) : <>
          <LabProfileOverlay shapes={result.labProfile || []} width={result.source.width} height={result.source.height} />
          {showDetection && <>
            <ZeroZoneOverlay clusters={zeroAreaClusters} width={result.source.width} height={result.source.height} />
            <GreenBeltOverlay belts={result.greenBelts || []} width={result.source.width} height={result.source.height} />
            <SimpleZeroLineOverlay lines={zeroLinesToShow(result)} width={result.source.width} height={result.source.height} />
            <ValleyLineOverlay lines={valleyLines} width={result.source.width} height={result.source.height} />
            <AnchorPicker anchors={zeroAnchors} width={result.source.width} height={result.source.height} selectedIds={selectedAnchors} onToggle={toggleAnchor} />
          </>}
        </>}
      </>}</Heatmap></div><div className="viewer-legend"><span><i className="legend-dot" style={{ background: meta.color }} /> 현재 표시: {meta.name}</span><span>{engine === 'deviation' ? '라벨이나 포인트 점을 누르면 개별 표시를 켜고 끌 수 있습니다.' : '표시된 값과 위치는 업로드 이미지의 실제 엔진 결과입니다.'}</span></div></div>
      <aside className="inspection-panel"><div className="score-card card"><span className="score-card__icon" style={{ color: meta.color, background: `${meta.color}12` }}>{engine === 'label' ? <Sparkles /> : engine === 'deviation' ? <Activity /> : <Gauge />}</span><span>핵심 결과</span><strong style={{ color: result.errors[engine] ? '#bd4650' : meta.color }}>{summary.stat}</strong><p>{summary.detail}</p></div><div className="card plain-summary"><h3>쉽게 보는 결과</h3><div className="summary-line"><Check size={16} /><div><b>처리 방식</b><span>{engine === 'label' ? 'label_removal의 인페인팅 결과입니다.' : engine === 'deviation' ? '라벨 제거 이미지에 deviation_extraction의 지시선 끝점과 판독값을 겹쳐 표시합니다.' : 'zero_line_detection의 컬러바 기반 결과입니다.'}</span></div></div>{engineWarnings.length > 0 && <div className="summary-line warning"><MoveRight size={16} /><div><b>확인 필요</b><span>{engineWarnings[0]}</span></div></div>}</div><div className="card mini-table"><div className="card-title"><h3>검출 포인트</h3><span>라벨 {visibleLabelIds.size}/{result.points.length}</span></div>{result.points.map((point) => { const visible = visibleLabelIds.has(point.id); return <div className="point-list-row" key={point.id}><span>{point.id}</span><b className={point.value > 0 ? 'positive' : 'negative'}>{point.value > 0 ? '+' : ''}{point.value.toFixed(3)} mm</b><small>{point.xPx}, {point.yPx}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleLabel(point.id)} aria-label={`${point.id} 라벨 ${visible ? '숨기기' : '표시하기'}`} title={`라벨 ${visible ? 'OFF' : 'ON'}`}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}{!result.points.length && <p className="empty-mini">검출된 포인트가 없습니다.</p>}</div>
      {engine === 'zero' && <div className="card mini-table anchor-panel">
        <div className="card-title"><h3>제로라인</h3><span>{
          (result.labZeroLines?.length ?? 0) > 0 ? `${result.labZeroLines!.length}개`
          : (result.labProfile || []).length > 0 ? `my_lab 도형 ${result.labProfile.length}개`
          : `앵커 ${zeroAnchors.length}개`}</span></div>
        {(result.labZeroRegions?.length ?? 0) > 0 ? <>
          <p className="anchor-panel__hint">
            <b>허용범위 ±0.7mm 밖</b>을 보정 영역으로 잡고, 그 영역이 윤곽과
            맞닿는 자리에서 윤곽을 따라 양쪽으로 걸어가 <b>처음 만나는
            제로포인트 둘</b>을 이은 선입니다. 보정 영역은 피해서 돌아갑니다.
          </p>
          {result.labZeroRegions!.map((region) => (
            <div className="point-list-row" key={region.label}>
              <span>{region.label}</span>
              <b className="positive">{region.zeroPoints.join(' – ')}</b>
              <small>{Math.round(region.area).toLocaleString()}px</small>
              <small>{region.attempts}회 시도</small>
            </div>
          ))}
          <p className="anchor-panel__hint">
            각 선은 <b>목표 영역을 실제로 닫는지</b> 검사를 통과한 것만
            남습니다(덮음 {Math.round((result.labZeroRegions![0]?.coverage ?? 0) * 100)}%).
            닫지 못하면 다음 제로포인트로 넘어가 다시 시도합니다.
          </p>
        </> : (result.labProfile || []).length > 0 ? <>
          <p className="anchor-panel__hint">
            빨간 점선은 <b>my_lab 파이프라인이 그린 제로라인</b>입니다.
            이 품번은 해당 도형이 있어 그대로 표시합니다.
          </p>
          <button type="button" className="tool-button" onClick={() => setShowDetection((v) => !v)}>
            {showDetection ? <EyeOff size={14} /> : <Eye size={14} />} 자동 검출 결과 {showDetection ? '숨기기' : '함께 보기'}
          </button>
          {showDetection && result.labDistance && <p className="anchor-panel__hint">
            자동 검출한 직선과 이 도형 사이 거리는 {result.labDistance.to_lab_pct}% /
            {' '}{result.labDistance.to_predicted_pct}% 입니다 (대각선 대비 중앙값).
            다만 <b>둘은 같은 0포인트를 끝점으로 쓰므로</b> 이 값이 정확도의 근거는 아닙니다.
          </p>}
        </> : <p className="anchor-panel__hint">
          이 품번은 승인 도면이 없어 <b>자동 검출 결과</b>를 표시합니다.
          초록 면은 오차가 0에 가까우면서 +/− 가 뒤바뀌는 구간, 주황 직선은
          주요 0포인트를 이은 제로라인입니다.
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
      const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(entry.path)}`); const data = await response.json() as FolderResponse;
      if (response.ok) { setChildren((data.entries || []).filter((item: FolderEntry) => item.isDirectory)); setLoaded(true); }
    }
    setExpanded((value) => !value); onOpen(entry.path);
  };
  return <div className="tree-node"><button className={`tree-root ${selectedPath === entry.path ? 'selected' : ''}`} onClick={toggle}>{expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}<Folder size={17} fill="currentColor" /> <span>{entry.name}</span></button>{expanded && <div className="tree-children">{children.map((child) => <FolderTreeNode key={child.path} entry={child} selectedPath={selectedPath} onOpen={onOpen} />)}{loaded && !children.length && <span className="tree-empty">하위 폴더 없음</span>}</div>}</div>;
}

function Explorer() {
  const [available, setAvailable] = useState<boolean | null>(null); const [rootName, setRootName] = useState('품번별 폴더'); const [rootEntries, setRootEntries] = useState<FolderEntry[]>([]); const [entries, setEntries] = useState<FolderEntry[]>([]); const [path, setPath] = useState(''); const [query, setQuery] = useState('');
  const openFolder = async (nextPath: string) => {
    const response = await fetch(`${API_BASE}/api/folders?path=${encodeURIComponent(nextPath)}`); const data = await response.json() as FolderResponse;
    if (!response.ok || data.available === false) { setAvailable(false); return; }
    setAvailable(true); setRootName(data.rootName || '품번별 폴더'); setEntries(data.entries || []); setPath(data.path || '');
    if (!nextPath) setRootEntries((data.entries || []).filter((item: FolderEntry) => item.isDirectory));
  };
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/folders?path=`)
      .then((response) => response.json().then((data) => ({ ok: response.ok, data: data as FolderResponse })))
      .then(({ ok, data }) => {
        if (cancelled) return;
        if (!ok || data.available === false) { setAvailable(false); return; }
        const nextEntries = data.entries || [];
        setAvailable(true); setRootName(data.rootName || '품번별 폴더'); setEntries(nextEntries); setPath('');
        setRootEntries(nextEntries.filter((item: FolderEntry) => item.isDirectory));
      })
      .catch(() => { if (!cancelled) setAvailable(false); });
    return () => { cancelled = true; };
  }, []);
  if (available === false) return null;
  const filtered = entries.filter((entry) => entry.name.toLowerCase().includes(query.toLowerCase())); const segments = path ? path.split('/') : [];
  return <div className="explorer card"><div className="explorer__title"><div><FolderOpen size={20} /><b>실시간 품번별 폴더</b></div><span>{available == null ? '연결 확인 중' : '현재 PC 폴더와 연결됨'}</span></div><div className="explorer__bar"><div className="explorer__crumb"><button disabled={!path} onClick={() => openFolder(segments.slice(0, -1).join('/'))}><ArrowLeft size={14} /></button><span><button onClick={() => openFolder('')}>{rootName}</button>{segments.map((segment, index) => <span key={`${segment}-${index}`}><ChevronRight size={13} /><button onClick={() => openFolder(segments.slice(0, index + 1).join('/'))}>{segment}</button></span>)}</span></div><label><ZoomIn size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="현재 폴더 검색" /></label></div><div className="explorer__body"><div className="folder-tree"><button className={`tree-root ${!path ? 'selected' : ''}`} onClick={() => openFolder('')}><ChevronDown size={15} /><FolderOpen size={17} /> <span>{rootName}</span></button><div className="tree-children">{rootEntries.map((entry) => <FolderTreeNode key={entry.path} entry={entry} selectedPath={path} onOpen={openFolder} />)}</div></div><div className="folder-content"><div className="folder-content__head"><span>이름</span><span>수정한 날짜</span><span>크기</span></div>{filtered.map((entry) => <button className="folder-row" key={entry.path} onDoubleClick={() => entry.isDirectory && openFolder(entry.path)} onClick={() => entry.isDirectory && openFolder(entry.path)}><span>{entry.isDirectory ? <Folder size={19} fill="currentColor" /> : <File size={18} />}{entry.name}</span><small>{new Date(entry.modified).toLocaleString('ko-KR')}</small><small>{entry.isDirectory ? '파일 폴더' : formatBytes(entry.size)}</small></button>)}{!filtered.length && <div className="empty-search">이 폴더는 비어 있습니다.</div>}<div className="folder-content__status">{filtered.length}개 항목 <span>·</span> 실시간 로컬 조회</div></div></div></div>;
}

/* 보정시트 머리말.
   실제 시트(JM 67312-DZ000_보정적용.xlsx)가 이렇게 채워져 있다 —

     관리 NO   JM 67312-DZ000-13      공   정  OP50
     PART NAME DASH UPR LHD           PART NO  67312-DZ000
     원소재     A6451P-T4S 1.8t        적용일자  2025-07-15

   그래서 파일명 규칙에서 읽어낸 값을 그대로 쓴다. 예전에는 PART NAME
   자리에 파일명 전체를 밀어 넣고 공정은 "금형 보정" 으로 박아 뒀는데,
   현업 시트에는 그런 값이 들어가지 않는다. 못 읽은 칸은 비워 둔다 —
   지어내는 것보다 비는 편이 낫다. */
function SheetTitleBlock({ scan }: { scan: ScanItem }) {
  const named = scan.result?.naming;
  const blank = <span className="sheet-title-block__blank">—</span>;
  const applied = named?.applied_at
    ? new Intl.DateTimeFormat('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' })
        .format(new Date(`${named.applied_at}T00:00:00`))
    : null;

  return <section className="sheet-title-block" aria-label="보정 적용 내용">
    <div className="sheet-title-block__heading"><strong>보정 적용 내용</strong></div>
    <div className="sheet-title-block__label">관리 NO</div><div className="sheet-title-block__value">{named?.control_no ?? `${scan.partNo}-01`}</div>
    <div className="sheet-title-block__label">PART NAME</div><div className="sheet-title-block__value" title={named?.part_name ?? ''}>{named?.part_name ?? blank}</div>
    <div className="sheet-title-block__label">공정</div><div className="sheet-title-block__value">{named?.process ?? blank}</div>
    <div className="sheet-title-block__label">PART NO</div><div className="sheet-title-block__value">{named?.part_no ?? scan.partNo}</div>
    <div className="sheet-title-block__label">원소재</div><div className="sheet-title-block__value">{blank}</div>
    <div className="sheet-title-block__label">적용일자</div><div className="sheet-title-block__value">{applied ?? blank}</div>
  </section>;
}

// 3D CAD/스캔 데이터 화면.
//
// 현업 자료(2026-08-25)가 정리한 제로라인 판정 4가지 방법 중 3가지
// (RPS 정렬, 수축 중심선, 단면 분석)는 3D 데이터가 있어야 한다. 지금까지
// 우리가 가진 건 2D 히트맵뿐이라 컬러맵 제로존 하나만 쓸 수 있었다.
// 여기서 STEP 을 읽으면 조립 홀 좌표가 나오므로 RPS 정렬의 입구가 열린다.
function CadWorkspace({ active, scans, coefficientByScan, hiddenPointIdsByScan, pointOverridesByScan, onOverrideChange, notesByCad, setNotesByCad, regionsByCad, setRegionsByCad, zeroEditsByScan }: {
  /* 이 화면을 지금 보고 있는가. 안 보고 있어도 컴포넌트는 살아 있다 —
     읽어 둔 CAD 를 지키려는 것이다. 감춰진 동안에는 그리기를 멈춘다. */
  active: boolean;
  scans: ScanItem[];
  coefficientByScan: Record<string, number>;
  hiddenPointIdsByScan: Record<string, Set<string>>;
  pointOverridesByScan: Record<string, Record<string, number>>;
  zeroEditsByScan: Record<string, ZeroEdit[]>;
  onOverrideChange: (scanId: string, pointId: string, value: number | null) => void;
  /* 주석과 공정 구역은 App 이 들고 있는다 — 새로고침해도 남으려면 세션
     저장소에 들어가야 하고, 그건 App 에 있다. */
  notesByCad: Record<string, CadNote[]>;
  setNotesByCad: React.Dispatch<React.SetStateAction<Record<string, CadNote[]>>>;
  regionsByCad: Record<string, CadRegion[]>;
  setRegionsByCad: React.Dispatch<React.SetStateAction<Record<string, CadRegion[]>>>;
}) {
  /* CAD 를 여러 개 열어 놓고 골라 본다. 좌우 대칭품이나 공정별 형상을
     번갈아 보려면 파일을 다시 올리게 하면 안 된다 — STEP 파싱이 실측
     215MB 기준 42~100초다. */
  const [meshes, setMeshes] = useState<CadMesh[]>([]);
  const [activeCadId, setActiveCadId] = useState<string>('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [pending, setPending] = useState<string[]>([]);   // 읽는 중인 파일명
  /* 몇 초째 읽고 있는지. 실측 113MB STEP 이 **57초** 걸리는데 그동안
     아무 표시가 없어서 멈춘 것처럼 보였다. 같은 파일을 다시 열면
     디스크 캐시가 있어 3초다. */
  const [waited, setWaited] = useState(0);
  /* 연 파일을 들고 있는다. 서버는 CAD 를 6개까지만 캐시하므로 7번째를
     열면 첫 파일이 밀려나 "CAD 가 만료됐습니다" 가 뜬다. 그때 조용히
     다시 올려 주면 사람은 모르고 지나간다(캐시 덕에 3초다). */
  const fileStore = useRef<Record<string, File>>({});
  const [error, setError] = useState<string | null>(null);
  const mesh = useMemo(
    () => meshes.find((m) => (m.cadId ?? m.summary.name) === activeCadId) ?? null,
    [meshes, activeCadId]);
  const keyOf = (item: CadMesh) => item.cadId ?? item.summary.name;
  /* 한 번 계산한 오버레이는 들고 있는다. CAD 를 바꿔 가며 견줄 때마다
     실루엣 정합을 다시 돌리면 몇 초씩 멈춘다. */
  const overlayCache = useRef<Record<string, CadOverlay>>({});
  /* 홀 핀은 데이텀 볼 때만 필요하다. 기본으로 켜두면 보정량 콜아웃을
     가린다(실측 67XX6 은 홀이 180개다). */
  const [showHoles, setShowHoles] = useState(false);
  const [overlay, setOverlay] = useState<CadOverlay | null>(null);
  const [overlayScanId, setOverlayScanId] = useState<string>('');
  const [overlayStatus, setOverlayStatus] =
    useState<'idle' | 'loading' | 'error'>('idle');
  const [overlayError, setOverlayError] = useState<string | null>(null);
  /* 주석·공정 구역의 열쇠는 cadId 가 아니라 **파일 이름**이다. cadId 는
     파일을 열 때마다 새로 만드는 uuid 라 새로고침하면 짝을 잃는다.
     스캔을 품번으로 묶는 것과 같은 이유다. */
  const notesKey = mesh?.summary.name ?? '';
  /* 보정 후 형상 — 원본과 견줘 본다. B-Rep 이 아니라 메시를 민 것이라
     가공용이 아니고 비교용이다. */
  const [morph, setMorph] = useState<CadMorph | null>(null);
  const [morphMode, setMorphMode] = useState<'off' | 'after' | 'both'>('off');
  const [morphState, setMorphState] = useState<'idle' | 'working' | 'error'>('idle');
  const [morphError, setMorphError] = useState<string | null>(null);
  /* 보정시트가 적어 둔 단면 위치로 계산하는 제로라인. 71XX2 시트의
     "H : 300" · "T : 1700" 같은 표기가 부품 좌표(mm)라, 그 평면으로
     CAD 를 자르면 제로라인이 **추정 없이** 나온다. */
  const [sectionNotes, setSectionNotes] = useState('');
  const [sections, setSections] = useState<CadSection[] | null>(null);
  const [sectionSide, setSectionSide] = useState<'both' | 'lh' | 'rh'>('both');
  const [sectionError, setSectionError] = useState<string | null>(null);
  const [sectionState, setSectionState] = useState<'idle' | 'working'>('idle');

  const cutSections = async () => {
    if (!mesh?.cadId) return;
    setSectionState('working'); setSectionError(null);
    try {
      const response = await fetch(`${API_BASE}/api/cad-sections`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cadId: mesh.cadId, notes: sectionNotes, side: sectionSide }),
      });
      const data = await response.json() as { sections?: CadSection[]; error?: string };
      if (!response.ok) throw new Error(data.error || '자르지 못했습니다.');
      setSections(data.sections ?? []);
    } catch (err) {
      setSectionError(String((err as Error).message || err));
      setSections(null);
    } finally {
      setSectionState('idle');
    }
  };

  const morphPayload = () => {
    const spots: Record<string, [number, number, number]> = {};
    for (const point of overlay?.points ?? []) {
      if (typeof sheetValues?.values?.[point.id] === 'number') {
        spots[point.id] = point.position;
      }
    }
    return { cadId: mesh?.cadId, corrections: sheetValues?.values ?? {}, positions: spots };
  };

  const buildMorph = async () => {
    if (!mesh?.cadId || !overlay || !sheetValues) {
      setMorphError('스캔 결과를 먼저 올리세요.'); return;
    }
    setMorphState('working'); setMorphError(null);
    try {
      const response = await fetch(`${API_BASE}/api/cad-morph`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(morphPayload()),
      });
      const data = await response.json() as CadMorph & { error?: string };
      if (!response.ok) throw new Error(data.error || '만들지 못했습니다.');
      setMorph(data); setMorphMode('both'); setMorphState('idle');
    } catch (err) {
      setMorphError(String((err as Error).message || err));
      setMorphState('error');
    }
  };

  const saveMorphStl = async () => {
    if (!mesh?.cadId) return;
    const response = await fetch(`${API_BASE}/api/cad-morph-stl`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(morphPayload()),
    });
    if (!response.ok) { setMorphError('STL 을 만들지 못했습니다.'); return; }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${mesh.summary.name}_보정후.stl`;
    link.click();
    URL.revokeObjectURL(url);
  };
  /* 시트에 담아둔 3D 화면들. 현업 시트도 전체도와 확대도를 따로 싣는다. */
  const [shots, setShots] = useState<string[]>([]);
  const [sheetState, setSheetState] = useState<'idle' | 'saving' | 'error'>('idle');
  const [sheetError, setSheetError] = useState<string | null>(null);

  const makeSheet = async () => {
    const scan = scans.find((s) => s.id === overlayScanId);
    const analysisId = scan?.result?.analysisId;
    if (!analysisId || !sheetValues) { setSheetError('스캔을 먼저 고르세요.'); return; }
    setSheetState('saving'); setSheetError(null);
    try {
      const response = await fetch(`${API_BASE}/api/sheet-excel`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId, corrections: sheetValues.values, images: shots,
          meta: {
            partNo: scan?.result?.naming?.part_no || scan?.partNo,
            partName: scan?.result?.naming?.part_name || '',
            process: scan?.result?.naming?.process || '',
            controlNo: scan?.result?.naming?.control_no || '',
            appliedAt: scan?.result?.naming?.applied_at || '',
            coefficient: sheetValues.coefficient,
            /* 시트 아래쪽 "① : 하형 용접" 줄에 그대로 들어간다. */
            processes: (regionsByCad[notesKey] ?? [])
              .map((region, order) =>
                `${CIRCLED[order] ?? order + 1} : ${region.die} ${region.work}`),
          },
        }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(data.error || '시트를 만들지 못했습니다.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${scan?.partNo || 'ADC'}_보정시트_3D.xlsx`;
      link.click();
      URL.revokeObjectURL(url);
      setSheetState('idle');
    } catch (err) {
      setSheetError(String((err as Error).message || err));
      setSheetState('error');
    }
  };
  const inputRef = useRef<HTMLInputElement>(null);

  // 분석이 끝나 analysisId 를 가진 스캔만 3D 로 올릴 수 있다.
  const analysed = useMemo(
    () => scans.filter((s) => s.result?.analysisId), [scans]);

  /* CAD 파일명과 스캔 품번이 다르다. 현업 제품데이터 폴더가 짝을 보여준다 —
     64XX1 CAD 는 64XX2 스캔과, 71XX1 은 71XX2 와 짝이다(좌우 대칭품이라
     CAD 가 한쪽만 온다). 백엔드 CAD_TO_SCAN_PART 과 같은 표다. */
  const suggested = useMemo(
    () => (mesh ? analysed.find((s) => scanFitsCad(s, mesh)) : undefined),
    [analysed, mesh]);

  /* 3D 에 얹을 값은 최종 보정시트가 정한다 — 작업자가 고친 값과 숨긴
     포인트, 보정 계수를 그대로 따른다. 여기서 따로 계산하면 시트와
     3D 가 어긋난다. 시트의 displayFor 와 같은 식이다. */
  const sheetValues = useMemo(() => {
    const scan = scans.find((s) => s.id === overlayScanId);
    if (!scan?.result) return null;
    const coefficient = coefficientByScan[scan.id] ?? 1;
    const overrides = pointOverridesByScan[scan.id] || {};
    const hidden = hiddenPointIdsByScan[scan.id] || new Set<string>();
    const values: Record<string, number> = {};
    for (const point of scan.result.points) {
      if (hidden.has(point.id)) continue;
      values[point.id] = overrides[point.id] !== undefined
        ? overrides[point.id] : -(point.value * coefficient);
    }
    return { values, coefficient, hiddenCount: hidden.size,
             overrideCount: Object.keys(overrides).length };
  }, [scans, overlayScanId, coefficientByScan,
      pointOverridesByScan, hiddenPointIdsByScan]);

  /* 품번이 맞는 스캔이 있으면 **고르지 않아도** 바로 얹는다.
     예전에는 드롭다운의 값만 바꿔 놓고 요청은 안 해서, 짝이 뻔한데도
     사람이 한 번 더 골라야 결과가 나왔다. */
  /* CAD 를 바꾸면 **그 CAD 에 맞는 스캔**으로 다시 잡는다.
     
     예전에는 고른 스캔이 하나뿐이라 CAD 탭을 옮겨도 그대로 따라갔다.
     67XX6 스캔을 걸어 둔 채 64XX1 CAD 로 넘어가면 그 스캔을 64XX1 에
     얹어 버려서 얹힘이 34% 로 떨어지고 "이 부품은 안 된다" 처럼 보였다
     (71XX1 은 21%). 품번이 다른 스캔을 얹는 것 자체가 틀린 일이다. */
  useEffect(() => {
    if (!mesh) return;
    const chosen = analysed.find((scan) => scan.id === overlayScanId);
    const fits = chosen && scanFitsCad(chosen, mesh);
    if (fits) return;
    if (suggested) requestOverlay(suggested.id, mesh.cadId);
    else if (chosen) return;      // 짝이 없으면 사람이 고른 것을 존중한다
    else setOverlay(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [suggested, overlayScanId, mesh, analysed]);

  /* 시트에서 제로라인을 고쳐 [3D 에 적용] 을 누르면 다시 얹는다.
     3D 의 제로라인은 시트가 정하는 것이므로 시트를 따라가야 한다. */
  const appliedEdits = overlayScanId
    ? JSON.stringify(zeroEditsByScan[overlayScanId] ?? []) : '';
  useEffect(() => {
    if (!overlayScanId || !mesh) return;
    requestOverlay(overlayScanId, mesh.cadId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedEdits]);

  /* 고른 CAD 가 바뀌면 얹어 둔 스캔 결과도 그 CAD 기준으로 다시 잡는다.
     캐시에 있으면 즉시 돌아온다. */
  useEffect(() => {
    setMorph(null); setMorphMode('off'); setMorphError(null);
    setSections(null); setSectionError(null);
    if (overlayScanId) requestOverlay(overlayScanId, mesh?.cadId);
    else setOverlay(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCadId]);

  /* CAD 마다 따로 들고 있는다 — 부품이 다르면 맞춘 값도 다르다.
     열쇠는 **파일 이름**이다. cadId 는 파일을 다시 읽을 때마다 새로
     생겨서(서버 캐시에서 밀려나면 다시 올린다) 맞춰 둔 값이 날아간다 —
     notesByCad·regionsByCad 와 같은 이유다. */
  const [adjustByCad, setAdjustByCad] = useState<Record<string, FitAdjust>>({});
  const adjustKey = mesh?.summary.name ?? '';
  const adjust = adjustByCad[adjustKey] ?? NO_ADJUST;
  const [showAlign, setShowAlign] = useState(false);

  /* 손으로 맞춘 정렬이 바뀌면 다시 얹는다. */
  const nudge = (next: Partial<FitAdjust>) => {
    // activeCadId 는 cadId 가 없을 때 파일 이름으로 떨어진다(keyOf).
    // 서버에 보낼 때는 반드시 진짜 cadId 를 써야 한다.
    const cadId = mesh?.cadId;
    if (!cadId || !adjustKey || !overlayScanId) return;
    const moved = { ...adjust, ...next };
    setAdjustByCad((current) => ({ ...current, [adjustKey]: moved }));
    requestOverlay(overlayScanId, cadId, undefined, false, moved);
  };

  const requestOverlay = (scanId: string, forCadId?: string,
                          edits?: ZeroEdit[], retried = false,
                          moved?: FitAdjust) => {
    const cadId = forCadId ?? mesh?.cadId;
    setOverlayScanId(scanId);
    setOverlayError(null);
    const scan = scans.find((s) => s.id === scanId);
    const analysisId = scan?.result?.analysisId;
    if (!cadId || !analysisId) { setOverlay(null); return; }
    // CAD 를 바꿔 가며 견줄 때 같은 짝을 다시 계산하지 않는다.
    // 제로라인을 손봤으면 다른 결과이므로 열쇠에 같이 넣는다.
    const applied = edits ?? zeroEditsByScan[scanId] ?? [];
    const placed = moved ?? adjustByCad[adjustKey] ?? NO_ADJUST;
    const stamp = applied.length ? `:${JSON.stringify(applied)}` : '';
    const sameAsAuto = placed.angle === 0 && placed.dx === 0
      && placed.dy === 0 && placed.scale === 1;
    const cacheKey = `${cadId}:${analysisId}${stamp}`
      + (sameAsAuto ? '' : `:${JSON.stringify(placed)}`);
    const cached = overlayCache.current[cacheKey];
    if (cached) { setOverlay(cached); setOverlayStatus('idle'); return; }
    setOverlay(null);
    setOverlayStatus('loading');
    fetch(`${API_BASE}/api/cad-overlay`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cadId, analysisId,
                             zeroEdits: applied.length ? applied : undefined,
                             fitAdjust: sameAsAuto ? undefined : placed }),
    })
      .then(async (response) => {
        const data = await response.json() as CadOverlay & { error?: string };
        if (!response.ok) throw new Error(data.error || '올리지 못했습니다.');
        return data;
      })
      .then((data) => {
        overlayCache.current[cacheKey] = data;
        setOverlay(data); setOverlayStatus('idle');
      })
      .catch(async (err) => {
        // 서버 캐시에서 밀려났으면 조용히 다시 올리고 한 번만 더 해 본다
        const said = String(err.message || err);
        const name = meshes.find((m) => m.cadId === cadId)?.summary.name;
        if (said.includes('만료') && name && !retried) {
          const fresh = await reopenCad(name);
          if (fresh) { requestOverlay(scanId, fresh, applied, true); return; }
        }
        setOverlayError(said);
        setOverlayStatus('error');
      });
  };

  /* 여러 개를 한꺼번에 받아 차례로 읽는다. 동시에 보내면 큰 STEP 두
     개가 같은 순간 OCCT 를 물어 서버가 몇 배로 느려진다. */
  const upload = async (files: File[]) => {
    if (!files.length) return;
    setStatus('loading'); setError(null);
    setPending(files.map((f) => f.name));
    setWaited(0);
    const started = Date.now();
    const ticking = window.setInterval(
      () => setWaited(Math.round((Date.now() - started) / 1000)), 1000);
    const failed: string[] = [];
    for (const file of files) {
      const body = new FormData();
      body.append('file', file);
      try {
        const response = await fetch(`${API_BASE}/api/cad`, { method: 'POST', body });
        const data = await response.json() as CadMesh & { error?: string };
        if (!response.ok) throw new Error(data.error || '읽지 못했습니다.');
        fileStore.current[data.summary.name] = file;
        setMeshes((current) => {
          // 같은 파일을 다시 열면 갈아 끼운다 — 목록이 중복으로 불어나지
          // 않게. 이름이 같으면 같은 파일로 본다.
          const rest = current.filter((m) => m.summary.name !== data.summary.name);
          return [...rest, data];
        });
        setActiveCadId(data.cadId ?? data.summary.name);
      } catch (err) {
        failed.push(`${file.name} — ${String((err as Error).message || err)}`);
      } finally {
        setPending((current) => current.filter((n) => n !== file.name));
      }
    }
    window.clearInterval(ticking);
    setWaited(0);
    setError(failed.length ? failed.join(' / ') : null);
    setStatus(failed.length ? 'error' : 'idle');
  };

  /* 서버에서 밀려난 CAD 를 조용히 다시 올린다. 새 cadId 를 돌려준다. */
  const reopenCad = async (name: string): Promise<string | null> => {
    const file = fileStore.current[name];
    if (!file) return null;
    const body = new FormData();
    body.append('file', file);
    const response = await fetch(`${API_BASE}/api/cad`, { method: 'POST', body });
    const data = await response.json() as CadMesh & { error?: string };
    if (!response.ok) return null;
    setMeshes((current) => current.map(
      (m) => (m.summary.name === data.summary.name ? data : m)));
    if (data.cadId) setActiveCadId(data.cadId);
    return data.cadId ?? null;
  };

  /* 목록에서 뺀다. 서버 캐시는 그대로 두어도 곧 밀려난다. */
  const closeCad = (key: string) => {
    setMeshes((current) => {
      const rest = current.filter((m) => keyOf(m) !== key);
      if (key === activeCadId) setActiveCadId(rest.length ? keyOf(rest[0]) : '');
      return rest;
    });
  };

  const onPick = (event: ChangeEvent<HTMLInputElement>) => {
    void upload(Array.from(event.target.files ?? []));
    event.target.value = '';
  };
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    void upload(Array.from(event.dataTransfer.files ?? []));
  };

  const summary = mesh?.summary;
  return <section className="page page--cad" hidden={!active}>
    <div className="page-heading page-heading--compact">
      <div>
        <span className="breadcrumb">ADC · Ajin Die Compensation</span>
        <h2>3D 데이터</h2>
        <p>STEP·STL을 읽어 형상과 조립 홀을 확인합니다. 이 PC 안에서만 처리됩니다.</p>
      </div>
      <button className="primary-button" onClick={() => inputRef.current?.click()}>
        3D 파일 열기 <UploadCloud size={17} />
      </button>
      <input ref={inputRef} type="file" hidden multiple accept=".step,.stp,.stl,.ply,.obj,.glb,.gltf,.3mf" onChange={onPick} />
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
              {(meshes.length > 1 || pending.length > 0) && <div className="cad-files">
                {meshes.map((item) => {
                  const key = keyOf(item);
                  return <span key={key}
                    className={`cad-files__tab${key === activeCadId ? ' cad-files__tab--on' : ''}`}>
                    <button type="button" onClick={() => setActiveCadId(key)}
                      title={`${item.summary.name} · 삼각형 ${item.summary.n_faces.toLocaleString()}`}>
                      <Box size={13} /> {item.summary.name}
                    </button>
                    <button type="button" className="cad-files__close"
                      aria-label={`${item.summary.name} 닫기`}
                      onClick={() => closeCad(key)}><X size={12} /></button>
                  </span>;
                })}
                {pending.map((name) => <span key={name} className="cad-files__tab cad-files__tab--wait">
                  <button type="button" disabled><Play size={13} /> {name} 읽는 중…
                    {waited > 2 && ` ${waited}초`}</button>
                </span>)}
                {pending.length > 0 && waited > 8 && (
                  <span className="cad-files__note">
                    처음 읽는 STEP 은 오래 걸립니다 — 실측 113MB 가 57초입니다.
                    같은 파일을 다시 열면 3초입니다.
                  </span>
                )}
              </div>}
              <div className="cad-viewer">
                <CadViewer active={active} sections={sections} mesh={mesh} showHoles={showHoles} overlay={overlay}
                  sheetValues={sheetValues?.values ?? null}
                  onCorrectionChange={(pointId, value) =>
                    overlayScanId && onOverrideChange(overlayScanId, pointId, value)}
                  onCapture={(url) => setShots((current) => [...current, url])}
                  morph={morph} morphMode={morphMode}
                  regions={regionsByCad[notesKey] ?? []}
                  onRegionsChange={(next) => notesKey && setRegionsByCad(
                    (current) => ({ ...current, [notesKey]: next }))}
                  notes={notesByCad[notesKey] ?? []}
                  onNotesChange={(next) => notesKey && setNotesByCad(
                    (current) => ({ ...current, [notesKey]: next }))} />
              </div>
              {/* 보정시트가 적어 둔 단면 위치로 제로라인을 계산한다.
                  스캔이 없어도 되고 추정이 없다 — 시트 숫자로 CAD 를 자를 뿐이다. */}
              <div className="cad-section-bar">
                <label htmlFor="cad-sections">시트 단면 표기</label>
                <input id="cad-sections" type="text" value={sectionNotes}
                  placeholder="예: H : 300   H : 250   T : 1700"
                  onChange={(event) => setSectionNotes(event.target.value)}
                  onKeyDown={(event) => { if (event.key === 'Enter') void cutSections(); }} />
                <select aria-label="좌우" value={sectionSide}
                  onChange={(event) => setSectionSide(event.target.value as typeof sectionSide)}>
                  <option value="both">양쪽</option>
                  <option value="lh">LH</option>
                  <option value="rh">RH</option>
                </select>
                <button type="button" className="tool-button" onClick={cutSections}
                  disabled={sectionState === 'working' || !sectionNotes.trim()}>
                  {sectionState === 'working' ? '자르는 중…' : '제로라인 계산'}
                </button>
                {sections && sections.length > 0 && <>
                  <span className="cad-section-bar__ok">
                    {sections.map((s) => `${s.label}`).join(' · ')} ·
                    {' '}곡선 {sections.reduce((n, s) => n + s.polylines.length, 0)}개
                  </span>
                  <button type="button" className="tool-button"
                    onClick={() => setSections(null)}>지우기</button>
                </>}
                {sections && sections.length === 0 &&
                  <span className="cad-overlay-bar__err">그 자리에서 잘리지 않았습니다</span>}
                {sectionError && <span className="cad-overlay-bar__err">{sectionError}</span>}
                <span className="cad-section-bar__note">
                  H = 높이(Z) · T = 전후(X) · 시트 숫자로 CAD 를 자르므로 추정이 없습니다
                </span>
              </div>
              <div className="viewer-legend">
                <span>
                  {analysed.length > 0
                    ? '분석한 스캔을 골라 제로라인과 보정량을 형상 위에 올릴 수 있습니다'
                    : '스캔을 먼저 분석하면 제로라인과 보정량을 여기에 올릴 수 있습니다'}
                </span>
                <span>{summary?.n_faces.toLocaleString()} 삼각형</span>
              </div>
              {overlay && showAlign && <div className="cad-align">
                <div className="cad-align__head">
                  <b>정렬 맞추기 — 스캔 그림을 형상 위에서 밀고 돌립니다</b>
                  <span>스캔에서 읽은 제로라인·보정량이 형상의 엉뚱한 자리에
                    찍혔을 때 씁니다. 자동 정합은 겉모양만 보므로 스캔에 안
                    보이는 살이 있으면 조금 어긋납니다. 누를 때마다
                    <b>얹힘</b>이 다시 계산되니 숫자가 오르는 쪽으로
                    맞추면 됩니다.</span>
                </div>
                <div className="cad-align__rows">
                  <span className="cad-align__group">
                    <label>이동</label>
                    <button type="button" onClick={() => nudge({ dx: adjust.dx - 4 })}>←</button>
                    <button type="button" onClick={() => nudge({ dx: adjust.dx + 4 })}>→</button>
                    <button type="button" onClick={() => nudge({ dy: adjust.dy - 4 })}>↑</button>
                    <button type="button" onClick={() => nudge({ dy: adjust.dy + 4 })}>↓</button>
                    <i>{adjust.dx.toFixed(0)}, {adjust.dy.toFixed(0)} px</i>
                  </span>
                  <span className="cad-align__group">
                    <label>회전</label>
                    <button type="button" onClick={() => nudge({ angle: adjust.angle - 1 })}>↺</button>
                    <button type="button" onClick={() => nudge({ angle: adjust.angle + 1 })}>↻</button>
                    <i>{adjust.angle.toFixed(0)}°</i>
                  </span>
                  <span className="cad-align__group">
                    <label>크기</label>
                    <button type="button" onClick={() => nudge({ scale: Math.max(0.5, adjust.scale - 0.02) })}>−</button>
                    <button type="button" onClick={() => nudge({ scale: Math.min(2, adjust.scale + 0.02) })}>+</button>
                    <i>{Math.round(adjust.scale * 100)}%</i>
                  </span>
                  <button type="button" className="cad-align__reset"
                    onClick={() => nudge({ ...NO_ADJUST })}>자동으로 되돌리기</button>
                  <b className={`cad-align__rate${overlay.fit.reliable ? ' is-ok' : ''}`}>
                    얹힘 {Math.round((overlay.fit.hit_rate ?? 0) * 100)}%
                    {overlay.fit.reliable ? ' · 표시함' : ' · 기준 60% 미달'}
                  </b>
                </div>
              </div>}
              {analysed.length > 0 && <div className="cad-overlay-bar">
                <label htmlFor="cad-overlay-scan">스캔 결과 올리기</label>
                <select id="cad-overlay-scan" value={overlayScanId}
                  onChange={(e) => requestOverlay(e.target.value)}>
                  <option value="">선택 안 함</option>
                  {analysed.map((scan) => (
                    <option key={scan.id} value={scan.id}>
                      {scan.partNo || scan.name}
                      {suggested?.id === scan.id ? '  (품번 일치)' : ''}
                    </option>
                  ))}
                </select>
                {overlay && <button type="button"
                  className={`cad-align__toggle${showAlign ? ' is-on' : ''}`}
                  onClick={() => setShowAlign((v) => !v)}
                  title="제로라인·보정량이 형상의 엉뚱한 자리에 찍혔을 때 손으로 맞춥니다">
                  정렬 맞추기
                </button>}
                {overlayStatus === 'loading' && <span className="cad-overlay-bar__note">올리는 중…</span>}
                {overlayError && <span className="cad-overlay-bar__err">{overlayError}</span>}
                {overlay && <span className="cad-overlay-bar__note">
                  {'XYZ'[overlay.fit.axis]}축에서 본 그림으로 맞춤 ·
                  {/* "겹침 97%" 만 띄우면 거의 완벽한 줄 안다. 그건 볼록
                      껍질끼리 잰 값이고(구멍·오목한 곳을 메우고 비교),
                      실측 64XX2 는 껍질 96.9% · 실루엣 42.2% 다.
                      실제 기준인 얹힘 비율을 앞에 둔다. */}
                  {' '}형상에 얹힘 {Math.round((overlay.fit.hit_rate ?? 0) * 100)}%
                  {' '}(윤곽 겹침 {Math.round(overlay.fit.iou * 100)}%) ·
                  {' '}{overlay.fit.mm_per_px.toFixed(2)} mm/px ·
                  {' '}제로라인 {overlay.zeroLines.length}개
                </span>}
                {/* 제로라인이 비어 있으면 이유를 말해 준다. 대개 품번을
                    못 잡아서다 — 품번이 컬러바 범위를 고르는 열쇠라
                    파일명에 품번이 없으면 이 단계가 통째로 빈다. */}
                {overlay && overlay.zeroLines.length === 0 &&
                  <span className="cad-overlay-bar__err">
                    이 스캔에서 제로라인이 나오지 않았습니다 — 분석 작업실에서
                    품번을 지정하고 다시 분석해 보세요
                  </span>}
                {overlay && sheetValues && <>
                  <button type="button" className="tool-button"
                    onClick={makeSheet} disabled={sheetState === 'saving'}>
                    <Download size={14} />
                    {sheetState === 'saving' ? '만드는 중…'
                      : `보정시트 만들기${shots.length ? ` (화면 ${shots.length}장)` : ''}`}
                  </button>
                  {shots.length > 0 && <button type="button" className="tool-button"
                    onClick={() => setShots([])}>담은 화면 비우기</button>}
                  <button type="button" className="tool-button" onClick={buildMorph}
                    disabled={morphState === 'working'}>
                    {morphState === 'working' ? '만드는 중…' : '보정 후 형상'}
                  </button>
                  {morph && <>
                    <select aria-label="형상 비교" value={morphMode}
                      onChange={(event) =>
                        setMorphMode(event.target.value as typeof morphMode)}>
                      <option value="off">원본만</option>
                      <option value="both">겹쳐 보기</option>
                      <option value="after">보정 후만</option>
                    </select>
                    <button type="button" className="tool-button"
                      onClick={saveMorphStl}>STL 저장</button>
                    <span className="cad-overlay-bar__note">
                      최대 {morph.stats.max_shift.toFixed(2)}mm ·
                      {' '}평균 {morph.stats.mean_shift.toFixed(2)}mm ·
                      {' '}반경 {morph.stats.reach_mm.toFixed(0)}mm ·
                      {' '}포인트 {morph.points}개
                    </span>
                  </>}
                  {morphError && <span className="cad-overlay-bar__err">{morphError}</span>}
                  {sheetError && <span className="cad-overlay-bar__err">{sheetError}</span>}
                </>}
                {overlay && sheetValues && <span className="cad-overlay-bar__note">
                  보정시트 기준 · 계수 {sheetValues.coefficient.toFixed(2)}×
                  {sheetValues.overrideCount > 0
                    ? ` · 작업자 수정 ${sheetValues.overrideCount}개` : ''}
                  {sheetValues.hiddenCount > 0
                    ? ` · 숨긴 포인트 ${sheetValues.hiddenCount}개 제외` : ''}
                </span>}
                {overlay && overlay.rejected && overlay.rejected.length > 0 &&
                  <span className="cad-overlay-bar__err">
                    판독값 {overlay.rejected.length}개 제외 — 컬러바 범위
                    ±{overlay.colorbarLimit}mm 를 벗어납니다
                    ({overlay.rejected.slice(0, 4).map((r) => r.id).join(', ')}
                    {overlay.rejected.length > 4 ? ' 외' : ''})
                  </span>}
              </div>}
            </>
          : <div className="cad-drop">
              {status === 'loading'
                ? <><Play size={30} /><b>읽는 중…</b><span>큰 파일은 시간이 걸립니다.</span></>
                : <><Box size={30} /><b>3D 파일을 여기에 놓으세요</b>
                    <span>STEP · STL · PLY · OBJ · GLB · 3MF · 여러 개를 한 번에 놓아도 됩니다</span>
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

function ServicePreview({ scan, folderAvailable, hiddenPointIds, onPointToggle, onKeepKeyPointsOnly, onShowAllPoints, pointOverrides, onOverrideChange, onClearAllOverrides, annotations = [], setAnnotations, coefficient, setCoefficient, zeroEdits, onZeroEditsChange }: { scan: ScanItem; folderAvailable: boolean; hiddenPointIds: Set<string>; onPointToggle: (id: string) => void; onKeepKeyPointsOnly: () => void; onShowAllPoints: () => void; pointOverrides: Record<string, number>; onOverrideChange: (id: string, value: number | null) => void; onClearAllOverrides: () => void; annotations: Annotation[]; setAnnotations: (updater: (current: Annotation[]) => Annotation[]) => void; coefficient: number; setCoefficient: (value: number) => void;
  zeroEdits: ZeroEdit[]; onZeroEditsChange: (edits: ZeroEdit[]) => void }) {
  const result = scan.result!; const points = result.points; const [showPoints, setShowPoints] = useState(true); const [showZero, setShowZero] = useState(true);
  const [tool, setTool] = useState<AnnotationTool>('select'); const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null); const [showAnnotations, setShowAnnotations] = useState(true); const [detailMode, setDetailMode] = useState(false); const [labelAreaMode, setLabelAreaMode] = useState<'hide' | 'show' | null>(null);
  /* 엔진 결과는 그대로 두고 작업자가 찍은 포인트만 따로 얹는다. */
  /* 제로라인 손질 — 시트에서 고치고 [3D 에 적용] 을 눌러야 3D 가 따른다.
     누를 때마다 서버에 다시 물으면 느려서, 초안을 들고 있다가 한 번에
     보낸다. */
  const [draftEdits, setDraftEdits] = useState<ZeroEdit[]>(zeroEdits);
  const [zeroPanel, setZeroPanel] = useState(false);
  /* 품번을 바꾸면 그 스캔에 적용해 둔 것에서 다시 시작한다 —
     앞 부품에서 손본 값이 따라오면 안 된다. */
  useEffect(() => { setDraftEdits(zeroEdits);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scan.id]);
  const [addedPoints, setAddedPoints] = useState<PointResult[]>([]);
  const [addPointMode, setAddPointMode] = useState(false);
  const [sampling, setSampling] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const removeAddedPoint = (id: string) => setAddedPoints((current) => current.filter((item) => item.id !== id));
  const addPointAt = async (xNorm: number, yNorm: number) => {
    setSampling(true); setSampleError(null);
    try {
      const form = new FormData();
      form.append('file', scan.file);
      form.append('x', String(xNorm));
      form.append('y', String(yNorm));
      const response = await fetch(`${API_BASE}/api/sample`, { method: 'POST', body: form });
      // 엔진 서버가 옛 코드로 떠 있으면 /api/sample 이 없어 HTML 404 가 온다.
      // 그대로 json() 하면 파싱 예외라 원인이 안 보여서, 상태코드로 알려준다.
      const body = await response.text();
      let data: { error?: string; xPx: number; yPx: number; x: number; y: number; value: number };
      try {
        data = JSON.parse(body) as typeof data;
      } catch {
        data = { error: response.status === 404
          ? '보정 포인트 API를 찾을 수 없습니다. 로컬 엔진 서버를 최신 코드로 다시 시작하세요.'
          : `엔진 서버 응답을 읽을 수 없습니다. (HTTP ${response.status})` } as typeof data;
      }
      if (!response.ok) { setSampleError(data?.error || '편차값을 추정하지 못했습니다.'); return; }
      setAddedPoints((current) => [...current, {
        id: `M-${String(current.length + 1).padStart(2, '0')}`,
        xPx: data.xPx, yPx: data.yPx, x: data.x, y: data.y,
        value: data.value, labelColor: 'white', confidence: 'colormap', source: 'colormap',
      }]);
    } catch (error) {
      setSampleError(error instanceof Error ? error.message : '엔진 서버에 연결하지 못했습니다.');
    } finally {
      setSampling(false);
    }
  };
  /* 시트에는 엔진이 찾은 포인트와 작업자가 찍은 포인트를 함께 올린다.
     표시 여부도 합친 목록 기준으로 계산해야 추가한 포인트의 라벨이 숨김 처리되지 않는다. */
  const sheetPoints = [...points, ...addedPoints];
  const visiblePointIds = new Set(sheetPoints.filter((point) => !hiddenPointIds.has(point.id)).map((point) => point.id));
  const createAnnotation = (annotation: Annotation) => setAnnotations((current) => [...current, annotation]);
  const commitAnnotation = (annotation: Annotation) => setAnnotations((current) => current.map((item) => item.id === annotation.id ? annotation : item));
  const deleteAnnotation = (id: string) => setAnnotations((current) => current.filter((item) => item.id !== id));
  const clearAnnotations = () => { setAnnotations(() => []); setSelectedAnnotationId(null); setTool('select'); };
  /* 브라우저 인쇄를 그대로 쓴다. 캔버스로 굽지 않아 글자가 벡터로 남고 추가 의존성도 없다.
     인쇄 대화상자에서 '대상: PDF로 저장'을 고르면 된다. */
  /* 현업 엑셀 양식으로 내보낸다. 보정량은 화면이 들고 있는 최종값을
     그대로 보내야 시트와 엑셀이 어긋나지 않는다. */
  const [excelState, setExcelState] = useState<'idle' | 'saving' | 'error'>('idle');
  const [excelError, setExcelError] = useState<string | null>(null);
  const saveExcel = async () => {
    const analysisId = result.analysisId;
    if (!analysisId) { setExcelError('분석 결과가 없습니다.'); return; }
    setExcelState('saving'); setExcelError(null);
    const corrections: Record<string, number> = {};
    for (const point of sheetPoints) {
      if (!visiblePointIds.has(point.id)) continue;
      corrections[point.id] = displayFor(point);
    }
    try {
      const response = await fetch(`${API_BASE}/api/sheet-excel`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId, corrections,
          filename: `${scan.partNo || 'ADC'}_보정시트`,
          /* 머리말은 파일명 규칙에서 읽은 값을 그대로 보낸다.
             화면에 보이는 것과 엑셀이 달라지면 안 된다. */
          meta: {
            partNo: result.naming?.part_no || scan.partNo,
            partName: result.naming?.part_name || '',
            process: result.naming?.process || '',
            controlNo: result.naming?.control_no || '',
            appliedAt: result.naming?.applied_at || '',
            coefficient,
          },
        }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(data.error || '엑셀을 만들지 못했습니다.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${scan.partNo || 'ADC'}_보정시트.xlsx`;
      link.click();
      URL.revokeObjectURL(url);
      setExcelState('idle');
    } catch (err) {
      setExcelError(String((err as Error).message || err));
      setExcelState('error');
    }
  };

  const sheetRef = useRef<HTMLDivElement>(null);
  const savePdf = () => {
    setSelectedAnnotationId(null);
    setTool('select');
    setDetailMode(false);
    setLabelAreaMode(null);
    /* 시트의 조상만 남기고 형제는 인쇄에서 빼야 한다. visibility 로 감추면 자리를
       그대로 차지해 빈 둘째 장이 생긴다. */
    const chain: HTMLElement[] = [];
    for (let node = sheetRef.current?.parentElement; node && node !== document.body; node = node.parentElement) {
      node.classList.add('adc-print-chain');
      chain.push(node);
    }
    document.body.classList.add('adc-printing');
    const cleanup = () => {
      document.body.classList.remove('adc-printing');
      chain.forEach((node) => node.classList.remove('adc-print-chain'));
      window.removeEventListener('afterprint', cleanup);
    };
    window.addEventListener('afterprint', cleanup);
    /* 전환이 끝나 자리가 확정된 뒤에 인쇄해야 중간값이 찍히지 않는다. */
    window.setTimeout(() => window.print(), 80);
  };
  /* 새 주석은 늘 기본색으로 그려지고, 색 변경은 주석을 고른 뒤 팔레트를 누르는 동작으로만 일어난다. */
  const selectedColor = selectedAnnotationId ? (annotations.find((item) => item.id === selectedAnnotationId)?.color ?? DEFAULT_ANNOTATION_COLOR) : null;
  const changeColor = (hex: string) => {
    if (!selectedAnnotationId) return;
    setAnnotations((current) => current.map((item) => item.id === selectedAnnotationId ? { ...item, color: hex } : item));
  };
  const displayFor = (point: PointResult) => pointOverrides[point.id] !== undefined ? pointOverrides[point.id] : -(point.value * coefficient);
  /* 시트에 실제로 실린 포인트만 센다. 예전에는 숨긴 것까지 포함해서,
     핵심 15개만 남겨 놓고도 "최대 보정량 9.000mm" 처럼 시트에 없는 값이
     떴다 — 그 9.0 은 컬러바 범위(+-3.0)를 벗어난 판독 오류였다. */
  const maxCorrection = useMemo(() => {
    const shown = points.filter((point) => visiblePointIds.has(point.id));
    return shown.length
      ? Math.max(...shown.map((point) => Math.abs(displayFor(point)))) : 0;
  }, [coefficient, points, pointOverrides, visiblePointIds]);
  const overrideCount = useMemo(() => points.filter((point) => pointOverrides[point.id] !== undefined).length, [points, pointOverrides]);
  const baseImage = showZero && result.zeroOverlay ? result.zeroOverlay : result.cleanImage || scan.url;
  return <section className="page page--service">
    <div className="page-heading page-heading--compact"><div><span className="breadcrumb">ADC · Ajin Die Compensation <span className="demo-badge">DEMO</span></span><h2>ADC 금형 보정 시트</h2><p>흰 시트 위에 정면도와 Detail View를 독립 레이아웃으로 구성합니다.</p></div></div>
    <div className="service-grid"><div className="correction-card card">
      <div className="viewer-toolbar"><div><span className="status status--done"><Check size={13} /> 레이아웃 편집</span><b>{scan.partNo} · 보정 작업 지시도</b></div><div className="layer-toggles"><button className={showPoints ? 'active orange' : ''} onClick={() => setShowPoints(!showPoints)}><i /> 보정치</button><button className={showZero ? 'active green' : ''} onClick={() => setShowZero(!showZero)} disabled={!zeroLinesToShow(result).length}><i /> 제로라인</button><button className={zeroPanel ? 'active green' : ''} onClick={() => setZeroPanel((v) => !v)} disabled={!zeroLinesToShow(result).length} title="제로라인을 옮기거나 숨깁니다">제로라인 수정</button><button className={showAnnotations ? 'active amber' : ''} onClick={() => { setShowAnnotations(!showAnnotations); setTool('select'); setSelectedAnnotationId(null); }}><i /> 주석</button></div></div>
      <AnnotationToolbar tool={tool} setTool={(next) => { setShowAnnotations(true); setTool(next); setDetailMode(false); setLabelAreaMode(null); if (next !== 'select') setSelectedAnnotationId(null); }} hasAnnotations={annotations.length > 0} onClearAll={clearAnnotations} selectedColor={selectedColor} onColorChange={changeColor} detailMode={detailMode} onDetailMode={() => { setDetailMode(!detailMode); setLabelAreaMode(null); setTool('select'); setSelectedAnnotationId(null); }} labelAreaMode={labelAreaMode} onLabelAreaMode={(mode) => { setLabelAreaMode((current) => current === mode ? null : mode); setDetailMode(false); setAddPointMode(false); setTool('select'); setSelectedAnnotationId(null); }} addPointMode={addPointMode} onAddPointMode={() => { setAddPointMode(!addPointMode); setDetailMode(false); setLabelAreaMode(null); setTool('select'); setSelectedAnnotationId(null); setSampleError(null); }} />
      {zeroPanel && <div className="zero-edit">
        <div className="zero-edit__head">
          <b>제로라인 수정</b>
          <span>그림 좌표로 옮깁니다 · 적용하면 3D 시각화도 같이 따라갑니다</span>
        </div>
        {zeroLinesToShow(result).map((line, index) => {
          const edit = draftEdits.find((e) => e.index === index)
            ?? { index, dx: 0, dy: 0 };
          const put = (next: ZeroEdit) => setDraftEdits((current) => [
            ...current.filter((e) => e.index !== index), next]
            .sort((a, b) => a.index - b.index));
          const nudge = (dx: number, dy: number) =>
            put({ ...edit, dx: edit.dx + dx, dy: edit.dy + dy });
          return <div key={index}
            className={`zero-edit__row${edit.hidden ? ' is-off' : ''}`}>
            <b>제로라인 {index + 1}</b>
            <span className="zero-edit__axis">
              가로
              <button type="button" onClick={() => nudge(-5, 0)}>−</button>
              <i>{edit.dx.toFixed(0)}</i>
              <button type="button" onClick={() => nudge(5, 0)}>+</button>
            </span>
            <span className="zero-edit__axis">
              세로
              <button type="button" onClick={() => nudge(0, -5)}>−</button>
              <i>{edit.dy.toFixed(0)}</i>
              <button type="button" onClick={() => nudge(0, 5)}>+</button>
            </span>
            <button type="button"
              onClick={() => put({ ...edit, hidden: !edit.hidden })}>
              {edit.hidden ? '다시 보이기' : '숨기기'}
            </button>
            <button type="button"
              onClick={() => setDraftEdits((c) => c.filter((e) => e.index !== index))}>
              되돌리기
            </button>
          </div>;
        })}
        <div className="zero-edit__foot">
          <button type="button" className="zero-edit__apply"
            onClick={() => onZeroEditsChange(draftEdits)}>3D 에 적용</button>
          <button type="button" onClick={() => { setDraftEdits([]); onZeroEditsChange([]); }}>
            전부 되돌리기
          </button>
          {JSON.stringify(draftEdits) !== JSON.stringify(zeroEdits) && (
            <em>아직 3D 에 적용하지 않았습니다</em>
          )}
        </div>
      </div>}
      <div className="sheet-page" ref={sheetRef}><SheetTitleBlock scan={scan} /><div className="sheet-stage sheet-stage--light"><SheetCanvas key={scan.id} scan={scan} imageUrl={baseImage} points={sheetPoints} coefficient={coefficient} showPoints={showPoints} showZero={showZero} visiblePointIds={visiblePointIds} onPointToggle={onPointToggle} pointOverrides={pointOverrides} onOverrideChange={onOverrideChange} annotations={annotations} showAnnotations={showAnnotations} annotationTool={tool} setAnnotationTool={setTool} selectedAnnotationId={selectedAnnotationId} setSelectedAnnotationId={setSelectedAnnotationId} onAnnotationCommit={commitAnnotation} onAnnotationCreate={createAnnotation} onAnnotationDelete={deleteAnnotation} detailMode={detailMode} setDetailMode={setDetailMode} labelAreaMode={labelAreaMode} setLabelAreaMode={setLabelAreaMode} addPointMode={addPointMode} onAddPointAt={addPointAt} sampling={sampling} sampleError={sampleError} addedPoints={addedPoints} onRemoveAddedPoint={removeAddedPoint} zeroEdits={draftEdits} /><div className="sheet-stamp sheet-stamp--paper"><span>AJIN INDUSTRIAL</span><b>DIE CORRECTION SHEET</b><small>{scan.partNo} · REV.01</small></div></div></div>
      <div className="sheet-note"><ShieldCheck size={17} /><span><b>레이아웃의 제목 막대를 끌어 이동하고, 선택 테두리의 핸들 또는 W/H 슬라이더로 크기를 조절할 수 있습니다.</b></span><button type="button" className="sheet-print" onClick={savePdf}><Printer size={14} /> 보정 시트 PDF 저장</button><button type="button" className="sheet-print" onClick={saveExcel} disabled={excelState === 'saving'}><Download size={14} /> {excelState === 'saving' ? '엑셀 만드는 중…' : '기업 양식 엑셀 저장'}</button>{excelError && <span className="sheet-excel-error">{excelError}</span>}</div>
    </div><aside className="control-panel"><div className="card coefficient-card"><div className="card-title"><div><h3>보정 계수</h3><p>편차값에 곱할 비율을 조절합니다.</p></div><span>{coefficient.toFixed(2)}×</span></div><div className="coefficient-input"><input aria-label="보정 계수 직접 입력" type="number" min="0.5" max="1.5" step="0.01" value={coefficient} onChange={(e) => { const value = e.target.valueAsNumber; if (!Number.isNaN(value)) setCoefficient(Math.max(0.5, Math.min(1.5, value))); }} /><span>×</span></div><input aria-label="보정 계수" type="range" min="0.5" max="1.5" step="0.05" value={coefficient} onChange={(e) => setCoefficient(Number(e.target.value))} /><div className="range-labels"><span>보수적 0.50</span><span>기준 1.00</span><span>적극적 1.50</span></div><div className="formula"><span>보정치</span><b>= 편차 × {coefficient.toFixed(2)} × (−1)</b></div>{overrideCount > 0 && <p className="coefficient-note">수정된 {overrideCount}개 포인트는 계수 영향을 받지 않습니다.</p>}</div>{(result.keyPoints?.length ?? 0) > 0 && <div className="card key-point-card">
      <div className="card-title"><div><h3>핵심 포인트</h3>
        <p>시트에 적을 만한 것만 골랐습니다.</p></div>
        <span>{result.keyPoints!.length}개</span></div>
      <p className="key-point-card__why">
        전체 {result.points.length}개 중
        {(result.keyPointsRejected?.length ?? 0) > 0
          ? ` 판독 범위를 벗어난 ${result.keyPointsRejected!.length}개를 빼고,` : ''}
        {' '}편차가 크고 주변보다 도드라지며 서로 떨어진 자리를 골랐습니다.
      </p>
      <div className="key-point-card__list">
        {result.keyPoints!.slice(0, 6).map((k) => <div key={k.point_id}>
          <b>{k.point_id}</b>
          <span className={k.value < 0 ? 'blue' : 'orange'}>
            {k.value > 0 ? '+' : ''}{k.value.toFixed(1)} mm</span>
          <small>{k.reason}</small>
        </div>)}
        {result.keyPoints!.length > 6 &&
          <div className="key-point-card__more">외 {result.keyPoints!.length - 6}개</div>}
      </div>
      <div className="key-point-card__buttons">
        <button type="button" onClick={onKeepKeyPointsOnly}>핵심만 남기기</button>
        <button type="button" onClick={onShowAllPoints}>전체 되돌리기</button>
      </div>
      <p className="key-point-card__note">
        선별 기준은 아직 현업 확인 전입니다 — 골라낸 뒤 개별로 다시 켜고 끌 수 있습니다.
      </p>
    </div>}
    <div className="card correction-summary"><h3>실제 엔진 요약</h3><div><span>보정 포인트</span><b>{visiblePointIds.size}개</b></div>{overrideCount > 0 && <div><span>수정된 포인트</span><b className="blue">{overrideCount}개</b></div>}<div><span>최대 보정량</span><b className="orange">{maxCorrection.toFixed(3)} mm</b></div><div><span>제로라인</span><b className="green">{result.stats.zeroRegions}개 영역</b></div><div><span>처리 품번</span><b>{scan.partNo}</b></div>{overrideCount > 0 && <button type="button" className="reset-all-overrides" onClick={onClearAllOverrides}>모든 수정 취소</button>}</div></aside></div>{folderAvailable && <Explorer />}
  </section>;
}

export default function Home() {
  const [view, setView] = useState<View>('workspace'); const [scans, setScans] = useState<ScanItem[]>([]); const [activeId, setActiveId] = useState<string>(); const [collapsed, setCollapsed] = useState(false); const [backendOnline, setBackendOnline] = useState<boolean | null>(null); const [folderAvailable, setFolderAvailable] = useState(false); const [hiddenPointIdsByScan, setHiddenPointIdsByScan] = useState<Record<string, Set<string>>>({}); const [pointOverridesByScan, setPointOverridesByScan] = useState<Record<string, Record<string, number>>>({}); const [annotationsByScan, setAnnotationsByScan] = useState<Record<string, Annotation[]>>({}); /* 보정 계수는 시트에만 있으면 3D 표시가 시트와 어긋난다. 스캔별로 위에서 들고 있는다. */ const [coefficientByScan, setCoefficientByScan] = useState<Record<string, number>>({}); /* 3D 주석과 공정 구역. CAD 파일 이름을 열쇠로 쓴다 — cadId 는 열 때마다 새로 생겨 새로고침을 못 넘긴다. */ const [notesByCad, setNotesByCad] = useState<Record<string, CadNote[]>>({}); const [regionsByCad, setRegionsByCad] = useState<Record<string, CadRegion[]>>({});
  /* 시트에서 손본 제로라인. 스캔별로 위에서 들고 있어야 시트와 3D 가
     같은 것을 본다. */
  const [zeroEditsByScan, setZeroEditsByScan] = useState<Record<string, ZeroEdit[]>>({});
  /* 작업 내용을 이 PC 에 남긴다 — 새로고침으로 날아가면 안 된다.
     스캔 아이디는 파일을 다시 올릴 때마다 바뀌므로 품번으로 묶는다. */
  const [sessionNote, setSessionNote] = useState<string | null>(null);
  const sessionRef = useRef<SessionSnapshot>(emptySession());
  const sessionReady = useRef(false);
  const sessionFileRef = useRef<HTMLInputElement>(null);
  const [valleyLinesByScan, setValleyLinesByScan] = useState<Record<string, ValleyLine[]>>({});
  useEffect(() => { fetch(`${API_BASE}/api/health`).then((response) => response.json() as Promise<HealthResponse>).then((data) => { setBackendOnline(Boolean(data.ok)); setFolderAvailable(Boolean(data.folderAvailable)); }).catch(() => setBackendOnline(false)); }, []);
  const resolvedActiveId = activeId || scans[0]?.id;
  const activeScan = scans.find((scan) => scan.id === resolvedActiveId); const completedScan = activeScan?.result ? activeScan : scans.find((scan) => scan.result); const hasResult = Boolean(completedScan?.result);
  const hiddenPointIds = completedScan ? hiddenPointIdsByScan[completedScan.id] || new Set<string>() : new Set<string>();
  const pointOverrides = completedScan ? pointOverridesByScan[completedScan.id] || {} : {};
  const coefficient = completedScan ? coefficientByScan[completedScan.id] ?? 1 : 1;
  const setCoefficient = (value: number) => {
    if (!completedScan) return;
    setCoefficientByScan((current) => ({ ...current, [completedScan.id]: value }));
  };
  const togglePoint = (id: string) => completedScan && setHiddenPointIdsByScan((current) => { const next = new Set(current[completedScan.id] || []); if (next.has(id)) next.delete(id); else next.add(id); return { ...current, [completedScan.id]: next }; });
  const setAllPointsVisible = (visible: boolean) => completedScan && setHiddenPointIdsByScan((current) => ({ ...current, [completedScan.id]: visible ? new Set() : new Set(completedScan.result!.points.map((point) => point.id)) }));
  /* 핵심 포인트만 남긴다. 새 상태를 만들지 않고 기존 "숨김" 기능을 그대로
     쓴다 — 그래야 시트와 3D 가 같은 것을 보고, 작업자가 개별로 다시
     꺼낼 수도 있다. */
  const keepKeyPointsOnly = () => {
    if (!completedScan?.result) return;
    const keep = new Set((completedScan.result.keyPoints ?? []).map((k) => k.point_id));
    if (!keep.size) return;
    setHiddenPointIdsByScan((current) => ({
      ...current,
      [completedScan.id]: new Set(
        completedScan.result!.points.filter((p) => !keep.has(p.id)).map((p) => p.id)),
    }));
  };
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
  /* 분석이 끝나면 **핵심 포인트만 남긴 상태로 시작**한다.
     현업 보정시트를 보면 콜아웃이 열몇 개뿐인데 우리는 검출된 것을 전부
     찍고 있었다(실측 71XX2 는 40개, 67XX6 은 67개). 그러면 형상이
     콜아웃에 덮여 시트로 못 쓴다.

     저장해 둔 작업이 있으면 건드리지 않는다 — 작업자가 고른 것이
     우선이다. "전체 되돌리기" 로 언제든 다 볼 수 있다. */
  useEffect(() => {
    if (!sessionReady.current) return;
    for (const scan of scans) {
      const keys = scan.result?.keyPoints;
      if (!keys?.length) continue;
      if (hiddenPointIdsByScan[scan.id] !== undefined) continue;
      if (sessionRef.current.byPart[scan.partNo]?.hidden) continue;
      const keep = new Set(keys.map((k) => k.point_id));
      setHiddenPointIdsByScan((current) => ({
        ...current,
        [scan.id]: new Set(
          scan.result!.points.filter((p) => !keep.has(p.id)).map((p) => p.id)),
      }));
    }
  }, [scans, hiddenPointIdsByScan]);

  const valleyLines = completedScan ? valleyLinesByScan[completedScan.id] || [] : [];
  const setValleyLines = (updater: ValleyLine[] | ((current: ValleyLine[]) => ValleyLine[])) => completedScan && setValleyLinesByScan((current) => {
    const previous = current[completedScan.id] || [];
    const next = typeof updater === 'function' ? (updater as (value: ValleyLine[]) => ValleyLine[])(previous) : updater;
    return { ...current, [completedScan.id]: next };
  });
  /* 3D 탭은 보고 있는 스캔이 시트와 다를 수 있어 스캔을 지정해 고친다.
     시트와 같은 저장소라 어느 쪽에서 고쳐도 양쪽에 함께 반영된다. */
  /* 스캔 아이디 -> 품번. 저장은 품번 기준이라 매번 이걸로 옮긴다. */
  const partOf = (scanId: string) =>
    scans.find((scan) => scan.id === scanId)?.partNo || scanId;

  /* 저장해 둔 작업을 스캔이 올라오는 대로 되살린다. 같은 품번이 다시
     분석되면 그때 붙인다 — 스캔 아이디는 매번 새로 생기기 때문이다. */
  useEffect(() => {
    if (sessionReady.current) return;
    const saved = loadSession();
    if (!saved) { sessionReady.current = true; return; }
    sessionRef.current = saved;
    sessionReady.current = true;
    /* 3D 주석·공정 구역은 CAD 파일 이름이 열쇠라 스캔을 기다릴 필요가
       없다 — 바로 되살린다. 같은 이름의 파일을 다시 열면 붙는다. */
    const notes: Record<string, CadNote[]> = {};
    const regions: Record<string, CadRegion[]> = {};
    for (const [name, entry] of Object.entries(saved.byCad ?? {})) {
      if (entry.notes) notes[name] = entry.notes as CadNote[];
      if (entry.regions) regions[name] = entry.regions as CadRegion[];
    }
    if (Object.keys(notes).length) setNotesByCad(notes);
    if (Object.keys(regions).length) setRegionsByCad(regions);
    const parts = Object.keys(saved.byPart).length;
    const cads = Object.keys(saved.byCad ?? {}).length;
    if (parts || cads) setSessionNote(
      `저장된 작업을 불러왔습니다 (품번 ${parts}개${cads ? ` · 3D ${cads}개` : ''})`);
  }, []);

  useEffect(() => {
    if (!sessionReady.current) return;
    const saved = sessionRef.current;
    let touched = false;
    for (const scan of scans) {
      if (!scan.result) continue;
      const entry = saved.byPart[scan.partNo];
      if (!entry) continue;
      if (entry.coefficient !== undefined
          && coefficientByScan[scan.id] === undefined) {
        setCoefficientByScan((c) => ({ ...c, [scan.id]: entry.coefficient! }));
        touched = true;
      }
      if (entry.overrides && !pointOverridesByScan[scan.id]) {
        setPointOverridesByScan((c) => ({ ...c, [scan.id]: entry.overrides! }));
        touched = true;
      }
      if (entry.hidden && !hiddenPointIdsByScan[scan.id]) {
        setHiddenPointIdsByScan((c) => ({ ...c, [scan.id]: new Set(entry.hidden!) }));
        touched = true;
      }
    }
    if (touched) setSessionNote('저장해 둔 보정 내용을 되살렸습니다');
  }, [scans, coefficientByScan, pointOverridesByScan, hiddenPointIdsByScan]);

  /* 바뀔 때마다 남긴다. 저장이 막혀 있어도 작업은 계속되어야 한다. */
  useEffect(() => {
    if (!sessionReady.current) return;
    const snapshot = { ...sessionRef.current, byPart: { ...sessionRef.current.byPart } };
    for (const scan of scans) {
      if (!scan.result) continue;
      const overrides = pointOverridesByScan[scan.id];
      const hidden = hiddenPointIdsByScan[scan.id];
      const coefficient = coefficientByScan[scan.id];
      if (!overrides && !hidden && coefficient === undefined) continue;
      snapshot.byPart[partOf(scan.id)] = {
        coefficient,
        overrides: overrides ? { ...overrides } : undefined,
        hidden: hidden ? [...hidden] : undefined,
      };
    }
    /* 3D 주석·공정 구역. 빈 것은 담지 않는다 — 지운 뒤에도 되살아나면
       안 되므로, 내용이 있는 CAD 만 남긴다. */
    snapshot.byCad = {};
    for (const name of new Set([...Object.keys(notesByCad),
                                ...Object.keys(regionsByCad)])) {
      const notes = notesByCad[name] ?? [];
      const regions = regionsByCad[name] ?? [];
      if (!notes.length && !regions.length) continue;
      snapshot.byCad[name] = {
        notes: notes.length ? notes : undefined,
        regions: regions.length ? regions : undefined,
      };
    }
    sessionRef.current = snapshot;
    saveSession(snapshot);
  }, [scans, pointOverridesByScan, hiddenPointIdsByScan, coefficientByScan,
      notesByCad, regionsByCad]);

  const setPointOverrideFor = (scanId: string, id: string, value: number | null) => setPointOverridesByScan((current) => { const next = { ...(current[scanId] || {}) }; if (value === null) delete next[id]; else next[id] = value; return { ...current, [scanId]: next }; });
  const setPointOverride = (id: string, value: number | null) => { if (completedScan) setPointOverrideFor(completedScan.id, id, value); };
  const clearAllOverrides = () => completedScan && setPointOverridesByScan((current) => ({ ...current, [completedScan.id]: {} }));
  const annotations = completedScan ? annotationsByScan[completedScan.id] || [] : [];
  const setAnnotations = (updater: (current: Annotation[]) => Annotation[]) => completedScan && setAnnotationsByScan((current) => ({ ...current, [completedScan.id]: updater(current[completedScan.id] || []) }));
  const openResults = (id: string) => { setActiveId(id); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const selectView = (next: View) => { if (next === 'workspace' || next === 'cad' || hasResult) setView(next); };
  return <main className={`app-shell ${collapsed ? 'app-shell--collapsed' : ''}`}><Sidebar view={view} setView={selectView} collapsed={collapsed} setCollapsed={setCollapsed} hasResult={hasResult} /><div className="app-main"><Header scans={scans} activeId={resolvedActiveId} setActiveId={setActiveId} note={sessionNote}
      onSaveFile={() => downloadSession(sessionRef.current)}
      onLoadFile={async (file) => {
        const parsed = await readSessionFile(file);
        if (!parsed) { setSessionNote('읽을 수 없는 파일입니다'); return; }
        sessionRef.current = parsed;
        saveSession(parsed);
        /* 이미 올라온 스캔에 바로 붙인다 */
        for (const scan of scans) {
          const entry = parsed.byPart[scan.partNo];
          if (!entry) continue;
          if (entry.coefficient !== undefined) setCoefficientByScan((c) => ({ ...c, [scan.id]: entry.coefficient! }));
          if (entry.overrides) setPointOverridesByScan((c) => ({ ...c, [scan.id]: entry.overrides! }));
          if (entry.hidden) setHiddenPointIdsByScan((c) => ({ ...c, [scan.id]: new Set(entry.hidden!) }));
        }
        setSessionNote(`${Object.keys(parsed.byPart).length}개 품번을 불러왔습니다`);
      }}
      onReset={() => {
        clearSession();
        sessionRef.current = emptySession();
        setPointOverridesByScan({}); setHiddenPointIdsByScan({});
        setCoefficientByScan({});
        setSessionNote('작업 내용을 비웠습니다');
      }} />{view === 'workspace' && <Workspace scans={scans} setScans={setScans} onOpenResults={openResults} backendOnline={backendOnline} />}{view === 'results' && completedScan?.result && <Results scan={completedScan} onService={() => setView('service')} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} onAllPointsToggle={setAllPointsVisible} valleyLines={valleyLines} setValleyLines={setValleyLines} />}{view === 'service' && completedScan?.result && <ServicePreview scan={completedScan} folderAvailable={folderAvailable} hiddenPointIds={hiddenPointIds} onPointToggle={togglePoint} onKeepKeyPointsOnly={keepKeyPointsOnly} onShowAllPoints={() => setAllPointsVisible(true)} pointOverrides={pointOverrides} onOverrideChange={setPointOverride} onClearAllOverrides={clearAllOverrides} annotations={annotations} setAnnotations={setAnnotations} coefficient={coefficient} setCoefficient={setCoefficient}
        zeroEdits={completedScan ? zeroEditsByScan[completedScan.id] ?? [] : []}
        onZeroEditsChange={(edits) => completedScan && setZeroEditsByScan(
          (current) => ({ ...current, [completedScan.id]: edits }))} />}{/* 3D 화면은 조건부로 그리지 않는다. 탭을 옮길 때마다 컴포넌트가
        통째로 사라져서 **읽어 둔 CAD 가 초기화됐다** — 215MB STEP 을
        다시 읽으라는 얘기다. 감추기만 하고 상태는 살려 둔다.
        카메라 위치와 오버레이까지 그대로 돌아온다. */}
      <CadWorkspace active={view === 'cad'} scans={scans} coefficientByScan={coefficientByScan} hiddenPointIdsByScan={hiddenPointIdsByScan} pointOverridesByScan={pointOverridesByScan} onOverrideChange={setPointOverrideFor} notesByCad={notesByCad} setNotesByCad={setNotesByCad} regionsByCad={regionsByCad} setRegionsByCad={setRegionsByCad}
        zeroEditsByScan={zeroEditsByScan} /></div></main>;
}
