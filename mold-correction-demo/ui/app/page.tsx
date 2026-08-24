'use client';

/* Blob/data URLs are local engine outputs and cannot use Next's remote image loader. */
/* eslint-disable @next/next/no-img-element */

import {
  Activity, ArrowLeft, ArrowRight, BarChart3, Check, ChevronDown, ChevronRight,
  CircleHelp, Eye, EyeOff, File, Folder, FolderOpen, Gauge, Grid2X2, Image as ImageIcon,
  Layers3, ListFilter, Maximize2, MoveRight, PanelLeftClose, Play, Settings2,
  ShieldCheck, Sparkles, UploadCloud, X, ZoomIn, ZoomOut,
} from 'lucide-react';
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

type View = 'workspace' | 'results' | 'service';
type Engine = 'label' | 'deviation' | 'zero';
type ScanStatus = 'ready' | 'analyzing' | 'done' | 'error';
type PointResult = { id: string; xPx: number; yPx: number; x: number; y: number; value: number; labelColor: string; confidence: string };
type ZeroAnchor = { anchor_id: number; x: number; y: number; boundary_arclen: number };
type ValleyLine = { id: string; anchorStartId: number; anchorEndId: number; points: [number, number][]; length_px: number; mean_abs_deviation: number };
type AnalysisResult = {
  analysisId: string | null;
  source: { name: string; width: number; height: number };
  cleanImage: string | null;
  zeroOverlay: string | null;
  zeroMask: string | null;
  zeroAnchors: ZeroAnchor[];
  points: PointResult[];
  stats: { labelsRemoved: number; pointsDetected: number; zeroRegions: number; zeroRatio: number; zeroTolerance: number | null; qwenReads: number; fallbackReads: number };
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

function Heatmap({ imageUrl, width, height, children }: { imageUrl?: string | null; width: number; height: number; children?: React.ReactNode }) {
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

  return <div
    ref={viewportRef}
    className={`heatmap heatmap--actual ${scale > 1 ? 'heatmap--zoomed' : ''} ${dragging ? 'heatmap--dragging' : ''}`}
    onWheel={(event) => { event.preventDefault(); setZoom(scale + (event.deltaY < 0 ? 0.25 : -0.25)); }}
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
  return <div className="point-layer">{points.map((point) => {
    const correction = -(point.value * coefficient);
    const labelVisible = !visibleLabelIds || visibleLabelIds.has(point.id);
    return <button type="button" className={`measure-point ${correction >= 0 ? 'measure-point--plus' : 'measure-point--minus'} ${onLabelToggle ? 'measure-point--interactive' : ''}`} style={{ left: `${point.x}%`, top: `${point.y}%` }} key={point.id} onClick={() => onLabelToggle?.(point.id)} aria-label={`${point.id} 라벨 ${labelVisible ? '숨기기' : '표시하기'}`} aria-pressed={labelVisible} title={`${point.id} 편차 ${point.value > 0 ? '+' : ''}${point.value.toFixed(3)} mm · 클릭하여 라벨 ${labelVisible ? 'OFF' : 'ON'}`}>
      <span className="measure-point__dot" />
      {labels && labelVisible && <span className="measure-point__label"><b>{point.id}</b>{correction > 0 ? '+' : ''}{correction.toFixed(3)} mm</span>}
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

function ValleyLineOverlay({ lines, width, height }: { lines: ValleyLine[]; width: number; height: number }) {
  if (!lines.length) return null;
  return <svg className="valley-line-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
    {lines.map((line) => <polyline key={line.id} points={line.points.map(([x, y]) => `${x},${y}`).join(' ')} fill="none" stroke="#e0303f" strokeWidth={Math.max(width, height) * 0.0026} strokeLinecap="round" strokeLinejoin="round" />)}
  </svg>;
}

function Sidebar({ view, setView, collapsed, setCollapsed, hasResult }: { view: View; setView: (view: View) => void; collapsed: boolean; setCollapsed: (value: boolean) => void; hasResult: boolean }) {
  const items = [
    { id: 'workspace' as const, label: '분석 작업실', icon: Grid2X2 },
    { id: 'results' as const, label: '엔진 결과', icon: BarChart3 },
    { id: 'service' as const, label: '보정 시트', icon: Layers3 },
  ];
  return <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
    <div className="brand"><div className="brand__mark">A</div><div className="brand__copy"><strong>AJIN</strong><span>Die Insight</span></div></div>
    <nav className="sidebar__nav" aria-label="주 메뉴"><span className="sidebar__eyebrow">WORKSPACE</span>{items.map((item) => { const Icon = item.icon; const disabled = item.id !== 'workspace' && !hasResult; return <button key={item.id} disabled={disabled} onClick={() => !disabled && setView(item.id)} className={view === item.id ? 'active' : ''}><Icon size={19} /><span>{item.label}</span></button>; })}</nav>
    <div className="sidebar__guide"><CircleHelp size={19} /><div><b>실제 엔진 연결</b><span>모든 처리는 이 PC 안에서 실행</span></div><ChevronRight size={16} /></div>
    <button className="sidebar__collapse" onClick={() => setCollapsed(!collapsed)} aria-label="사이드바 접기"><PanelLeftClose size={18} /><span>메뉴 접기</span></button>
  </aside>;
}

function Header({ scans, activeId, setActiveId }: { scans: ScanItem[]; activeId?: string; setActiveId: (id: string) => void }) {
  return <header className="topbar"><div><span className="topbar__context">금형생산팀 · 통합 보정 분석</span><h1>3D 스캔 보정 워크벤치</h1></div><div className="topbar__actions">
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
  if (engine === 'deviation') return { stat: `${result.stats.pointsDetected}개`, detail: `Qwen 판독 ${result.stats.qwenReads}개 · 대체 판독 ${result.stats.fallbackReads}개` };
  return { stat: `${result.stats.zeroRegions}개`, detail: `부품 면적의 ${(result.stats.zeroRatio * 100).toFixed(1)}% · 실제 검출 결과` };
}

function Results({ scan, onService }: { scan: ScanItem; onService: () => void }) {
  const [engine, setEngine] = useState<Engine>('label');
  const result = scan.result!; const meta = engineMeta[engine]; const summary = engineSummary(engine, result);
  const engineWarnings = result.warningsByEngine?.[engine] ?? (engine === 'zero' ? result.warnings : []);
  const [visibleLabelIds, setVisibleLabelIds] = useState<Set<string>>(() => new Set(result.points.map((point) => point.id)));
  const image = engine === 'zero' ? result.zeroOverlay : result.cleanImage || scan.url;
  const toggleLabel = (id: string) => setVisibleLabelIds((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const allLabelsVisible = result.points.length > 0 && visibleLabelIds.size === result.points.length;

  const zeroAnchors = result.zeroAnchors || [];
  const [selectedAnchors, setSelectedAnchors] = useState<number[]>([]);
  const [valleyLines, setValleyLines] = useState<ValleyLine[]>([]);
  const [valleyStatus, setValleyStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [valleyError, setValleyError] = useState<string | null>(null);
  useEffect(() => { setSelectedAnchors([]); setValleyLines([]); setValleyStatus('idle'); setValleyError(null); }, [scan.id]);
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
        setValleyLines((current) => [...current, {
          id: `${line.anchor_start_id}-${line.anchor_end_id}-${current.length}`,
          anchorStartId: line.anchor_start_id, anchorEndId: line.anchor_end_id,
          points: line.points, length_px: line.length_px, mean_abs_deviation: line.mean_abs_deviation,
        }]);
        setValleyStatus('idle'); setSelectedAnchors([]);
      })
      .catch(() => { if (!cancelled) { setValleyStatus('error'); setValleyError('엔진 서버에 연결할 수 없습니다.'); setSelectedAnchors([]); } });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAnchors, result.analysisId]);
  return <section className="page page--results"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">분석 작업실 <ChevronRight size={14} /> {scan.partNo}</span><h2>엔진별 실제 분석 결과</h2><p>{scan.name} · {result.source.width} × {result.source.height}px</p></div><button className="primary-button" onClick={onService}>보정 시트 만들기 <ArrowRight size={17} /></button></div>
    <div className="result-tabs" role="tablist">{(Object.keys(engineMeta) as Engine[]).map((key, index) => { const item = engineMeta[key]; const failed = Boolean(result.errors[key]); return <button role="tab" aria-selected={engine === key} className={engine === key ? 'active' : ''} onClick={() => setEngine(key)} key={key}><span style={{ color: failed ? '#bd4650' : item.color }}>0{index + 1}</span><div><b>{item.name}</b><small>{failed ? '실행 오류' : item.short}</small></div>{!failed && <Check size={17} />}</button>; })}</div>
    <div className="results-layout"><div className="viewer-card card"><div className="viewer-toolbar"><div><span className={`status ${result.errors[engine] ? 'status--error' : 'status--done'}`}>{result.errors[engine] ? <><X size={13} /> 실행 실패</> : <><Check size={13} /> 실제 분석 완료</>}</span><b>{meta.name}</b></div>{engine === 'deviation' && <button className="tool-button" onClick={() => setVisibleLabelIds(allLabelsVisible ? new Set() : new Set(result.points.map((point) => point.id)))}>{allLabelsVisible ? <EyeOff size={14} /> : <Eye size={14} />} 라벨 전체 {allLabelsVisible ? 'OFF' : 'ON'}</button>}</div><div className="viewer-stage"><Heatmap key={`${scan.id}-${engine}`} imageUrl={image} width={result.source.width} height={result.source.height}>{engine === 'deviation' && <CorrectionPoints coefficient={-1} points={result.points} visibleLabelIds={visibleLabelIds} onLabelToggle={toggleLabel} />}{engine === 'zero' && <>
        <ValleyLineOverlay lines={valleyLines} width={result.source.width} height={result.source.height} />
        <AnchorPicker anchors={zeroAnchors} width={result.source.width} height={result.source.height} selectedIds={selectedAnchors} onToggle={toggleAnchor} />
      </>}</Heatmap></div><div className="viewer-legend"><span><i className="legend-dot" style={{ background: meta.color }} /> 현재 표시: {meta.name}</span><span>{engine === 'deviation' ? '라벨이나 포인트 점을 누르면 개별 표시를 켜고 끌 수 있습니다.' : '표시된 값과 위치는 업로드 이미지의 실제 엔진 결과입니다.'}</span></div></div>
      <aside className="inspection-panel"><div className="score-card card"><span className="score-card__icon" style={{ color: meta.color, background: `${meta.color}12` }}>{engine === 'label' ? <Sparkles /> : engine === 'deviation' ? <Activity /> : <Gauge />}</span><span>핵심 결과</span><strong style={{ color: result.errors[engine] ? '#bd4650' : meta.color }}>{summary.stat}</strong><p>{summary.detail}</p></div><div className="card plain-summary"><h3>쉽게 보는 결과</h3><div className="summary-line"><Check size={16} /><div><b>처리 방식</b><span>{engine === 'label' ? 'label_removal의 인페인팅 결과입니다.' : engine === 'deviation' ? '라벨 제거 이미지에 deviation_extraction의 지시선 끝점과 판독값을 겹쳐 표시합니다.' : 'zero_line_detection의 컬러바 기반 결과입니다.'}</span></div></div>{engineWarnings.length > 0 && <div className="summary-line warning"><MoveRight size={16} /><div><b>확인 필요</b><span>{engineWarnings[0]}</span></div></div>}</div><div className="card mini-table"><div className="card-title"><h3>검출 포인트</h3><span>라벨 {visibleLabelIds.size}/{result.points.length}</span></div>{result.points.map((point) => { const visible = visibleLabelIds.has(point.id); return <div className="point-list-row" key={point.id}><span>{point.id}</span><b className={point.value > 0 ? 'positive' : 'negative'}>{point.value > 0 ? '+' : ''}{point.value.toFixed(3)} mm</b><small>{point.xPx}, {point.yPx}</small><button type="button" className={visible ? 'label-visibility active' : 'label-visibility'} onClick={() => toggleLabel(point.id)} aria-label={`${point.id} 라벨 ${visible ? '숨기기' : '표시하기'}`} title={`라벨 ${visible ? 'OFF' : 'ON'}`}>{visible ? <Eye size={14} /> : <EyeOff size={14} />}</button></div>; })}{!result.points.length && <p className="empty-mini">검출된 포인트가 없습니다.</p>}</div>
      {engine === 'zero' && <div className="card mini-table anchor-panel">
        <div className="card-title"><h3>제로라인 잇기</h3><span>앵커 {zeroAnchors.length}개</span></div>
        <p className="anchor-panel__hint">이미지 위 초록 점(앵커) 2개를 순서대로 클릭하면, 그 사이를 실측 보정시트 수준 정확도로 잇습니다 (검증: 대각선 대비 오차 약 3.68%). 어떤 두 점을 이을지는 사람이 직접 골라야 합니다 — 자동으로는 정답 쌍을 못 찾았습니다.</p>
        {selectedAnchors.length > 0 && valleyStatus !== 'error' && <p className="anchor-panel__status">선택됨: {selectedAnchors.join(', ')} {valleyStatus === 'loading' && '· 잇는 중…'}</p>}
        {valleyStatus === 'error' && valleyError && <p className="anchor-panel__status anchor-panel__status--error">{valleyError}</p>}
        {valleyLines.map((line) => <div className="point-list-row" key={line.id}><span>{line.anchorStartId}↔{line.anchorEndId}</span><b className="positive">{Math.round(line.length_px)}px</b><small>평균|편차| {line.mean_abs_deviation.toFixed(3)}</small><button type="button" className="label-visibility" onClick={() => setValleyLines((current) => current.filter((item) => item.id !== line.id))} aria-label="이 선 지우기" title="이 선 지우기"><X size={14} /></button></div>)}
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

function ServicePreview({ scan, folderAvailable }: { scan: ScanItem; folderAvailable: boolean }) {
  const result = scan.result!; const points = result.points; const [coefficient, setCoefficient] = useState(0.8); const [showPoints, setShowPoints] = useState(true); const [showZero, setShowZero] = useState(true);
  const maxCorrection = useMemo(() => points.length ? Math.max(...points.map((point) => Math.abs(point.value * coefficient))) : 0, [coefficient, points]);
  const baseImage = showZero && result.zeroOverlay ? result.zeroOverlay : result.cleanImage || scan.url;
  return <section className="page page--service"><div className="page-heading page-heading--compact"><div><span className="breadcrumb">실제 서비스 예상 화면 <span className="demo-badge">DEMO</span></span><h2>가상 금형 보정 시트</h2><p>실제 라벨 제거 이미지에 검출된 편차값과 제로라인을 합성합니다.</p></div></div><div className="service-grid"><div className="correction-card card"><div className="viewer-toolbar"><div><span className="status status--done"><Check size={13} /> 실제 결과 합성</span><b>{scan.partNo} · 보정 작업 지시도</b></div><div className="layer-toggles"><button className={showPoints ? 'active orange' : ''} onClick={() => setShowPoints(!showPoints)}><i /> 보정치</button><button className={showZero ? 'active green' : ''} onClick={() => setShowZero(!showZero)} disabled={!result.zeroOverlay}><i /> 제로라인</button></div></div><div className="sheet-stage"><Heatmap key={`${scan.id}-${showZero ? 'zero' : 'clean'}`} imageUrl={baseImage} width={result.source.width} height={result.source.height}>{showPoints && <CorrectionPoints coefficient={coefficient} points={points} />}</Heatmap><div className="sheet-stamp"><span>AJIN INDUSTRIAL</span><b>DIE CORRECTION SHEET</b><small>{scan.partNo} · REV.01</small></div></div><div className="sheet-note"><ShieldCheck size={17} /><span><b>검토용 가상 보정치입니다.</b> 실제 가공 전 담당자 승인과 현장 검증이 필요합니다.</span></div></div><aside className="control-panel"><div className="card coefficient-card"><div className="card-title"><div><h3>보정 계수</h3><p>편차값에 곱할 비율을 조절합니다.</p></div><span>{coefficient.toFixed(2)}×</span></div><input aria-label="보정 계수" type="range" min="0.3" max="1.3" step="0.05" value={coefficient} onChange={(e) => setCoefficient(Number(e.target.value))} /><div className="range-labels"><span>보수적 0.30</span><span>기준 0.80</span><span>적극적 1.30</span></div><div className="formula"><span>보정치</span><b>= 편차 × {coefficient.toFixed(2)} × (−1)</b></div></div><div className="card correction-summary"><h3>실제 엔진 요약</h3><div><span>보정 포인트</span><b>{points.length}개</b></div><div><span>최대 보정량</span><b className="orange">{maxCorrection.toFixed(3)} mm</b></div><div><span>제로라인</span><b className="green">{result.stats.zeroRegions}개 영역</b></div><div><span>처리 품번</span><b>{scan.partNo}</b></div></div></aside></div>{folderAvailable && <Explorer />}</section>;
}

export default function Home() {
  const [view, setView] = useState<View>('workspace'); const [scans, setScans] = useState<ScanItem[]>([]); const [activeId, setActiveId] = useState<string>(); const [collapsed, setCollapsed] = useState(false); const [backendOnline, setBackendOnline] = useState<boolean | null>(null); const [folderAvailable, setFolderAvailable] = useState(false);
  useEffect(() => { fetch(`${API_BASE}/api/health`).then((response) => response.json()).then((data) => { setBackendOnline(Boolean(data.ok)); setFolderAvailable(Boolean(data.folderAvailable)); }).catch(() => setBackendOnline(false)); }, []);
  const resolvedActiveId = activeId || scans[0]?.id;
  const activeScan = scans.find((scan) => scan.id === resolvedActiveId); const completedScan = activeScan?.result ? activeScan : scans.find((scan) => scan.result); const hasResult = Boolean(completedScan?.result);
  const openResults = (id: string) => { setActiveId(id); setView('results'); window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const selectView = (next: View) => { if (next === 'workspace' || hasResult) setView(next); };
  return <main className={`app-shell ${collapsed ? 'app-shell--collapsed' : ''}`}><Sidebar view={view} setView={selectView} collapsed={collapsed} setCollapsed={setCollapsed} hasResult={hasResult} /><div className="app-main"><Header scans={scans} activeId={resolvedActiveId} setActiveId={setActiveId} />{view === 'workspace' && <Workspace scans={scans} setScans={setScans} onOpenResults={openResults} backendOnline={backendOnline} />}{view === 'results' && completedScan?.result && <Results scan={completedScan} onService={() => setView('service')} />}{view === 'service' && completedScan?.result && <ServicePreview scan={completedScan} folderAvailable={folderAvailable} />}</div></main>;
}
