'use client';

/* The supplied AJIN logo is local artwork, reused without a remote image loader. */
/* eslint-disable @next/next/no-img-element */

import { ArrowDown, ArrowUpRight, BarChart3, Box, ChevronRight, FileSpreadsheet, FolderOpen, Layers3 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

export type WorkspaceView = 'overview' | 'workspace' | 'results' | 'service' | 'files' | 'cad';

type NavigationProps = {
  view: WorkspaceView;
  onSelect: (view: WorkspaceView) => void;
  hasResult: boolean;
  scans: { id: string; partNo: string; name: string }[];
  activeId?: string;
  onScanChange: (id: string) => void;
};

const correctionTools = [
  { view: 'workspace' as const, label: '엔진 결과', icon: BarChart3, number: '01' },
  { view: 'service' as const, label: '보정시트', icon: FileSpreadsheet, number: '02' },
  { view: 'cad' as const, label: '3D CAD 뷰어', icon: Box, number: '03' },
];

export function WorkspaceNavigation({ view, onSelect, hasResult, scans, activeId, onScanChange }: NavigationProps) {
  const navigationRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const navigation = navigationRef.current;
    if (!navigation) return;
    let frame = 0;
    const paint = () => {
      frame = 0;
      const range = document.documentElement.scrollHeight - window.innerHeight;
      const progress = range > 0 ? Math.min(1, Math.max(0, window.scrollY / range)) : 0;
      navigation.style.setProperty('--studio-scroll-progress', String(progress));
      navigation.classList.toggle('is-scrolled', window.scrollY > 24);
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(paint); };
    const resize = new ResizeObserver(schedule);
    resize.observe(document.body);
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    paint();
    return () => {
      window.cancelAnimationFrame(frame);
      resize.disconnect();
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
    };
  }, []);
  const correctionActive = view !== 'overview' && view !== 'files';
  return <header className="studio-navigation" ref={navigationRef}>
    <div className="studio-navigation__main">
      <button type="button" className="studio-brand" aria-label="아진산업 작업 선택" onClick={() => onSelect('overview')}>
        <span className="studio-brand__yellow" aria-hidden="true" />
        <img className="studio-brand__wordmark" src="/ajin-industrial-logo.png" alt="" width={957} height={311} />
      </button>
      <nav className="studio-areas" aria-label="작업 영역">
        <button type="button" className={correctionActive ? 'is-active' : ''} aria-current={correctionActive ? 'page' : undefined} onClick={() => onSelect('workspace')}><span>01</span>보정시트 작성</button>
        <button type="button" className={view === 'files' ? 'is-active is-files' : ''} aria-current={view === 'files' ? 'page' : undefined} onClick={() => onSelect('files')}><span>02</span>품번 파일 정리</button>
      </nav>
    </div>
    {correctionActive && <div className="studio-navigation__tools">
      <nav aria-label="보정시트 작성 도구">
        {correctionTools.map(({ view: target, label, icon: Icon, number }) => {
          const active = target === 'workspace' ? view === 'workspace' || view === 'results' : view === target;
          const disabled = target === 'service' && !hasResult;
          return <button key={target} type="button" onClick={() => onSelect(target)} disabled={disabled} aria-current={active ? 'page' : undefined} className={active ? 'is-active' : ''} title={disabled ? '이미지 분석 후 사용 가능' : label}><Icon size={17} /><span>{label}</span><small>{number}</small></button>;
        })}
      </nav>
      <label className="studio-part-select"><span>현재 품번</span><select aria-label="현재 품번" value={activeId || ''} disabled={!scans.length} onChange={(event) => onScanChange(event.target.value)}><option value="">등록된 이미지 없음</option>{scans.map((scan) => <option key={scan.id} value={scan.id}>{scan.partNo} · {scan.name}</option>)}</select></label>
    </div>}
    <div className="studio-scroll-progress" aria-hidden="true" />
  </header>;
}

export function WorkspaceHub({ onSelect, hasResult, scanCount, backendOnline }: {
  onSelect: (view: WorkspaceView) => void;
  hasResult: boolean;
  scanCount: number;
  backendOnline: boolean | null;
}) {
  const hubRef = useRef<HTMLElement>(null);
  const [departing, setDeparting] = useState<WorkspaceView | null>(null);
  useEffect(() => {
    const hub = hubRef.current;
    if (!hub) return;
    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const finePointer = window.matchMedia('(pointer: fine)');
    let frame = 0;
    let pointerX = 0;
    let pointerY = 0;
    const paint = () => {
      frame = 0;
      const offset = motionPreference.matches ? 0 : Math.min(650, Math.max(0, -hub.getBoundingClientRect().top + 82));
      hub.style.setProperty('--studio-depth-back', `${(offset * .21).toFixed(2)}px`);
      hub.style.setProperty('--studio-depth-front', `${(-offset * .09).toFixed(2)}px`);
      hub.style.setProperty('--studio-logo-turn', `${(offset * .07).toFixed(2)}deg`);
      hub.style.setProperty('--studio-pointer-x', `${motionPreference.matches ? 0 : pointerX}px`);
      hub.style.setProperty('--studio-pointer-y', `${motionPreference.matches ? 0 : pointerY}px`);
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(paint); };
    const point = (event: PointerEvent) => {
      if (!finePointer.matches || motionPreference.matches) return;
      const rect = hub.getBoundingClientRect();
      pointerX = Math.max(-1, Math.min(1, ((event.clientX - rect.left) / rect.width - .5) * 2)) * 10;
      pointerY = Math.max(-1, Math.min(1, ((event.clientY - rect.top) / 310 - .5) * 2)) * 7;
      schedule();
    };
    const resetPointer = () => { pointerX = 0; pointerY = 0; schedule(); };
    const reveal = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-revealed');
          reveal.unobserve(entry.target);
        }
      }
    }, { threshold: .08 });
    for (const element of hub.querySelectorAll('[data-studio-reveal]')) {
      // Only defer offscreen content, keeping the initial working choices visible.
      if (element.getBoundingClientRect().top >= window.innerHeight && !motionPreference.matches) {
        element.classList.add('will-reveal');
        reveal.observe(element);
      }
    }
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    hub.addEventListener('pointermove', point, { passive: true });
    hub.addEventListener('pointerleave', resetPointer);
    motionPreference.addEventListener('change', schedule);
    paint();
    return () => {
      window.cancelAnimationFrame(frame);
      reveal.disconnect();
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
      hub.removeEventListener('pointermove', point);
      hub.removeEventListener('pointerleave', resetPointer);
      motionPreference.removeEventListener('change', schedule);
    };
  }, []);

  const scrollToWorkspaces = () => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    hubRef.current?.querySelector('#studio-workspaces')?.scrollIntoView({ behavior: reduce ? 'instant' : 'smooth', block: 'start' });
  };
  const openWorkspace = (target: WorkspaceView) => {
    if (departing) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      onSelect(target);
      return;
    }
    setDeparting(target);
    window.setTimeout(() => onSelect(target), 860);
  };

  return <section className="studio-hub" aria-label="작업 선택" ref={hubRef}>
    <div className="studio-intro">
      <div className="studio-intro__copy">
        <h1>금형 보정<span>WORKSPACE</span></h1>
        <button type="button" className="studio-scroll-link" onClick={scrollToWorkspaces}>작업 선택<ArrowDown size={17} /></button>
      </div>
      <div className="studio-logo-scene" aria-hidden="true">
        <div className="studio-symbol-arrival"><div className="studio-symbol"><img src="/ajin-symbol.svg" alt="" width={618} height={616} /></div></div>
      </div>
    </div>

    <div className="studio-workspaces" id="studio-workspaces">
      <article className="studio-area studio-area--correction" data-studio-reveal>
        <div className="studio-area__heading"><span className="studio-area__number">01 / 보정 작업</span><Layers3 size={25} /></div>
        <h2>보정시트 작성</h2>
        <div className="studio-area__meta"><span>{scanCount > 0 ? `등록 이미지 ${scanCount}개` : '스캔 이미지 · 보정치 · 도면'}</span><span className={`studio-engine-status ${backendOnline ? 'is-online' : ''}`}><i />{backendOnline === null ? '엔진 확인 중' : backendOnline ? '엔진 연결됨' : '엔진 미연결'}</span></div>
        <div className="studio-tool-links">
          {correctionTools.map(({ view, label, icon: Icon, number }) => <button type="button" key={view} disabled={view === 'service' && !hasResult} onClick={() => openWorkspace(view)}><Icon size={22} /><span>{label}<small>{view === 'service' && !hasResult ? '분석 후 사용' : view === 'workspace' ? '이미지 분석' : view === 'cad' ? '도면 열기' : '시트 편집'}</small></span><span className="studio-tool-links__end">{number}<ArrowUpRight size={17} /></span></button>)}
        </div>
      </article>

      <article className="studio-area studio-area--files" data-studio-reveal>
        <div className="studio-area__heading"><span className="studio-area__number">02 / 파일 관리</span><FolderOpen size={25} /></div>
        <h2>품번 파일 정리</h2>
        <div className="studio-area__meta"><span>품번별 분류 · 폴더 구조 · 작업 이력</span></div>
        <div className="studio-file-path" aria-label="파일 분류 항목">
          <span><FolderOpen size={18} />품번</span><ChevronRight size={15} /><span>차종</span><ChevronRight size={15} /><span>자료유형</span>
        </div>
        <button type="button" className="studio-files-entry" onClick={() => openWorkspace('files')}><span>파일 정리 열기</span><ArrowUpRight size={23} /></button>
      </article>
    </div>

    <div className="studio-collaboration" data-studio-reveal>
      <h2>Collaboration in Action</h2>
    </div>
    {departing && <div className="studio-page-transition" aria-hidden="true" />}
  </section>;
}
