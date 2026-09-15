"""CATIA 관리면의 BOUNDARY 곡선을 로컬 IGES로 꺼내 3D 점열로 만든다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable
import uuid

import numpy as np

from .management_surface import ManagementGroup, ManagementSurface, discover_management_groups


_MANIFEST_VERSION = 1


def _safe_name(value: str) -> str:
    import re
    cleaned = re.sub(r"[^0-9A-Za-z가-힣.+-]+", "_", value).strip("_")
    return cleaned or "management"


def management_source_key(catpart: str | Path) -> str:
    """CATPart **내용**의 해시. 같은 파일을 다시 등록해도(mtime만 바뀜) 캐시를 그대로 쓴다."""
    digest = hashlib.sha1(usedforsecurity=False)
    with open(catpart, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:10]


def export_boundary_iges(
    catia: Any,
    source_document: Any,
    group: ManagementGroup,
    output: str | Path,
) -> Path:
    """관리 그룹의 BOUNDARY 안 곡선을 링크 없는 IGES로 내보낸다."""
    curves: list[Any] = []
    for boundary in group.boundary_objects:
        shapes = getattr(boundary, "HybridShapes", None)
        for index in range(1, int(getattr(shapes, "Count", 0)) + 1):
            curves.append(shapes.Item(index))
    if not curves:
        raise ValueError(f"{group.name}: BOUNDARY 곡선이 없습니다.")

    source_selection = source_document.Selection
    source_selection.Clear()
    for curve in curves:
        source_selection.Add(curve)
    source_selection.Copy()

    target_document = catia.Documents.Add("Part")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ExportData는 이미 있는 파일 위에 덮어쓰지 못하고 "The method ExportData
    # failed"를 낸다(catia_convert 와 같은 증상). 새 이름으로 내보낸 뒤 바꿔 끼운다.
    pending = path.with_name(f"{path.stem}__pending_{uuid.uuid4().hex}{path.suffix}")
    try:
        target_part = target_document.Part
        target_group = target_part.HybridBodies.Add()
        target_group.Name = "BOUNDARY"
        target_part.InWorkObject = target_group
        target_selection = target_document.Selection
        target_selection.Clear()
        target_selection.Add(target_group)
        target_selection.PasteSpecial("CATPrtResultWithOutLink")
        target_part.Update()
        try:
            target_document.ExportData(str(pending.resolve()), "igs")
        except Exception as exc:
            raise ValueError(
                f"{group.name}: CATIA IGES 내보내기에 실패했습니다. CATIA에 열린 대화상자가 있거나 "
                f"IGES 라이선스가 없을 수 있습니다. 원인: {exc}"
            ) from exc
        if not pending.is_file() or pending.stat().st_size == 0:
            raise ValueError(f"{group.name}: CATIA가 BOUNDARY IGES를 만들지 못했습니다.")
        pending.replace(path)
        return path
    finally:
        pending.unlink(missing_ok=True)
        target_document.Close()
        source_selection.Clear()


def _join_connected_edges(edges: list[np.ndarray], tolerance: float = 0.5) -> list[np.ndarray]:
    """IGES에서 잘게 분리된 edge를 끝점 기준의 원래 곡선으로 재조립한다."""
    remaining = [np.asarray(edge, dtype=float) for edge in edges if len(edge) >= 2]
    chains: list[np.ndarray] = []
    while remaining:
        chain = remaining.pop(0)
        while remaining:
            choices: list[tuple[float, int, str]] = []
            for index, edge in enumerate(remaining):
                choices.extend([
                    (float(np.linalg.norm(chain[-1] - edge[0])), index, "append"),
                    (float(np.linalg.norm(chain[-1] - edge[-1])), index, "append-reverse"),
                    (float(np.linalg.norm(chain[0] - edge[-1])), index, "prepend"),
                    (float(np.linalg.norm(chain[0] - edge[0])), index, "prepend-reverse"),
                ])
            distance, index, mode = min(choices, key=lambda item: item[0])
            if distance > tolerance:
                break
            edge = remaining.pop(index)
            if mode == "append":
                chain = np.vstack([chain, edge[1:]])
            elif mode == "append-reverse":
                chain = np.vstack([chain, edge[-2::-1]])
            elif mode == "prepend":
                chain = np.vstack([edge[:-1], chain])
            else:
                chain = np.vstack([edge[:0:-1], chain])
        chains.append(chain)
    return chains


def sample_iges_edges(path: str | Path, points_per_edge: int = 16) -> list[np.ndarray]:
    """IGES edge를 읽고 서로 이어진 것끼리 원래 BOUNDARY로 합친다."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    reader = IGESControl_Reader()
    status = reader.ReadFile(str(Path(path).resolve()))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"BOUNDARY IGES를 읽지 못했습니다: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_EDGE)
    polylines: list[np.ndarray] = []
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        first, last = float(curve.FirstParameter()), float(curve.LastParameter())
        if np.isfinite(first) and np.isfinite(last) and last > first:
            params = np.linspace(first, last, max(3, int(points_per_edge)))
            points = []
            for parameter in params:
                point = curve.Value(float(parameter))
                points.append([point.X(), point.Y(), point.Z()])
            polylines.append(np.asarray(points, dtype=float))
        explorer.Next()
    return _join_connected_edges(polylines)


def _manifest_path(cache_root: Path, source_key: str) -> Path:
    return cache_root / f"management_{source_key}_manifest.json"


def _iges_path(cache_root: Path, source_key: str, index: int) -> Path:
    # CATIA V5 ExportData가 한글·따옴표·+/-가 섞인 대상 경로에서
    # 응답 없이 멈추는 환경이 있다. 캐시는 항상 ASCII 이름을 쓴다.
    return cache_root / f"management_{source_key}_boundary_{index:02d}.igs"


def _load_cached_groups(cache_root: Path, source_key: str) -> list[tuple[ManagementGroup, Path]] | None:
    """CATIA 없이 트리 정보와 IGES 경로를 복원한다. 하나라도 빠져 있으면 None."""
    manifest = _manifest_path(cache_root, source_key)
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("version") != _MANIFEST_VERSION:
        return None
    groups: list[tuple[ManagementGroup, Path]] = []
    for entry in payload.get("groups", []):
        iges = cache_root / str(entry.get("iges", ""))
        if not iges.is_file() or iges.stat().st_size == 0:
            return None
        groups.append((ManagementGroup(
            name=str(entry["name"]),
            target_mm=float(entry["targetMm"]),
            surfaces=[ManagementSurface(
                surface_id=str(surface.get("surfaceId", "")),
                spec_name=str(surface.get("specSurface", "")),
                applied_name=str(surface.get("appliedSurface", "")),
            ) for surface in entry.get("surfaces", [])],
            boundary_names=[str(name) for name in entry.get("boundaries", [])],
            warnings=[str(warning) for warning in entry.get("warnings", [])],
        ), iges))
    return groups


def _catia_pids() -> set[int]:
    try:
        import psutil
    except ImportError:
        return set()
    return {proc.pid for proc in psutil.process_iter(["name"]) if str(proc.info.get("name") or "").lower() == "cnext.exe"}


def _connect_catia(report: Callable[[str], None]) -> tuple[Any, set[int], Any]:
    """이미 떠 있는 CATIA가 있으면 붙고, 없을 때만 새 인스턴스를 띄운다.

    새로 띄우면 기동·종료에만 수십 초가 든다. 붙은 경우 종료하지 않고
    Visible 상태만 되돌린다. 반환하는 pid 집합은 이번에 새로 생긴 CNEXT
    프로세스라 종료 때 남아 있으면 정리한다.
    """
    import win32com.client

    try:
        catia = win32com.client.GetActiveObject("CATIA.Application")
        report("catia-attach")
        return catia, set(), getattr(catia, "Visible", None)
    except Exception:
        report("catia-launch")
        before = _catia_pids()
        catia = win32com.client.DispatchEx("CATIA.Application")
        return catia, _catia_pids() - before, None


def _quit_launched_catia(catia: Any, launched_pids: set[int], report: Callable[[str], None]) -> None:
    """보이지 않게 띄운 CATIA는 Quit()을 무시하고 남는 경우가 있다(실측: 문서 0개인
    채 355MB 점유). 잠시 기다린 뒤 우리가 만든 프로세스만 강제 종료한다."""
    try:
        catia.Quit()
    except Exception:
        pass
    if not launched_pids:
        return
    try:
        import psutil
    except ImportError:
        return
    deadline = time.time() + 5.0
    while time.time() < deadline and any(psutil.pid_exists(pid) for pid in launched_pids):
        time.sleep(0.25)
    for pid in launched_pids:
        if psutil.pid_exists(pid):
            try:
                psutil.Process(pid).kill()
                report(f"catia-kill pid={pid}")
            except psutil.Error:
                pass


def extract_management_boundaries(
    catpart: str | Path,
    *,
    cache_dir: str | Path | None = None,
    points_per_edge: int = 16,
    progress: Callable[[str], None] | None = None,
) -> list[tuple[ManagementGroup, list[np.ndarray]]]:
    """CATPart 트리와 각 관리 그룹의 BOUNDARY 3D 점열을 함께 반환한다.

    캐시(manifest + IGES)가 있으면 CATIA를 아예 띄우지 않는다. 캐시 키는
    파일 내용 해시라 같은 파일을 다시 등록해도 다시 내보내지 않는다.
    """
    source = Path(catpart)
    if not source.is_file():
        raise FileNotFoundError(source)
    cache_root = Path(cache_dir) if cache_dir else source.parent / ".management_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda message: print(f"[management] {message}", file=sys.stderr, flush=True))
    started = time.time()
    source_key = management_source_key(source)

    cached = _load_cached_groups(cache_root, source_key)
    if cached is not None:
        report(f"cache-hit key={source_key} groups={len(cached)}")
        extracted: list[tuple[ManagementGroup, list[np.ndarray]]] = []
        for group, iges in cached:
            polylines = sample_iges_edges(iges, points_per_edge=points_per_edge)
            if polylines:
                extracted.append((group, polylines))
        report(f"iges-read {time.time() - started:.1f}s")
        return extracted

    import pythoncom

    pythoncom.CoInitialize()
    report("catia-start")
    catia, launched_pids, previous_visible = _connect_catia(report)
    launched = previous_visible is None
    try:
        catia.Visible = False
    except Exception:
        pass
    try:
        # 보이지 않는 CATIA에서 파일 대화상자가 뜨면 ExportData가 그대로 실패한다.
        catia.DisplayFileAlerts = False
    except Exception:
        pass
    report(f"catia-open {time.time() - started:.1f}s")
    document = catia.Documents.Open(str(source.resolve()))
    try:
        groups = discover_management_groups(document.Part)
        report(f"catia-groups:{len(groups)} {time.time() - started:.1f}s")
        extracted = []
        manifest_groups: list[dict[str, Any]] = []
        for index, group in enumerate(groups):
            if not group.boundary_objects:
                continue
            iges = _iges_path(cache_root, source_key, index)
            report(f"catia-export:{group.name}")
            export_boundary_iges(catia, document, group, iges)
            report(f"iges-read:{group.name} {time.time() - started:.1f}s")
            polylines = sample_iges_edges(iges, points_per_edge=points_per_edge)
            if polylines:
                extracted.append((group, polylines))
                manifest_groups.append({**group.to_dict(), "iges": iges.name})
        _manifest_path(cache_root, source_key).write_text(
            json.dumps({"version": _MANIFEST_VERSION, "source": str(source.resolve()), "groups": manifest_groups},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        report(f"done {time.time() - started:.1f}s")
        return extracted
    finally:
        try:
            document.Close()
        finally:
            if launched:
                _quit_launched_catia(catia, launched_pids, report)
            else:
                try:
                    catia.Visible = previous_visible
                except Exception:
                    pass
            pythoncom.CoUninitialize()


__all__ = [
    "export_boundary_iges", "extract_management_boundaries", "management_source_key", "sample_iges_edges",
]
