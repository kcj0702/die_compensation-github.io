"""CATIA COM 을 통해 .CATPart 를 STEP 으로 변환한다.

[언제 쓰이나]
사용자가 .CATPart 를 mesh 라이브러리 폴더에 그대로 던져 넣으면, 분석 도중
`cad_import.mesh_io.load_any` 가 이 모듈을 불러 STEP 을 만든다. 변환 결과는
같은 폴더의 ``.cache/`` 아래 캐시되어, 같은 부품을 여러 스캔에 걸쳐 다시
변환하지 않는다.

[왜 STEP 인가]
CATIA 는 STL 로도 export 하지만, 그 경우 CATIA 내부의 tessellation 값이
그대로 굳어 온다 — 자동차 판넬 곡면에서 눈에 띄게 성글다. STEP 은 B-Rep
을 유지하므로 우리 쪽 `step_reader.tessellate` 가 필요한 세밀도로 다시
잘게 나눈다.

[성능]
첫 번째 `Dispatch` 는 CATIA 프로세스 기동에 30~60초 걸릴 수 있다. 같은
파이썬 프로세스 안에서 두 번째부터는 즉시. 사용자가 이미 CATIA 를 켜
두었으면 그 세션에 붙는다 — 문서를 열고 닫는 것 외에는 사용자의 작업을
건드리지 않도록 Visible 상태를 원래대로 복구한다.

[보안]
파일은 로컬에서만 처리하고 어디에도 보내지 않는다. 캐시(STEP) 는 mesh
라이브러리 폴더 안에 저장되며, 저장소 최상단 `.gitignore` 로 `data/`
전체와 함께 커밋에서 제외되어 있다.

[한계]
- CATIA 라이선스가 다 쓰이고 있으면 Dispatch 는 되지만 Documents.Open 이
  실패한다. 그 경우 ValueError 로 명확한 원인을 던져 상위가 사용자에게
  안내하게 한다.
- .CATProduct(어셈블리) 도 같은 API 로 열 수 있지만, 여러 파트가 얽혀
  있으면 좌표계·단위가 파트별로 달라 검사 대상이 뭔지 자동으로 못
  가른다. 지금 지원은 단일 파트(.CATPart)만.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path


CATPART_SUFFIXES = {".catpart"}
CATPRODUCT_SUFFIXES = {".catproduct"}
CATIA_SUFFIXES = CATPART_SUFFIXES | CATPRODUCT_SUFFIXES


def is_catia_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in CATIA_SUFFIXES


# CATIA ExportData 는 라이선스에 따라 특정 포맷만 허용된다. 자동차 판넬의
# 완만한 곡면은 STL 로 먼저 굳히면 CATIA의 tessellation 설정에 따라 각져
# 보일 수 있다. STEP(B-Rep)을 우선 내보내 우리 쪽 OCCT가 표시 해상도로 다시
# tessellate하고, STEP 라이선스가 없는 환경에서만 STL로 내려간다.
# CATIA Automation의 STEP ExportData 토큰은 ``step``이 아니라 ``stp``다.
# ``__quality_v3`` 접미사는 잘못된 토큰의 실패 캐시와 예전 STL 캐시를
# 재사용하지 않게 한다.
_EXPORT_FORMATS: tuple[tuple[str, str], ...] = (
    ("stp", "__quality_v3.step"),
    ("stl", "__quality_v3.stl"),
)

# CATIA Automation은 한 프로세스에 여러 ExportData 호출이 동시에 들어오면
# RPC_S_CALL_FAILED / RPC_E_CALL_REJECTED를 내기 쉽다. 분석과 CAD 뷰어가 같은
# 순간에 변환을 요청해도 CATIA에는 하나씩만 전달한다.
_CATIA_EXPORT_LOCK = threading.Lock()
_TRANSIENT_COM_MARKERS = (
    "-2147023170",  # 0x800706BE RPC_S_CALL_FAILED
    "-2147418111",  # 0x80010001 RPC_E_CALL_REJECTED
    "-2147417846",  # 0x8001010A RPC_E_SERVERCALL_RETRYLATER
    "rpc_",
    "remote procedure",
    "원격 프로시저",
    "-2147467259",  # 0x80004005 E_FAIL from CATIA ExportData
    "the method exportdata failed",
)


def _is_transient_com_failure(reason: object) -> bool:
    folded = str(reason).lower()
    return any(marker in folded for marker in _TRANSIENT_COM_MARKERS)


def _pending_export_path(target: Path) -> Path:
    """Return a non-existing sibling path while preserving the CAD suffix."""
    return target.with_name(f"{target.stem}__pending_{uuid.uuid4().hex}{target.suffix}")


def _convert_to_mesh_once(
    source: str | Path,
    cache_dir: str | Path,
    *,
    step_only: bool = False,
    force_new_instance: bool = False,
) -> Path:
    """.CATPart 를 열어 STEP 우선, STL 순으로 변환해 성공한 경로를 준다.

    STEP export 모듈이 없을 때는 해당 호출만 실패하며, 이어서 호환성이 높은
    STL을 시도한다. STEP이 성공하면 곡면을 B-Rep 상태로 보존하므로 뷰어용
    tessellation 품질을 우리 쪽에서 통제할 수 있다.

    Args:
        source:    변환할 .CATPart / .CATProduct.
        cache_dir: 결과 파일을 둘 폴더 (없으면 만든다).

    Returns:
        변환된 파일 경로 (확장자는 성공한 포맷). 이미 캐시가 최신이면
        그것을 그대로 반환.

    Raises:
        FileNotFoundError: 원본이 없을 때.
        ValueError: 확장자·pywin32·CATIA 실행·저장 어느 단계라도 실패했을 때.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() not in CATIA_SUFFIXES:
        raise ValueError(f"CATIA 파일이 아닙니다: {source_path.name}")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    export_formats = _EXPORT_FORMATS[:1] if step_only else _EXPORT_FORMATS
    failure_suffix = "__quality_v3_step.failed" if step_only else "__quality_v3.failed"
    failure_marker = cache_root / f"{source_path.stem}{failure_suffix}"

    # 이미 어떤 포맷으로든 캐시가 있고 원본보다 최신이면 그대로 쓴다.
    for _fmt, ext in export_formats:
        candidate = cache_root / f"{source_path.stem}{ext}"
        if candidate.is_file() and candidate.stat().st_mtime >= source_path.stat().st_mtime:
            return candidate

    if failure_marker.is_file() and failure_marker.stat().st_mtime >= source_path.stat().st_mtime:
        reason = failure_marker.read_text(encoding="utf-8", errors="replace").strip()
        if _is_transient_com_failure(reason):
            failure_marker.unlink(missing_ok=True)
        else:
            raise ValueError(f"이 CATPart의 이전 변환 실패를 재사용합니다: {reason}")

    try:
        import pythoncom  # noqa: WPS433
        import win32com.client
    except ImportError as exc:
        raise ValueError(
            "CATPart 변환에는 pywin32 (win32com) 가 필요합니다. "
            "이 PC 는 CATIA 가 설치돼 있으니 `pip install pywin32` 로 설치하세요. "
            f"원인: {exc}"
        ) from exc

    pythoncom.CoInitialize()
    try:
        try:
            dispatch = (
                win32com.client.DispatchEx
                if force_new_instance else win32com.client.Dispatch
            )
            catia = dispatch("CATIA.Application")
        except pythoncom.com_error as exc:
            raise ValueError(
                "CATIA 를 실행할 수 없습니다. CATIA 설치·라이선스를 확인하세요. "
                f"원인: {exc}"
            ) from exc

        previous_visible = None
        try:
            previous_visible = catia.Visible
            catia.Visible = False
        except Exception:
            pass

        doc = None
        failures: list[str] = []
        successful_path: Path | None = None
        try:
            try:
                doc = catia.Documents.Open(str(source_path.resolve()))
            except pythoncom.com_error as exc:
                raise ValueError(
                    f"CATIA 가 {source_path.name} 을 열지 못했습니다. "
                    "라이선스가 부족하거나 파일이 손상됐을 수 있습니다. "
                    f"원인: {exc}"
                ) from exc

            for fmt, ext in export_formats:
                target = cache_root / f"{source_path.stem}{ext}"
                pending = _pending_export_path(target)
                try:
                    # ExportData may reject an existing destination instead of
                    # replacing it.  Export beside the cache and atomically
                    # replace the old successful result only after validation.
                    doc.ExportData(str(pending.resolve()), fmt)
                except pythoncom.com_error as exc:
                    failures.append(f"{fmt}({exc})")
                    pending.unlink(missing_ok=True)
                    continue
                if pending.is_file() and pending.stat().st_size > 0:
                    pending.replace(target)
                    successful_path = target
                    break
                failures.append(f"{fmt}(파일 미생성)")
                pending.unlink(missing_ok=True)
        finally:
            if doc is not None:
                try:
                    doc.Close()
                except Exception:
                    pass
            if previous_visible is not None:
                try:
                    catia.Visible = previous_visible
                except Exception:
                    pass
    finally:
        pythoncom.CoUninitialize()

    if successful_path is None:
        reason = "CATIA export 가 모든 포맷에서 실패했습니다: " + ", ".join(failures)
        if _is_transient_com_failure(reason):
            failure_marker.unlink(missing_ok=True)
        else:
            failure_marker.write_text(reason, encoding="utf-8")
        raise ValueError(reason)
    failure_marker.unlink(missing_ok=True)
    return successful_path


def convert_to_mesh(
    source: str | Path,
    cache_dir: str | Path,
    *,
    step_only: bool = False,
) -> Path:
    """CATIA 변환을 직렬 실행하고 일시적인 RPC 단절은 한 번 재시도한다."""
    with _CATIA_EXPORT_LOCK:
        try:
            return _convert_to_mesh_once(
                source, cache_dir, step_only=step_only,
            )
        except ValueError as exc:
            if not _is_transient_com_failure(exc):
                raise
            time.sleep(0.5)
            return _convert_to_mesh_once(
                source, cache_dir, step_only=step_only, force_new_instance=True,
            )


def convert_to_step(source: str | Path, cache_dir: str | Path) -> Path:
    """CATIA 원본을 STEP으로만 변환한다.

    CAD 뷰어는 STEP의 B-Rep에서 홀과 기준 평면을 읽어야 하므로 STL로
    조용히 내려가지 않는다. 일반 형상 렌더링은 기존 ``convert_to_mesh``의
    STEP→STL 호환 경로를 계속 사용할 수 있다.
    """
    return convert_to_mesh(source, cache_dir, step_only=True)


__all__ = [
    "CATIA_SUFFIXES", "CATPART_SUFFIXES", "CATPRODUCT_SUFFIXES",
    "convert_to_mesh", "convert_to_step", "is_catia_file",
]
