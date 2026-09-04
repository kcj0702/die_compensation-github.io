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

from pathlib import Path


CATPART_SUFFIXES = {".catpart"}
CATPRODUCT_SUFFIXES = {".catproduct"}
CATIA_SUFFIXES = CATPART_SUFFIXES | CATPRODUCT_SUFFIXES


def is_catia_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in CATIA_SUFFIXES


# CATIA ExportData 는 라이선스에 따라 특정 포맷만 허용된다. 회사마다 STEP 이
# 없고 STL 만 되는 경우가 흔해서, 여러 포맷을 순서대로 시도하고 처음 성공한
# 것으로 저장한다. STL 을 앞에 두면 우리 파이프라인(trimesh) 이 아무 의존성
# 추가 없이 곧바로 로드할 수 있어 이득이다. IGES 는 우리 load_any 가 지금
# 파싱하지 못하므로 목록에서 뺐다.
_EXPORT_FORMATS: tuple[tuple[str, str], ...] = (
    ("stl",  ".stl"),
    ("step", ".step"),
)


def convert_to_mesh(source: str | Path, cache_dir: str | Path) -> Path:
    """.CATPart 를 열어 STL/STEP/IGES 순으로 변환 시도, 성공한 파일 경로를 준다.

    실측: 회사 CATIA 라이선스에 STEP export 모듈이 없을 때 그 호출만 실패한다.
    포맷마다 별도 라이선스라 여러 개를 순서대로 시도하는 게 안전하다. STL 을
    먼저 두는 이유는 (1) 라이선스가 가장 널리 포함되어 있고 (2) 우리 파이프
    라인이 trimesh 로 곧바로 열 수 있어 OCCT 를 우회할 수 있어서다.

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

    # 이미 어떤 포맷으로든 캐시가 있고 원본보다 최신이면 그대로 쓴다.
    for _fmt, ext in _EXPORT_FORMATS:
        candidate = cache_root / f"{source_path.stem}{ext}"
        if candidate.is_file() and candidate.stat().st_mtime >= source_path.stat().st_mtime:
            return candidate

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
            catia = win32com.client.Dispatch("CATIA.Application")
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

            for fmt, ext in _EXPORT_FORMATS:
                target = cache_root / f"{source_path.stem}{ext}"
                try:
                    doc.ExportData(str(target.resolve()), fmt)
                except pythoncom.com_error as exc:
                    failures.append(f"{fmt}({exc})")
                    continue
                if target.is_file() and target.stat().st_size > 0:
                    successful_path = target
                    break
                failures.append(f"{fmt}(파일 미생성)")
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
        raise ValueError(
            "CATIA export 가 모든 포맷에서 실패했습니다: " + ", ".join(failures)
        )
    return successful_path


# 하위 호환: 기존에 convert_to_step 를 import 하던 곳이 있다면 그대로 동작.
convert_to_step = convert_to_mesh


__all__ = [
    "CATIA_SUFFIXES", "CATPART_SUFFIXES", "CATPRODUCT_SUFFIXES",
    "convert_to_mesh", "convert_to_step", "is_catia_file",
]
