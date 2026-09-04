"""PolyWorks 워크스페이스에서 3D 스캔 점군을 꺼낸다.

[왜 필요한가]
지금까지 우리가 받은 "스캔" 은 검사 리포트를 캡처한 PNG 한 장이었다
(JD_64XX2-DR000 3D 스캔.png, 1306x680px). 그림에는 색 편차맵과 컬러바,
지시선 콜아웃만 있고 **3D 좌표가 없다.** 그래서 2D 그림을 CAD 위에
실루엣 모양만 보고 얹었고, 그 얹힘이 곧 정합률이었다(64XX2 99.7% ·
67XX6 99.0% · 71XX2 90.0%). 스캔 위 점의 3D 위치는 전부 그 얹기에서
나온 추정치였다.

진짜 스캔 원본(점군)이 있으면 얹을 필요가 없다 — CAD 와 점군을 직접
맞추면 되고(ICP), 편차도 픽셀 색이 아니라 실제 거리로 잰다.

[PolyWorks 파일을 어떻게 읽나]
`.pwk` 는 XML 색인이고, 실제 데이터는 `*_Files/wm-data/vault/` 아래에
SHA-1 이름으로 흩어져 있다(git 과 같은 방식이다). 형식은 공개돼 있지
않지만, 실측해 보니 점군 파일은 아주 단순했다 —

    바이트 0..7    머리(magic). 첫 4바이트가 ca af a0 00
    바이트 8..     float64 리틀엔디언 x, y, z 가 끝까지 되풀이

실측 `3D스캔 AX과제`(PolyWorks 23.7.0) 의 503.7MB 파일이 이 규칙으로
22,005,698 점이 나왔고, 표본 30만 점 중 **99.7%** 가 유한하고 |값| <
5000mm 인 좌표였다. 크기는 1129 x 1531 x 241mm 로 실제 패널 크기다.

[한계 — 읽어 두면 좋다]
· 삼각망(면)은 안 꺼낸다. 점만 나온다. 면이 필요하면 PolyWorks 에서
  STL 로 내보내는 편이 낫다.
· 좌표계가 CAD 와 같은지는 이 파일만으로 알 수 없다. 맞춰 보고 판단해야
  한다(bounds 로 어느 CAD 짝인지 좁힐 수 있다).
· 규칙을 실측으로 알아낸 것이라 PolyWorks 판이 바뀌면 안 맞을 수 있다.
  그래서 읽을 때마다 좌표처럼 보이는 비율을 재고, 낮으면 거른다.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 점군 파일의 머리. 실측 파일 여덟 개에서 모두 같았다.
MAGIC = bytes([0xCA, 0xAF, 0xA0, 0x00])
HEADER_BYTES = 8
POINT_BYTES = 24                 # float64 세 개

# 좌표라고 인정하는 범위(mm). 승용차 부품과 스캐너 작업 반경을 넉넉히 덮는다.
SANE_MM = 5000.0
# 표본 중 이 비율은 좌표여야 점군으로 본다.
MIN_SANE = 0.90


@dataclass
class ScanCloud:
    """꺼낸 점군 하나."""

    path: Path
    points: np.ndarray           # (n, 3) float64, 부품 좌표(mm)
    sane_ratio: float            # 표본 중 좌표처럼 보인 비율

    @property
    def bounds(self) -> tuple:
        return (self.points.min(axis=0), self.points.max(axis=0))

    @property
    def size(self) -> np.ndarray:
        low, high = self.bounds
        return high - low

    def to_ply(self, target: Path) -> Path:
        """점군을 PLY 로 쓴다 — 뷰어와 다른 도구가 바로 읽는다."""
        target = Path(target)
        count = len(self.points)
        head = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        with target.open("wb") as out:
            out.write(head.encode("ascii"))
            # float32 로 줄여 쓴다. 0.001mm 도 못 담을 이유가 없고(패널이
            # 2m 안쪽이라 float32 의 유효자리로 마이크로미터까지 남는다)
            # 파일이 절반이 된다.
            out.write(self.points.astype("<f4").tobytes())
        return target


def looks_like_cloud(path: Path) -> bool:
    """머리만 보고 점군 파일인지 가린다(파일을 다 읽지 않는다)."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < HEADER_BYTES + POINT_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == MAGIC


def read_cloud(path: Path, sample: int = 200_000) -> ScanCloud | None:
    """점군 하나를 읽는다. 규칙에 안 맞으면 None 을 준다.

    Args:
        path: vault 안의 파일.
        sample: 좌표인지 재 볼 표본 크기. 파일이 500MB 라 전부 재면 느리다.
    """
    path = Path(path)
    size = path.stat().st_size
    count = (size - HEADER_BYTES) // POINT_BYTES
    if count <= 0:
        return None

    seen = np.memmap(path, dtype="<f8", mode="r",
                     offset=HEADER_BYTES, shape=(count, 3))
    step = max(1, count // max(1, sample))
    probe = np.asarray(seen[::step])
    sane = (np.isfinite(probe).all(axis=1)
            & (np.abs(probe) < SANE_MM).all(axis=1))
    ratio = float(sane.mean()) if len(sane) else 0.0
    if ratio < MIN_SANE:
        return None

    # 좌표는 파일 앞쪽에 이어져 있고 꼬리에 다른 데이터가 붙는다.
    end = _coordinate_end(seen, count)
    points = np.asarray(seen[:end])
    # 블록 끝에는 (0,0,0) 을 채워 둔다. 원점에 점이 몰리면 정합이 끌려간다.
    points = points[np.abs(points).sum(axis=1) > 0]
    return ScanCloud(path=path, points=points, sane_ratio=ratio)


def _coordinate_end(seen: np.ndarray, count: int,
                    chunk: int = 200_000) -> int:
    """좌표가 이어지는 구간이 어디서 끝나는지 찾는다.

    [왜 걸러내기가 아니라 자르기인가]
    처음에는 점마다 |값| < 5000mm 인지 보고 걸렀다. 그런데 꼬리의
    데이터는 좌표가 아니라 **자리(phase)가 어긋난 채로 읽힌다** — 세
    칸씩 끊어 읽으니 (y, z, x) 처럼 섞인 값이 나오고, 그중에는 범위
    안에 드는 것도 있어 그대로 통과한다. 실측 파일에서 그렇게 새어
    들어온 값 때문에 Z 범위가 240mm 에서 960mm 로 부풀었다.

    좌표는 앞에서부터 이어지므로, 처음으로 깨지는 자리를 찾아 그
    앞까지만 쓴다. 덩어리로 훑어 대강의 자리를 잡고 그 안에서 정확한
    행을 찾는다 — 2200만 행을 한 번에 재면 메모리가 튄다.
    """
    def broken(block: np.ndarray) -> np.ndarray:
        return ~(np.isfinite(block).all(axis=1)
                 & (np.abs(block) < SANE_MM).all(axis=1))

    for start in range(0, count, chunk):
        block = np.asarray(seen[start:start + chunk])
        hit = np.where(broken(block))[0]
        if len(hit):
            return start + int(hit[0])
    return count


def clouds_in(workspace: Path, min_points: int = 10_000) -> list:
    """워크스페이스 하나에서 점군을 모두 찾는다.

    Args:
        workspace: `.pwk` 파일이나 그 옆의 `*_Files` 폴더, 또는 둘을 담은 폴더.

    Returns:
        점이 많은 것부터 ScanCloud 목록.
    """
    root = Path(workspace)
    if root.suffix.lower() == ".pwk":
        root = root.with_name(root.stem + "_Files")
    vault = root / "wm-data" / "vault"
    if not vault.is_dir():
        found = list(root.glob("*_Files/wm-data/vault"))
        if not found:
            raise FileNotFoundError(f"vault 를 찾지 못했습니다: {root}")
        vault = found[0]

    out: list = []
    for item in sorted(vault.rglob("*"), key=lambda f: -f.stat().st_size
                       if f.is_file() else 0):
        if not looks_like_cloud(item):
            continue
        cloud = read_cloud(item)
        if cloud is not None and len(cloud.points) >= min_points:
            out.append(cloud)
    return out


def decimate(points: np.ndarray, target: int) -> np.ndarray:
    """점을 고르게 솎는다.

    2200만 점을 그대로 브라우저에 보내면 열리지 않는다. 무작위로 고르면
    성긴 데가 생기므로 일정 간격으로 집는다 — 스캔은 훑은 순서대로
    쌓이므로 이것만으로도 고르게 퍼진다.
    """
    if target <= 0 or len(points) <= target:
        return points
    step = len(points) // target
    return points[::step][:target]


# ── 검사 포인트 ─────────────────────────────────────────────
#
# [무엇을 꺼내는가]
# 우리가 PNG 에서 힘들게 읽던 그 콜아웃이다. 지금 파이프라인은 라벨을
# 지우고, 지시선 끝점을 찾고, Qwen 으로 숫자를 읽는다(실측 79개). 그
# 숫자와 자리가 워크스페이스 안에 **그대로** 들어 있다 — 읽을 것도,
# 컬러바로 색을 되돌릴 것도 없다.
#
# 구조는 이렇다.
#   <O clsid="CmpPtSurf">   검사 포인트 하나
#     <P id="Name">surf pt 6</P>
#     <P id="PiercePt">      부품 좌표(mm) — hex float64 셋
#     <P id="EffectiveNormal">  그 자리의 법선
#   <O clsid="CmpPtDim">    그 포인트에 딸린 값들
#     <P id="DimName">... Surface Distance
#     <P id="Deviation">     편차(mm) — hex float64
#
# 값은 IEEE754 를 **빅엔디언 hex 글자**로 적어 둔다(점군의 리틀엔디언
# 바이트와 다르다). CmpPtDim 은 바로 앞의 CmpPtSurf 에 딸린다.

_OBJECT = None      # 정규식은 처음 쓸 때 만든다(임포트를 가볍게 둔다)


def _hex_float(text: str) -> float:
    """`C00650A7BB663A3D` 같은 빅엔디언 hex 를 실수로 되돌린다."""
    return struct.unpack(">d", bytes.fromhex(text.strip()))[0]


def _hex_triple(block: str) -> list:
    return [_hex_float(x) for x in re.findall(r"<E>([0-9A-Fa-f]{16})</E>", block)]


@dataclass
class InspectionPoint:
    """검사 포인트 하나 — 자리와 편차."""

    name: str
    position: tuple                # (x, y, z) 부품 좌표 mm
    normal: tuple | None           # 그 자리의 법선
    deviation: float | None        # mm. + 는 살이 더 있음, - 는 모자람

    def as_dict(self) -> dict:
        return {"name": self.name, "position": list(self.position),
                "normal": list(self.normal) if self.normal else None,
                "deviation": self.deviation}


def inspection_points(workspace: Path) -> list:
    """워크스페이스에서 검사 포인트를 모두 꺼낸다.

    Returns:
        InspectionPoint 목록. 실측 "3D스캔 AX과제" 에서 79개가 나온다 —
        우리 PNG 파이프라인이 읽어 내던 것과 같은 수다.
    """
    global _OBJECT
    if _OBJECT is None:
        _OBJECT = re.compile(
            r'<O\b[^>]*clsid="(CmpPtSurf|CmpPtDim)"[^>]*>(.*?)</O>', re.S)

    text = _metadata_text(workspace)
    out: list = []
    for kind, body in _OBJECT.findall(text):
        if kind == "CmpPtSurf":
            pierce = re.search(r'<P id="PiercePt">(.*?)</P>', body, re.S)
            if pierce is None:
                continue
            spot = _hex_triple(pierce.group(1))
            if len(spot) != 3:
                continue
            normal = re.search(r'<P id="EffectiveNormal">(.*?)</P>', body, re.S)
            way = _hex_triple(normal.group(1)) if normal else []
            name = re.search(r'<P id="Name">([^<]*)</P>', body)
            out.append(InspectionPoint(
                name=(name.group(1) if name else f"pt {len(out) + 1}"),
                position=tuple(spot),
                normal=tuple(way) if len(way) == 3 else None,
                deviation=None))
        elif out and out[-1].deviation is None:
            # 바로 앞 포인트에 딸린 값이다. 표면 거리만 쓴다 — 같은
            # 포인트에 각도나 반경 같은 다른 값이 함께 붙기도 한다.
            if "Surface Distance" not in body:
                continue
            found = re.search(r'<P id="Deviation">([0-9A-Fa-f]{16})</P>', body)
            if found:
                out[-1].deviation = _hex_float(found.group(1))
    return out


def _metadata_text(workspace: Path) -> str:
    """워크스페이스의 XML 메타데이터를 찾아 읽는다.

    vault 안에 XML 로 된 파일이 하나 있다(실측 4.2MB). 이름이 SHA-1 이라
    머리를 보고 가린다.
    """
    root = Path(workspace)
    if root.suffix.lower() == ".pwk":
        root = root.with_name(root.stem + "_Files")
    vault = root / "wm-data" / "vault"
    if not vault.is_dir():
        found = list(root.glob("*_Files/wm-data/vault"))
        if not found:
            raise FileNotFoundError(f"vault 를 찾지 못했습니다: {root}")
        vault = found[0]

    best = ""
    for item in vault.rglob("*"):
        if not item.is_file() or item.stat().st_size < 1024:
            continue
        with item.open("rb") as handle:
            head = handle.read(64)
        if b"<?xml" not in head:
            continue
        text = item.read_text(encoding="utf-8", errors="replace")
        # 검사 포인트가 든 쪽을 고른다. 워크스페이스 XML 이 여러 개다.
        if "CmpPtSurf" in text and len(text) > len(best):
            best = text
    if not best:
        raise FileNotFoundError("검사 포인트가 든 메타데이터를 찾지 못했습니다")
    return best


__all__ = ["InspectionPoint", "ScanCloud", "clouds_in", "decimate",
           "inspection_points", "looks_like_cloud", "read_cloud"]
