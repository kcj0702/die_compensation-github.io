"""PolyWorks 점군 읽기 시험.

실제 워크스페이스는 1.9GB 라 저장소에 넣지 않는다. 대신 실측 파일에서
알아낸 규칙 그대로 작은 파일을 지어 놓고 읽힌다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import polyworks  # noqa: E402


def make_vault_file(target: Path, points: np.ndarray,
                    tail: bytes = b"") -> Path:
    """실측 규칙대로 점군 파일을 짓는다 — 머리 8바이트 + float64 xyz."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        out.write(polyworks.MAGIC + b"\x00\x00\x00\x00")
        out.write(points.astype("<f8").tobytes())
        out.write(tail)
    return target


@pytest.fixture()
def cloud_file(tmp_path: Path) -> tuple:
    rng = np.random.default_rng(20260904)
    points = np.column_stack([
        rng.uniform(0.0, 1119.0, 5000),
        rng.uniform(-927.0, 603.0, 5000),
        rng.uniform(-230.0, 10.0, 5000),
    ])
    path = make_vault_file(tmp_path / "ab" / "abcdef", points)
    return path, points


def test_머리를_보고_점군_파일을_가린다(tmp_path, cloud_file):
    path, _ = cloud_file
    assert polyworks.looks_like_cloud(path)

    남 = tmp_path / "남의파일"
    남.write_bytes(b"<?xml version=\"1.0\"?><Objects/>")
    assert not polyworks.looks_like_cloud(남)


def test_좌표를_그대로_읽는다(cloud_file):
    path, points = cloud_file
    cloud = polyworks.read_cloud(path)
    assert cloud is not None
    assert len(cloud.points) == len(points)
    assert np.allclose(cloud.points, points)


def test_꼬리에_붙은_다른_데이터는_잘라낸다(tmp_path):
    """실측 파일은 좌표 뒤에 좌표가 아닌 데이터가 붙어 있었다.

    걸러내기로는 자리가 어긋난 값이 범위 안에 들어 새어 나온다. 그래서
    처음 깨지는 자리에서 자른다 — 잘린 뒤 크기가 원래 점군 그대로여야
    한다(실측에서는 이것 때문에 Z 범위가 240mm 에서 960mm 로 부풀었다).
    """
    rng = np.random.default_rng(7)
    points = np.column_stack([
        rng.uniform(0.0, 1000.0, 4000),
        rng.uniform(-900.0, 600.0, 4000),
        rng.uniform(-230.0, 10.0, 4000),
    ])
    쓰레기 = np.array([np.nan, -1.79e308, 3.1e224] * 400, dtype="<f8")
    path = make_vault_file(tmp_path / "cd" / "beefbeef", points,
                           tail=쓰레기.tobytes())

    cloud = polyworks.read_cloud(path)
    assert cloud is not None
    assert len(cloud.points) == len(points)
    # 꼬리가 새어 들어왔다면 Z 범위가 원래보다 넓어진다.
    assert cloud.size[2] <= 241.0


def test_원점_채움은_버린다(tmp_path):
    """블록 끝의 (0,0,0) 을 그대로 두면 정합이 원점으로 끌려간다."""
    points = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    채움 = np.zeros((50, 3))
    path = make_vault_file(tmp_path / "ef" / "cafecafe",
                           np.vstack([points, 채움]))

    cloud = polyworks.read_cloud(path)
    assert cloud is not None
    assert len(cloud.points) == 2


def test_좌표가_아니면_받지_않는다(tmp_path):
    """규칙이 안 맞는 파일을 점군이라고 우기지 않는다."""
    쓰레기 = np.full((3000, 3), 1e20)
    path = make_vault_file(tmp_path / "aa" / "deadbeef", 쓰레기)
    assert polyworks.read_cloud(path) is None


def test_솎아도_고르게_남는다():
    points = np.arange(30_000, dtype=float).reshape(-1, 3)
    작게 = polyworks.decimate(points, 1000)
    assert len(작게) <= 1000
    # 앞뒤 어느 한쪽에 몰리면 안 된다 — 전체를 훑어 집어야 한다.
    assert 작게[0][0] == points[0][0]
    assert 작게[-1][0] > points[len(points) // 2][0]


def test_PLY_로_쓰고_다시_읽는다(tmp_path, cloud_file):
    path, points = cloud_file
    cloud = polyworks.read_cloud(path)
    assert cloud is not None

    ply = cloud.to_ply(tmp_path / "scan.ply")
    raw = ply.read_bytes()
    머리 = raw.split(b"end_header\n", 1)[0].decode("ascii")
    assert f"element vertex {len(points)}" in 머리

    몸 = raw.split(b"end_header\n", 1)[1]
    다시 = np.frombuffer(몸, dtype="<f4").reshape(-1, 3)
    assert len(다시) == len(points)
    # float32 로 줄여 쓰므로 그만큼은 어긋난다 — 마이크로미터 안이면 된다.
    assert np.abs(다시 - points).max() < 0.001


def hex_of(value: float) -> str:
    """실수를 워크스페이스가 쓰는 빅엔디언 hex 글자로 바꾼다."""
    import struct as _struct
    return _struct.pack(">d", value).hex().upper()


def write_metadata(vault: Path, body: str) -> Path:
    target = vault / "2c" / "meta"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n'
        f"<Objects>{body}</Objects>\n" + " " * 1200,
        encoding="utf-8")
    return target


def surf_point(name: str, spot: tuple, normal: tuple, deviation: float) -> str:
    """실측 구조 그대로 검사 포인트 하나를 짓는다."""
    corner = "".join(f"<E>{hex_of(v)}</E>" for v in spot)
    way = "".join(f"<E>{hex_of(v)}</E>" for v in normal)
    return (
        f'<O id="x" clsid="CmpPtSurf">'
        f'<P id="Name">{name}</P>'
        f'<P id="PiercePt">{corner}</P>'
        f'<P id="EffectiveNormal">{way}</P>'
        f"</O>"
        f'<O id="y" clsid="CmpPtDim">'
        f'<P id="DimName"><E>1</E><E>Surface Distance</E></P>'
        f'<P id="Deviation">{hex_of(deviation)}</P>'
        f"</O>"
    )


def test_검사_포인트를_자리와_값째로_꺼낸다(tmp_path):
    work = tmp_path / "스캔.pwk"
    work.write_text("<?xml version=\"1.0\"?><PolyworksWorkspace/>",
                    encoding="utf-8")
    vault = tmp_path / "스캔_Files" / "wm-data" / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    write_metadata(vault, (
        surf_point("surf pt 6", (473.8, -739.3, 340.7), (-0.05, -0.99, 0.0), -2.79)
        + surf_point("surf pt 7", (476.7, -743.8, 374.6), (0.0, -1.0, 0.0), -2.02)
    ))

    points = polyworks.inspection_points(work)
    assert [p.name for p in points] == ["surf pt 6", "surf pt 7"]
    assert points[0].position == pytest.approx((473.8, -739.3, 340.7))
    assert points[0].deviation == pytest.approx(-2.79)
    assert points[1].deviation == pytest.approx(-2.02)
    assert points[0].normal == pytest.approx((-0.05, -0.99, 0.0))


def test_표면거리가_아닌_값은_섞지_않는다(tmp_path):
    """한 포인트에 각도·반경 같은 다른 값이 함께 붙는다.

    실측 파일은 CmpPtSurf 79개에 CmpPtDim 이 1580개였다 — 포인트마다
    값이 여러 개다. 표면 거리만 골라야 편차가 엉뚱한 값으로 덮이지 않는다.
    """
    work = tmp_path / "스캔.pwk"
    work.write_text("<x/>", encoding="utf-8")
    vault = tmp_path / "스캔_Files" / "wm-data" / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    write_metadata(vault, (
        f'<O clsid="CmpPtSurf">'
        f'<P id="Name">surf pt 1</P>'
        f'<P id="PiercePt">{"".join(f"<E>{hex_of(v)}</E>" for v in (1.0, 2.0, 3.0))}</P>'
        f"</O>"
        # 표면 거리가 아닌 값이 먼저 온다 — 이걸 집으면 안 된다.
        f'<O clsid="CmpPtDim">'
        f'<P id="DimName"><E>1</E><E>Angle</E></P>'
        f'<P id="Deviation">{hex_of(88.0)}</P>'
        f"</O>"
        f'<O clsid="CmpPtDim">'
        f'<P id="DimName"><E>1</E><E>Surface Distance</E></P>'
        f'<P id="Deviation">{hex_of(-0.42)}</P>'
        f"</O>"
    ))

    points = polyworks.inspection_points(work)
    assert len(points) == 1
    assert points[0].deviation == pytest.approx(-0.42)


def test_워크스페이스에서_점군을_찾는다(tmp_path):
    """`.pwk` 를 주면 옆의 `_Files/wm-data/vault` 를 훑는다."""
    work = tmp_path / "스캔.pwk"
    work.write_text("<?xml version=\"1.0\"?><PolyworksWorkspace/>",
                    encoding="utf-8")
    vault = tmp_path / "스캔_Files" / "wm-data" / "vault"

    rng = np.random.default_rng(3)
    make_vault_file(vault / "11" / "aaa",
                    rng.uniform(-100, 100, (20_000, 3)))
    make_vault_file(vault / "22" / "bbb",
                    rng.uniform(-100, 100, (12_000, 3)))
    (vault / "33").mkdir(parents=True, exist_ok=True)
    (vault / "33" / "ccc").write_bytes(b"<?xml version=\"1.0\"?>")

    clouds = polyworks.clouds_in(work)
    assert len(clouds) == 2
    # 점이 많은 것이 먼저다.
    assert len(clouds[0].points) >= len(clouds[1].points)
