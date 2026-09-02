# 3D CAD/스캔 데이터 가져오기

제로라인 판정의 나머지 절반을 여는 모듈이다.

## 왜 만들었나

현업 자료(2026-08-25)가 정리한 제로라인 판정 방법은 4가지인데, 지금까지
우리는 **3번 하나만** 쓰고 있었다.

| 방법 | 필요 데이터 | 이전 | 지금 |
|---|---|---|---|
| 1. RPS 정렬 | 3D 스캔 + CAD 기준점 | ❌ | 🟡 CAD 쪽 열림 |
| 2. 수축 중심선 | 3D 외곽 수축량 | ❌ | 🟡 CAD 쪽 열림 |
| 3. 컬러맵 제로존 | 2D 히트맵 | ✅ | ✅ |
| 4. 단면 분석 | 3D + 도면 | ❌ | 🟡 CAD 쪽 열림 |

부품별 우선순위를 보면 왜 3D 가 필요한지 분명하다 —

    선루프  : 1순위 가이드레일 장착 중심선, 2순위 섀시 조립 홀(Datum Hole)
    대시보드: 1순위 차량 센터 Y0, 3순위 크로스멤버 조립 마운트

전부 **조립 기준 좌표**다. 편차 히트맵 색에는 없는 정보다. 우리가 앞서
실측으로 확인한 "정답선 위 `|편차|` 평균이 0.273 이지 0 이 아니다"와
같은 얘기다.

## CATPart 를 왜 못 읽나 (실측)

`999 REINF SIDE OTR.CATPart` (53.7MB) 를 받아 분석한 결과다.

```
매직           V5_CFV2  (CATIA V5 R34 SP4 네이티브)
파트명         REINF SIDE OTR
zlib 스트림    0개
엔트로피       7.58 / 7.60 / 7.27  (8에 가까움 = 독자 인코딩)
내부 스트림    _PartBoundingBoxStream, _ReferencePlanes, HybridBody, CGMGeom
```

형상이 CGM(CATIA Geometric Modeler) 독자 포맷으로 인코딩돼 있다. 파일의
9.6% 가 "mm 처럼 보이는" float64 이지만 형상과 잡음을 구분할 방법이 없다.
명세 없이 파싱하는 건 성공 가능성이 낮은 연구과제이고, 이 PC 에는 CATIA
도 없다.

**온라인 변환 사이트는 쓰지 않는다.** 완성차 부품 CAD 를 외부에 올리는
건 회사 IP 유출이고, 이 프로젝트의 "모든 처리는 이 PC 안에서" 원칙에도
어긋난다.

→ **현업에 STEP(AP214) 또는 STL 내보내기를 요청한다.** CATIA 에서 2분이면
되는 작업이다.

## 무엇을 하나

```
mesh_io.py      STL/PLY/OBJ/GLB/3MF -> 웹 뷰어용 삼각망
step_reader.py  STEP -> 삼각망 + 원통면(홀/보스) + 평면 (RPS 후보)
```

### STEP vs STL

| | STEP | STL |
|---|---|---|
| 형상 | B-Rep 유지 | 삼각망만 |
| 홀 지름·중심 | ✅ 추출 가능 | ❌ 이미 뭉개짐 |
| 기준면 | ✅ 추출 가능 | ❌ |
| RPS 정렬 | ✅ 가능 | ❌ |

스캔 데이터는 보통 STL/PLY 로, CAD 는 STEP 으로 온다. 홀 정보가 필요하면
반드시 STEP 이어야 한다.

## 검증

정답(치수·홀 위치·지름)을 **우리가 지정해서** STEP 을 만들고 리더가 그걸
그대로 복원하는지 본다. `tests/test_cad_import.py` 7건 전부 통과.

실제 API 검증 (600×180×12mm 판재, 홀 4개):

```
치수(mm): [600.0, 180.0, 12.0]          <- 정답 일치
ø14.0  중심=(  60.0,  90.0, 6.0)  축=[0,0,1]
ø14.0  중심=( 540.0,  90.0, 6.0)  축=[0,0,1]
ø10.0  중심=( 200.0,  40.0, 6.0)  축=[0,0,1]
ø10.0  중심=( 400.0, 140.0, 6.0)  축=[0,0,1]
```

지름과 중심좌표가 지정값과 정확히 일치한다.

## 쓰는 법

UI 사이드바 **3D 데이터** 탭에서 파일을 놓거나, 직접:

```python
from cad_import import step_reader, mesh_io

parsed = step_reader.read_step_full("part.step")
print(parsed["counts"])          # {'cylinders': 4, 'holes': 4, 'planes': 6}
for hole in parsed["holes"]:
    print(hole["diameter"], hole["center"])

web = mesh_io.to_web_mesh(parsed["mesh"])   # three.js/WebGL 용 JSON
```

API: `POST /api/cad` (multipart, 필드명 `file`) — 최대 60MB.

## 아직 안 되는 것

- **RPS 정렬 자체**는 아직 없다. 홀 좌표를 뽑는 데까지가 지금 단계다.
  정렬하려면 **같은 부품의 3D 스캔**(STL/PLY/점군)이 있어야 하는데,
  현재 스캔은 전부 2D 히트맵 PNG 다.
- 어느 홀이 실제 RPS 점인지는 도면에 지정돼 있다. 여기서는 기하학적
  후보만 뽑고 최종 지정은 사람이 한다.
- 원통면이 홀인지 보스인지는 면 방향으로 판정한다. 열린 채널·복잡한
  형상에서는 틀릴 수 있다.

## 의존성

```
trimesh        메시 읽기/간략화
cadquery-ocp   OCCT 커널 (STEP 파싱·테셀레이션)
```

프론트 뷰어는 **의존성 없이** WebGL2 로 직접 그린다(`ui/app/cad-viewer.tsx`).
three.js 를 쓰려 했으나 이 저장소 node_modules 가 pnpm 트리라 npm/pnpm
양쪽 다 설치가 깨졌고, 필요한 게 삼각망 셰이딩 하나뿐이라 직접 그리는
쪽이 낫다고 판단했다.
