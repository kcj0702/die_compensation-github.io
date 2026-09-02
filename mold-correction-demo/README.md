# Mold Correction Demo

3D 스캔 편차 맵 한 장을 받아 **제로라인을 찾고, 보정치 시트를 만들고, CAD 형상 위에
얹어 확인**하는 로컬 도구다. 모든 처리는 이 PC 안에서 이뤄지며 외부로 나가는 통신은 없다.

## 현재 구현 범위

| 경로 | 상태 | 역할 |
|---|---|---|
| `deviation_extraction/` | 구현 | 라벨 검출, 좌표 산정, 편차값 판독, CSV 저장 |
| `product_alignment/` | 구현 | 제품데이터 등록·정렬, 측정점을 제품데이터 좌표로 전사 |
| `depth_measurement/` | 골격 | 깊이 측정 단계 예정 |
| `pipeline/` | 골격 | `run_demo.py` 는 아직 빈 파일이다 |
| `docs/` | 부분 구현 | 편차 추출 단계의 입출력 계약 |

세부 내용은 각 폴더의 README 를 본다 —
[`deviation_extraction`](deviation_extraction/README.md) ·
[`zero_line_detection`](zero_line_detection/README.md) ·
[`zero_line_advance`](zero_line_advance/README.md) ·
[`cad_import`](cad_import/README.md) ·
[`ui`](ui/README.md)

## 처리 흐름

```text
편차 맵 → 라벨 박스 검출 → 리더 선분 기반 좌표 산정 → VLM 숫자 판독 → CSV·디버그 이미지
                                                       └→ 선택: 제로 라인·컬러맵 확인
                                                       └→ 선택: 제품데이터 정렬 후 좌표 전사
```

검출 결과는 이미지 좌표계의 2차원 픽셀 좌표다. 부품 좌표계나 3차원 좌표로 변환하지 않는다.

보정시트에 들어가는 그림은 편차 히트맵이 아니라 깨끗한 제품데이터 렌더다.
`product_alignment`이 두 이미지를 맞춰 측정점을 옮기며, 좌표만 옮기고 편차값을
보정치로 바꾸지는 않는다. 자세한 내용은
[`product_alignment/README.md`](product_alignment/README.md)에 있다.

## 실행 환경

- Python 3.10 이상
- 최초 모델 로드 시 Hugging Face 모델을 받을 수 있는 환경 또는 준비된 로컬 캐시
- CUDA 사용 가능 시 FP16, 그 외에는 CPU FP32로 추론

저장소 루트에서 의존성을 설치한다. `run-ui.cmd`가 찾는 위치도 이곳의 `.venv`다.

torch는 PyPI 기본 휠이 CPU 전용이라 CUDA 인덱스에서 먼저 받는다. UI 백엔드는
CUDA가 없으면 Qwen 판독을 건너뛴다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r mold-correction-demo\deviation_extraction\requirements.txt
.venv\Scripts\python.exe -m pip install -r mold-correction-demo\ui\backend\requirements.txt
```

`ui/backend/requirements.txt`에는 파일 업로드 파싱에 필요한 `python-multipart`가
들어 있다. 이걸 빠뜨리면 서버는 뜨지만 `/api/analyze`가 form 파싱 오류로 실패한다.

## 실행

### 화면과 엔진 함께 띄우기

```powershell
ui\run-ui.cmd
```

`ui\stop-ui.cmd` 로 함께 내린다. 단, 이 스크립트는 `.venv` 가
**`mold-correction-demo` 폴더 옆**에 있다고 보고 찾는다. 그 자리에 없으면 아래처럼
직접 띄운다.

### 따로 띄우기

엔진(백엔드) — `127.0.0.1:8000`

```powershell
<venv>\Scripts\python.exe ui\backend\server.py
```

이 PC 에서는 CUDA PyTorch 와 `bitsandbytes` 를 둘 다 가진 환경이 하나뿐이다.

```
C:\Users\KDT033\Downloads\die_compensation-github.io-main\die_compensation-github.io-main\.venv
```

**엔진은 자동 재적재를 하지 않는다.** 파이썬 코드를 고쳤으면 반드시 다시 띄운다.

무거운 두 가지는 디스크에 남으므로 다시 띄워도 잃지 않는다 —

| 무엇 | 어디에 | 실측 |
|---|---|---|
| Qwen 라벨 판독 | `ui/backend/.label_cache.json` | 64XX2 한 장 71초 -> 0초 |
| 현업 제로라인 파이프라인 | `zero_line_detection/.lab_cache/` | 64XX2 117초 -> 0.05초 |
| STEP 파싱 | `cad_import/_parsed/` | 113MB 57초 -> 3초 |

분석 한 장이 **195초 -> 3.1초**가 된다. 열쇠는 내용 해시라 그림이나
스크립트가 바뀌면 저절로 다시 돈다. 자리를 옮기려면 `ADC_LABEL_CACHE`
`ADC_LAB_CACHE` 환경변수를 쓴다(시험이 이걸로 실제 캐시를 지킨다).
`GET /api/health` 의 `qwenLoaded` 가 분석을 돌린 뒤에도 `false` 면 라벨 판독이
안 되고 있다는 뜻이다.

화면(프론트) — `127.0.0.1:3000`

```powershell
cd ui
npm run dev
```

### 명령줄만 쓰기

편차 포인트만 뽑을 때:

```powershell
<venv>\Scripts\python.exe deviation_extraction/run.py `
  --image path/to/deviation_map.png `
  --out data/intermediate/deviation_points.csv `
  --debug
```

제로라인만 볼 때는 `zero_line_detection/run.py` 를 쓴다.

## 화면 네 개

왼쪽 메뉴 순서가 곧 작업 순서다.

| # | 화면 | 하는 일 |
|---|---|---|
| 01 | 분석 작업실 | 스캔 등록, **품번 지정**, 분석 실행 |
| 02 | 엔진 결과 | 엔진 세 개가 각각 낸 결과를 단계별로 확인 |
| 03 | ADC 보정 시트 | 보정 계수, 값 수정, 핵심 포인트, 주석, 엑셀 내보내기 |
| 04 | 3D 데이터 | CAD 위에 제로라인·보정량 얹기, 단면·측정·주석, 보정 후 형상 |

### 품번은 반드시 맞춰야 한다

품번이 **컬러바 범위와 제로라인 파라미터를 고르는 열쇠**다.

| 품번 | 컬러바 범위 |
|---|---|
| `64XX2` | −1.6 ~ +2.0 mm |
| `67XX6` | −3.0 ~ +3.0 mm |
| `71XX2` | −2.0 ~ +2.0 mm |

파일명에 품번이 있으면 자동으로 잡고, 없으면 분석 작업실의 파일 행에서 직접 고른다.
지정하지 않으면 제로라인 단계가 통째로 빈다 — 같은 그림을 파일명만 바꿔 확인했다.

```
_boundary_anchors.png                    제로라인 0개
JD_67XX6-DR000 3D 스캔.png (같은 그림)     제로라인 3개
```

### 3D 뷰어 마우스 (CATIA 와 같다)

| 조작 | 동작 |
|---|---|
| 가운데 끌기 | 이동 |
| 가운데 + 오른쪽 끌기 | 회전 (누른 도중에 더해도 바뀐다) |
| Ctrl + 가운데 끌기 | 확대 · 축소 |
| 휠 | 확대 · 축소 |
| 왼쪽 | 선택. 콜아웃을 누르면 값을 고친다 |

CATIA 네이티브(`.CATPart`)는 독자 포맷이라 읽지 못한다. STEP(AP214)으로 내보낸다.

## 테스트

```powershell
<venv>\Scripts\python.exe -m pytest -q
```

회귀 테스트는 합성 이미지를 쓰며 회사 원본이나 모델 가중치를 쓰지 않는다.

화면 쪽은 타입 검사와 빌드로 확인한다.

```powershell
cd ui
npx tsc --noEmit
npm run build
```

## 현재 제약

- 대칭 부품은 제품데이터 정렬 방향을 이미지만으로 정할 수 없어 사람이 한 번 확인해야 한다.
- 검출 임계값이 픽셀 크기와 색상에 고정되어 있어 입력 해상도와 스캔 조건에 민감하다.
- 꺾이거나 교차하는 리더라인은 단일 Hough 선분만으로 정확히 연결하기 어렵다.
- 저장소에 검증 이미지와 정답 데이터가 없어 검출 정확도는 아직 계량되지 않았다.
- `pipeline/run_demo.py`를 포함한 전체 보정 파이프라인은 아직 구현되지 않았다.
