# Mold Correction Demo

3D 스캔 편차 맵 한 장을 받아 **제로라인을 찾고, 보정치 시트를 만들고, CAD 형상 위에
얹어 확인**하는 로컬 도구다. 모든 처리는 이 PC 안에서 이뤄지며 외부로 나가는 통신은 없다.

## 현재 구현 범위

| 경로 | 상태 | 역할 |
|---|---|---|
| `label_removal/` | 구현 | 숫자 라벨 검출과 인페인팅 |
| `deviation_extraction/` | 구현 | 라벨 박스 검출, 리더선 끝점 좌표, VLM 숫자 판독, CSV 저장 |
| `zero_line_detection/` | 구현 | 컬러바 판독, 제로라인 검출, 핵심 포인트 선별, 현업 양식 엑셀 |
| `zero_line_advance/` | 구현 | 편차 계곡을 따라가는 제로라인 후보 |
| `cad_import/` | 구현 | STEP 읽기(OCCT), 홀·평면 추출, 실루엣 정합, 메시 변형 |
| `ui/backend/` | 구현 | 세 엔진을 묶는 로컬 API 서버 (11개 엔드포인트) |
| `ui/app/` | 구현 | 화면 4개와 three.js 3D 뷰어 |
| `shared/` | 구현 | 엔진 사이 공통 스키마 |
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
편차 맵(PNG)
  → 라벨 검출·인페인팅            label_removal
  → 리더선 끝점 + VLM 숫자 판독   deviation_extraction
  → 컬러바 판독 → 색을 mm 로      zero_line_detection
  → 제로라인 · 핵심 포인트
  → 보정치 시트(현업 양식 xlsx)
  → CAD 표면에 얹기 · 보정 후 형상 cad_import
```

좌표는 이미지 픽셀 기준이다. **부품 좌표계나 차량 좌표계로는 아직 변환하지 않는다** —
그러려면 3D 스캔 원본이 필요하다(아래 "현재 제약" 참고).

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

### 3D 스캔 원본이 없다

지금 받는 것은 검사 소프트웨어가 만든 **편차 히트맵 그림(PNG)** 한 장이다. 그 그림을
만들어 낸 측정 데이터 자체 — 점군, 측정 메시, 검사 프로젝트 파일, 점별 편차 표 — 는
아직 없다. 그래서 이런 것들이 막혀 있다.

- **좌표계가 픽셀이다.** 현업이 정한 제로라인 기준은 "가이드레일 장착 중심선",
  "차량 센터 Y0" 처럼 전부 **조립 기준 좌표**인데, 히트맵 색에는 그 정보가 없다.
- **RPS 정렬을 못 한다.** 현업 자료가 정리한 제로라인 판정 4가지 중 3가지
  (RPS 정렬 · 수축 중심선 · 단면 분석)가 3D 데이터를 전제한다. 지금은 4번째인
  컬러맵 제로존 하나만 쓴다. CAD 쪽 홀 좌표는 뽑아 뒀으니 스캔 쪽만 오면 된다.
- **값을 색과 글자로 추정한다.** 실측 67XX6 에서 편차 포인트 130개 중 33개가
  컬러바 범위를 벗어난 판독값이었다. 원본 수치가 있으면 이 오차가 통째로 사라진다.

### 그 밖에

- CAD 정합은 데이텀이 아니라 **실루엣 겉모양**으로 맞춘다. 겹침 비율(IoU)을 함께
  내보내므로 낮으면 화면에서 경고한다.
- 핵심 포인트 선별 기준은 **현업 확인 전**이다. 보정시트를 보고 세운 규칙이다.
- "보정 후 형상"은 B-Rep 이 아니라 **삼각망을 민 것**이라 그대로 가공에 쓸 수 없다.
  눈으로 견주고 STL 로 넘기는 용도다.
- 검출 임계값이 픽셀 크기와 색상에 묶여 있어 입력 해상도와 스캔 조건에 민감하다.
- 꺾이거나 교차하는 리더선은 단일 Hough 선분만으로 정확히 잇기 어렵다.
- `pipeline/run_demo.py` 를 포함한 일괄 실행 파이프라인은 아직 없다.
