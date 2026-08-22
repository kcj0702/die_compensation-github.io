# Deviation Extraction

편차 맵에 인쇄된 숫자 라벨을 검출하고, 라벨이 지시하는 픽셀 좌표와 판독값을 CSV로 저장한다.
좌표는 원본 이미지 기준이며 부품 좌표계나 3차원 좌표가 아니다.

## 입력

| 입력 | 필수 | 기본 경로 | 조건 |
|---|---:|---|---|
| 편차 맵 | 예 | `data/intermediate/deviation_map.png` | OpenCV로 읽을 수 있는 컬러 이미지 |
| 제로 라인 마스크 | 아니요 | `data/intermediate/zero_line_mask.png` | 0이 배경이고 0보다 큰 픽셀이 전경인 이미지 |
| 컬러바 ROI | 아니요 | `config.COLORBAR_ROI` | 원본 기준 `(x, y, width, height)` |

현재 검출기는 다음 표기를 전제로 한다.

- 숫자 라벨은 무채색 테두리 또는 빨간 채움 영역을 가진 가로형 박스다.
- 리더라인과 종점은 `config.py`의 파란색 HSV 범위 안에 있다.
- 숫자는 부호가 있는 정수 또는 `0.5` 형태의 소수로 적혀 있다.

## 코드 구성

| 파일 | 역할 |
|---|---|
| `run.py` | CLI 인자 처리와 추출 단계 실행 |
| `label_detector.py` | 라벨 박스, 리더 선분, 종점 후보 검출 |
| `vlm_reader.py` | 라벨 crop의 숫자 판독 |
| `point_extractor.py` | 검출 결과 결합, 상태 판정, 파일 저장 |
| `colormap_reader.py` | 컬러바 또는 기준 컬러맵으로 BGR 값을 mm 값에 대응 |
| `calibrate_colorbar.py` | 컬러바 설정값 산출 보조 |
| `config.py` | 경로, 모델 ID, 검출 임계값 |

## 처리 방식

1. 편차 맵을 회색조로 바꾸고 어두운 픽셀을 이진화한다.
2. 파란 지시선을 제외한 라벨 마스크를 면적, extent, 종횡비로 걸러 박스를 찾는다.
3. 박스에 닿은 Hough 선분 중 가장 긴 유효 선분의 반대쪽 끝을 좌표 후보로 삼는다.
4. 후보 끝점 반경 안에 작은 파란 점이 있으면 점의 무게중심으로 좌표를 보정한다.
5. 라벨 crop을 VLM으로 판독하고 생성문에서 첫 번째 숫자를 `value_mm`로 사용한다.
6. 선택적으로 제로 라인 포함 여부와 컬러맵 값 차이를 기록한다.

리더 선분을 찾지 못하면 좌표를 빈 값으로 남긴다. 라벨 중심을 실제 측정점으로 대체하지 않는다.

### 좌표 규칙

- 원점은 이미지 좌상단이며 `x`는 오른쪽, `y`는 아래쪽으로 증가한다.
- NumPy 배열은 `[y, x]`, 출력 레코드는 `(x, y)` 순서다.
- `x_norm = x / image_width`, `y_norm = y / image_height`이며 소수 넷째 자리로 반올림한다.
- 유효 픽셀의 정규화 좌표 범위는 `0 <= value < 1`이다.

### 검출 상태

`confidence`는 확률값이 아니라 후처리 상태 코드다.

| 값 | 의미 |
|---|---|
| `ok` | 리더 선분을 찾았고 교차검증을 생략했거나 큰 차이가 없음 |
| `value_not_read` | VLM 생성문에서 숫자를 찾지 못함 |
| `leader_line_not_traced` | 리더 선분을 찾지 못해 좌표를 비워 둠 |
| `color_mismatch` | VLM 값과 좌표 픽셀의 컬러맵 값 차이가 임계값보다 큼 |

문제가 둘 이상이면 `value_not_read|leader_line_not_traced`처럼 `|`로 연결한다.

## 설치와 실행

프로젝트 루트에서 실행한다.

```powershell
.venv\Scripts\python.exe -m pip install -r deviation_extraction/requirements.txt

.venv\Scripts\python.exe deviation_extraction/run.py `
  --image path/to/deviation_map.png `
  --zero-line-mask data/intermediate/zero_line_mask.png `
  --out data/intermediate/deviation_points.csv `
  --debug `
  --debug-out data/intermediate/deviation_points_debug.png
```

모델은 기본적으로 `Qwen/Qwen2.5-VL-3B-Instruct`를 사용한다. CUDA가 감지되면 FP16,
그 외에는 CPU FP32로 로드한다. `--offline`은 로컬 캐시만 사용하고 다운로드를 차단한다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--image` | `data/intermediate/deviation_map.png` | 편차 맵 경로 |
| `--zero-line-mask` | `data/intermediate/zero_line_mask.png` | 없으면 제로 라인 판정을 생략 |
| `--out` | `data/intermediate/deviation_points.csv` | CSV 저장 경로 |
| `--model` | `config.VLM_MODEL_ID` | 호환되는 Hugging Face 모델 ID |
| `--device` | 자동 선택 | `cuda`, `cuda:0`, `cpu` 등 추론 장치 |
| `--offline` | 꺼짐 | 모델을 로컬 파일에서만 로드 |
| `--cross-check` | 꺼짐 | 좌표 색상과 VLM 값 비교 |
| `--debug` | 꺼짐 | 검출 오버레이 이미지 저장 |
| `--debug-out` | `config.DEBUG_IMAGE_PATH` | 디버그 이미지 저장 경로 |

디버그 이미지는 라벨 박스, 연결선, 끝점, 판독값을 함께 표시한다.

## 컬러바 설정

`--cross-check`는 VLM 판독값을 바꾸지 않고 차이가 큰 포인트의 상태만 표시한다.
`COLORBAR_ROI`가 없으면 `jet`, -3.0~3.0 mm를 기준 LUT로 사용한다.

실제 컬러바를 사용할 때는 설정값 산출 도구를 실행한다.

```powershell
.venv\Scripts\python.exe deviation_extraction/calibrate_colorbar.py `
  --image path/to/deviation_map.png `
  --roi 950 120 20 300
```

도구가 출력한 `COLORBAR_ROI`, `COLORBAR_MIN_MM`, `COLORBAR_MAX_MM`을 검토한 뒤 `config.py`에
직접 반영한다. 세로 범례는 위가 최댓값, 가로 범례는 오른쪽이 최댓값이라는 규칙을 사용한다.

ROI 중앙선은 색 띠를 지나야 하고, 긴 축 양 끝 crop에는 최솟값·최댓값 숫자가 포함되어야 한다.

## 출력 CSV

| 컬럼 | 설명 |
|---|---|
| `point_id` | 라벨 박스를 위에서 아래, 같은 높이에서는 왼쪽부터 정렬한 실행 내 ID |
| `x_px`, `y_px` | 원본 이미지의 픽셀 좌표. 리더 선분 미검출 시 빈 값 |
| `x_norm`, `y_norm` | 이미지 폭과 높이로 나눈 정규화 좌표 |
| `value_mm` | VLM 생성문에서 파싱한 첫 숫자. 판독 실패 시 빈 값 |
| `label_color` | 상단 내부 패치를 `red` 또는 `white`로 단순 분류한 결과 |
| `in_zero_line` | 마스크가 있으면 해당 좌표의 픽셀이 0보다 큰지 여부 |
| `confidence` | 확률이 아닌 검출 상태 코드 |

마스크 크기가 원본과 다르면 최근접 보간으로 맞춘다. `point_id`는 실행 내 정렬 ID이며 데이터 간
영구 ID를 보장하지 않는다.

## 자동 검증

합성 이미지와 가짜 숫자 판독기를 사용하므로 회사 원본과 모델 가중치가 필요하지 않다.

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
```

## 현재 제약

- 면적과 거리 임계값이 픽셀 단위라 입력 배율이 달라지면 재조정이 필요하다.
- 히트맵의 파란 영역도 지시선 마스크에 포함될 수 있어 교차선과 꺾인 선에 취약하다.
- 컬러 교차검증은 좌표의 단일 BGR 픽셀을 사용하므로 파란 점이나 선 색의 영향을 받을 수 있다.
- 라벨별 VLM 추론을 순차 실행하며 배치 추론은 구현되어 있지 않다.
- 저장소에 검증 샘플과 정답셋이 없어 정확도 지표는 제공하지 않는다.
