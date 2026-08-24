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

- 숫자 라벨은 회색/검은 테두리 또는 빨간 채움 영역을 가진 가로형 박스다. 밝은 흰 라벨은 넓은 중성 회색 테두리와 내부의 검은 숫자를 함께 확인한다. 테두리가 잘렸거나 끊긴 경우에는 밝은 내부, 검은 숫자, 실제 파란 리더가 모두 있어야 한다.
- 리더라인 중심에는 순수 파랑에 가까운 얇은 픽셀이 있으며 라벨 박스 근처에서 시작한다.
- 배경은 흰색에 가깝고, 스캔 본체는 주석선보다 훨씬 큰 조밀한 전경이다.
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
| `image_io.py` | 한글 등 유니코드 경로의 이미지 입출력 |
| `config.py` | 경로, 모델 ID, 검출 임계값 |
| `__init__.py` | 패키지 import용 공개 진입점 |

## 처리 방식

1. 흰 배경과의 색 차이로 전경을 만들고 morphology opening으로 얇은 주석을 끊는다.
2. 가장 큰 조밀 연결 성분을 스캔 본체로 선택하고 경계 픽셀을 복구한다.
3. 빨간 채움 연결 성분과 회색/검은 폐곡선 contour를 크기, 채움률, extent로 걸러 라벨 박스를 찾는다. 회색 범위를 넓혀 찾은 흰 라벨 후보는 내부 검은 숫자가 있을 때만 채택한다. 폐곡선을 만들지 못한 흰 라벨은 밝은 내부 연결 성분을 복원한 뒤 실제 리더 연결까지 확인한다.
4. 한 라벨의 내부/외부 contour가 겹치면 IoU로 중복을 제거한다.
5. 정확한 파란 중심선 중 7×7 이웃에서 얇은 픽셀만 남겨 파란 히트맵 면을 배제한다. 엄격 마스크로 찾지 못한 라벨만 완화 마스크로 재시도한다.
6. 라벨 박스를 확장한 근처까지 이어진 연결 성분을 가장 가까운 단일 라벨에 할당한다. 그 라벨에서 가장 먼 스캔 내부 끝점을 우선 사용하고, 스캔과 겹치지 않으면 성분 전체의 끝점을 사용한다. 점 보정은 스캔 내부에서만 수행한다.
7. 모든 라벨 crop을 VLM 배치로 판독하고 생성문에서 첫 번째 숫자를 `value_mm`로 사용한다.
8. 선택적으로 제로 라인 포함 여부와 컬러맵 값 차이를 기록한다. 컬러 교차검증은 파란 선과 점을 제외한 종점 주변 표면색 중앙값을 사용한다.

리더 연결 성분을 찾지 못하면 좌표를 빈 값으로 남긴다. 라벨 중심을 실제 측정점으로 대체하지 않는다.
기존 `_trace_leader_line()` Hough 보조 함수는 하위 호환을 위해 남아 있지만 기본 검출 흐름에서는 사용하지 않는다.

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
| `color_sample_unavailable` | 파란 주석을 제외한 주변 스캔 표면색을 얻지 못함 |

문제가 둘 이상이면 `value_not_read|leader_line_not_traced`처럼 `|`로 연결한다.

## 설치와 실행

프로젝트 루트에서 실행한다.

```powershell
.venv\Scripts\python.exe -m pip install -r deviation_extraction/requirements.txt

.venv\Scripts\python.exe deviation_extraction/run.py `
  --image path/to/deviation_map.png `
  --zero-line-mask data/intermediate/zero_line_mask.png `
  --out data/intermediate/deviation_points.csv `
  --batch-size 8 `
  --debug `
  --debug-out data/intermediate/deviation_points_debug.png
```

패키지 방식으로도 같은 CLI를 실행할 수 있다.

```powershell
.venv\Scripts\python.exe -m deviation_extraction.run --image path/to/deviation_map.png
```

모델은 기본적으로 `Qwen/Qwen2.5-VL-3B-Instruct`를 사용한다. CUDA가 감지되면 FP16을 사용하되
GPU 메모리가 10 GiB 미만이면 자동으로 8-bit 로드하며, 그 외에는 CPU FP32로 로드한다.
`--offline`은 로컬 캐시만 사용하고 다운로드를 차단한다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--image` | `data/intermediate/deviation_map.png` | 편차 맵 경로 |
| `--zero-line-mask` | `data/intermediate/zero_line_mask.png` | 없으면 제로 라인 판정을 생략 |
| `--out` | `data/intermediate/deviation_points.csv` | CSV 저장 경로 |
| `--model` | `config.VLM_MODEL_ID` | 호환되는 Hugging Face 모델 ID |
| `--device` | 자동 선택 | `cuda`, `cuda:0`, `cpu` 등 추론 장치 |
| `--batch-size` | `8` | 한 번에 VLM으로 판독할 라벨 crop 수 |
| `--offline` | 꺼짐 | 모델을 로컬 파일에서만 로드 |
| `--cross-check` | 꺼짐 | 좌표 색상과 VLM 값 비교 |
| `--debug` | 꺼짐 | 검출 오버레이 이미지 저장 |
| `--debug-out` | `config.DEBUG_IMAGE_PATH` | 디버그 이미지 저장 경로 |

디버그 이미지는 라벨 박스, 연결선, 끝점, 판독값을 함께 표시한다.

## 컬러바 설정

`--cross-check`는 VLM 판독값을 바꾸지 않고 차이가 큰 포인트의 상태만 표시한다.
파란 주석을 제외한 주변 스캔 표면의 중앙 색을 사용한다. 유효한 주변 표본이 전혀 없으면
종점의 파란 픽셀로 되돌아가지 않고 `color_sample_unavailable` 상태를 기록한다.
`COLORBAR_ROI`가 없으면 `jet`, -3.0~3.0 mm를 기준 LUT로 사용한다.

실제 컬러바를 사용할 때는 설정값 산출 도구를 실행한다.

```powershell
.venv\Scripts\python.exe deviation_extraction/calibrate_colorbar.py `
  --image path/to/deviation_map.png `
  --roi 950 120 20 300
```

도구가 출력한 `COLORBAR_ROI`, `COLORBAR_MIN_MM`, `COLORBAR_MAX_MM`을 검토한 뒤 `config.py`에
직접 반영한다. 세로 범례는 위가 최댓값, 가로 범례는 오른쪽이 최댓값이라는 규칙을 사용한다.

ROI 중앙 띠는 색 띠를 지나야 한다. LUT는 중앙 한 줄 대신 중앙 띠의 중앙값을 사용한다.
긴 축 양 끝 crop은 짧은 축 방향으로도 `--margin`만큼 넓혀 최솟값·최댓값 숫자를 포함한다.

## 출력 CSV

| 컬럼 | 설명 |
|---|---|
| `point_id` | 라벨 박스를 위에서 아래, 같은 높이에서는 왼쪽부터 정렬한 실행 내 ID |
| `x_px`, `y_px` | 원본 이미지의 픽셀 좌표. 리더 선분 미검출 시 빈 값 |
| `x_norm`, `y_norm` | 이미지 폭과 높이로 나눈 정규화 좌표 |
| `value_mm` | VLM 생성문에서 파싱한 첫 숫자. 판독 실패 시 빈 값 |
| `label_color` | 테두리를 제외한 박스 내부의 빨간 픽셀 비율을 `red` 또는 `white`로 분류한 결과 |
| `in_zero_line` | 마스크와 좌표가 모두 있으면 해당 픽셀이 0보다 큰지 여부. 둘 중 하나가 없으면 `False` |
| `confidence` | 확률이 아닌 검출 상태 코드 |

마스크 크기가 원본과 다르면 최근접 보간으로 맞춘다. `point_id`는 실행 내 정렬 ID이며 데이터 간
영구 ID를 보장하지 않는다.

## 자동 검증

합성 이미지와 가짜 숫자 판독기를 사용하므로 회사 원본과 모델 가중치가 필요하지 않다.

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
.venv\Scripts\python.exe -B -m unittest discover -s deviation_extraction/tests -p "test_*.py" -v
```

## 현재 제약

- 라벨 최소 크기와 export 색 임계값은 설정값이므로 다른 렌더러 형식에서는 `config.py` 조정이 필요하다.
- 넓은 파란 면은 국소 두께로 제거하지만, 지시선과 파란 표면이 같은 두께로 길게 이어진 이미지는 연결 성분이 합쳐질 수 있다.
- 스캔 본체가 전체 이미지의 1%보다 작거나 흰 배경이 아닌 이미지는 본체 마스크를 안정적으로 만들기 어렵다.
- 컬러바의 방향은 세로일 때 위가 최댓값, 가로일 때 오른쪽이 최댓값이라고 가정한다.
- 저장소에 검증 샘플과 정답셋이 없어 정확도 지표는 제공하지 않는다.
