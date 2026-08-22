# Mold Correction Demo

편차 맵에 표시된 숫자 라벨과 지시 좌표를 구조화된 포인트 데이터로 변환하는 실험 코드다.
현재 실행 가능한 범위는 `deviation_extraction`이며, 나머지 디렉터리는 후속 단계를 위한 골격이다.

## 현재 구현 범위

| 경로 | 상태 | 역할 |
|---|---|---|
| `deviation_extraction/` | 구현 | 라벨 검출, 좌표 산정, 편차값 판독, CSV 저장 |
| `depth_measurement/` | 골격 | 깊이 측정 단계 예정 |
| `zero_line_detection/` | 골격 | 제로 라인 검출 단계 예정 |
| `label_removal/` | 골격 | 라벨 제거 단계 예정 |
| `pipeline/`, `ui/`, `shared/` | 골격 | 통합 실행, UI, 공통 코드 예정 |
| `docs/` | 부분 구현 | 편차 추출 단계의 입출력 계약 |

## 처리 흐름

```text
편차 맵 → 라벨 박스 검출 → 리더 선분 기반 좌표 산정 → VLM 숫자 판독 → CSV·디버그 이미지
                                                       └→ 선택: 제로 라인·컬러맵 확인
```

검출 결과는 이미지 좌표계의 2차원 픽셀 좌표다. 부품 좌표계나 3차원 좌표로 변환하지 않는다.

## 실행 환경

- Python 3.10 이상
- 최초 모델 로드 시 Hugging Face 모델을 받을 수 있는 환경 또는 준비된 로컬 캐시
- CUDA 사용 가능 시 FP16, 그 외에는 CPU FP32로 추론

프로젝트 루트에서 의존성을 설치한다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r deviation_extraction/requirements.txt
```

## 실행

입력 이미지를 직접 지정하는 방식이 가장 명확하다.

```powershell
.venv\Scripts\python.exe deviation_extraction/run.py `
  --image path/to/deviation_map.png `
  --out data/intermediate/deviation_points.csv `
  --debug `
  --debug-out data/intermediate/deviation_points_debug.png
```

`--image`를 생략하면 `data/intermediate/deviation_map.png`를 사용한다. 저장소에는 예제 이미지와
모델 가중치가 포함되어 있지 않다.

기본 출력은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `data/intermediate/deviation_points.csv` | 좌표, 편차값, 검출 상태 |
| `data/intermediate/deviation_points_debug.png` | 검출 좌표와 값을 겹쳐 그린 확인용 이미지 |

세부 알고리즘, 컬러바 보정, 출력 스키마는
[`deviation_extraction/README.md`](deviation_extraction/README.md)에 정리되어 있다.
실행 전 확인 사항은 [`docs/first-demo-checklist.md`](docs/first-demo-checklist.md)를 따른다.

합성 이미지 회귀 테스트는 회사 원본이나 모델을 사용하지 않는다.

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
```

## 현재 제약

- 검출 임계값이 픽셀 크기와 색상에 고정되어 있어 입력 해상도와 스캔 조건에 민감하다.
- 꺾이거나 교차하는 리더라인은 단일 Hough 선분만으로 정확히 연결하기 어렵다.
- 저장소에 검증 이미지와 정답 데이터가 없어 검출 정확도는 아직 계량되지 않았다.
- `pipeline/run_demo.py`를 포함한 전체 보정 파이프라인은 아직 구현되지 않았다.
