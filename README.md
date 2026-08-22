# 금형 보정시트 자동 생성 — 데모

아진산업 기업 프로젝트 · KDT 3팀

3D 스캔 편차 이미지에서 보정시트 작성에 필요한 정보를 자동으로 뽑아낸다.
**보정치 산출 직전 단계까지**가 이번 데모의 범위다 (8/21 회의 결정).

---

## 지금 상태

| # | 파트 | 상태 | 폴더 |
|---|---|---|---|
| 1 | UI | 미착수 | [`ui/`](ui/) |
| 2 | **0-Line 영역 검출** | **완료** | [`zero_line_detection/`](zero_line_detection/) |
| 3 | 편차값·좌표 추출 | 미착수 | [`deviation_extraction/`](deviation_extraction/) |
| 4 | 라벨 제거 이미지 | 미착수 | [`label_removal/`](label_removal/) |
| 5 | 깊이 측정 | 별도 탐색 과제 | [`depth_measurement/`](depth_measurement/) |

각 파트는 **서로의 코드를 import 하지 않는다.** `data/intermediate/` 의 파일로만 연결한다.
규격은 [`docs/input-output-contract.md`](docs/input-output-contract.md) 참고.

---

## 빠른 실행

```bash
pip install -r requirements.txt
```

회사 데이터 없이도 바로 돌려볼 수 있다. 합성 샘플을 만들어 검출하고 정확도까지 측정한다.

```bash
python -m zero_line_detection.make_sample --evaluate
```

```bash
python pipeline/run_demo.py --input data/sample/sample_deviation_map.png
```

결과는 `data/intermediate/` 와 `data/output/result.json` 에 생성된다.

---

## 파트 2 — 0-Line 영역 검출

3D 스캔 히트맵에서 편차가 0 근처인 영역을 마스크로 뽑는다.

핵심은 **이미지 안의 컬러바를 매번 읽어 색→편차값 대응표를 새로 만드는 것**이다.
실제 스캔 3장을 보니 컬러바가 좌/우로 다르고, 이미지가 180° 회전된 것도 있고,
값 범위도 ±3.0 / ±2.0 / −1.5~+2.0(잘림) 으로 제각각이었다.
"초록색 = 0" 같은 고정 기준으로는 한 장도 제대로 처리되지 않는다.

합성 정답 대비 **IoU 0.914 / 정밀도 0.998** 이다.

출력은 두 가지로 나뉜다. **부호가 바뀌는 경계선**은 임계값을 쓰지 않으므로 누가 계산해도
같은 결과가 나오고, **0 영역(면)**은 허용오차가 필요해 그 값에 따라 8.8%~68.6% 까지
달라진다. 후자는 현장이 정할 문제라 민감도 표를 함께 낸다.

자세한 내용은 [`zero_line_detection/README.md`](zero_line_detection/README.md).

---

## 폴더 구조

```
├─ ui/                     [1] 화면 및 결과 시각화
├─ zero_line_detection/    [2] 0-Line 영역 검출          ← 완료
├─ deviation_extraction/   [3] 편차값·좌표 추출
├─ label_removal/          [4] 라벨 제거 이미지 생성
├─ depth_measurement/      [5] 깊이 측정 및 형상 특징
│
├─ shared/                 공통 코드
│  ├─ schemas/             파트 간 데이터 구조
│  ├─ constants/           경로·파일명 상수
│  └─ utils/               이미지 IO, 로깅, JSON
│
├─ data/
│  ├─ raw/                 원본 스캔 (Git 제외)
│  ├─ sample/              데모용 합성 샘플
│  ├─ intermediate/        파트 간 연결 파일
│  └─ output/              최종 결과
│
├─ pipeline/run_demo.py    전체 데모 실행
└─ docs/
   └─ input-output-contract.md
```

---

## 데이터 취급 주의

**이 저장소는 공개(Public) 상태다.**

`data/raw/` 와 `data/intermediate/` 의 이미지는 `.gitignore` 로 제외되어 있다.
아진산업 3D 스캔 이미지와 보정시트는 생산 데이터이므로 **커밋하지 않는다.**
부품번호가 가려져 있어도 마찬가지다.

저장소를 받은 사람은 `make_sample.py` 가 만드는 합성 이미지로 실행하면 된다.
실제 데이터가 필요하면 팀 내부에서 따로 전달받는다.

---

## 개발 규칙

- 좌표는 **원본 이미지 픽셀 기준**으로만 주고받는다. 리사이즈·회전한 좌표를 내보내지 않는다.
- 새 파일명이 필요하면 `shared/constants/__init__.py` 에 추가한다. 경로 하드코딩 금지.
- 한글 경로에서 `cv2.imread` 는 실패한다. `shared.utils.read_rgb` / `imwrite` 를 쓴다.
- 각 파트 폴더의 `README.md` 에 입출력과 할 일을 적어 둔다.

---

## 일정

| 시점 | 내용 |
|---|---|
| 일요일 18:00 | 각자 파트 결과물 제출 → 취합 |
| 월요일 | 파트별 개발 완료 |
| 이후 | UI 통합 및 엔진 연결 |
