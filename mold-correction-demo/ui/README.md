# AJIN Die Insight UI

금형생산팀 멘토링용 통합 UI 데모입니다. `deviation_extraction`,
`label_removal`, `zero_line_detection` 결과를 한 화면에서 설명하고,
가상의 보정 계수를 적용한 작업 시트를 미리 볼 수 있습니다.

## 실행

Node.js 22 이상과 pnpm이 필요합니다.

Windows에서는 `run-ui.cmd`를 더블클릭하는 것이 가장 간단합니다.
종료할 때는 같은 폴더의 `stop-ui.cmd`를 더블클릭하면 UI와 로컬 엔진 서버가
함께 종료됩니다.

```bash
pnpm install
pnpm dev
```

브라우저에서 `http://127.0.0.1:3000`을 엽니다.

`run-ui.cmd`는 화면과 함께 로컬 Python 엔진 서버도 자동으로 실행합니다. 업로드한 이미지는
`label_removal`, `deviation_extraction`, `zero_line_detection` 모듈에서 이 PC 안에서 처리됩니다.
편차 라벨 숫자는 로컬에 설치된 `Qwen/Qwen2.5-VL-3B-Instruct`를 CUDA 8비트 모드로 판독하며,
여러 라벨을 GPU 배치로 처리하고 로드된 모델을 다음 이미지에도 재사용합니다.
품번별 폴더 화면은 지정된 기준 폴더가 실제로 존재할 때만 표시되며, 폴더 내용은 열 때마다
실시간으로 다시 읽습니다.

## 현재 구현 범위

- 여러 스캔 이미지 선택 및 드래그 앤 드롭
- 품번별 분석 진행 상태와 엔진별 결과 화면
- 편차 × 보정 계수 × `-1` 방식의 가상 보정치 실시간 계산
- 복원 이미지, 보정 포인트, 선정 제로라인 합성 미리보기
- 실제 품번 폴더 예시를 반영한 Explorer형 폴더 탐색
- 데스크톱, 태블릿, 모바일 반응형 레이아웃

현재 엔진 수치와 분석 진행은 멘토링용 데모 데이터입니다. 실제 서비스에서는
Python 엔진의 산출물(JSON, CSV, 이미지)을 같은 화면 모델에 연결하면 됩니다.
