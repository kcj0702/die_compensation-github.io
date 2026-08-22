# [파트 1] UI

## 역할
`data/output/result.json` 과 거기에 적힌 이미지·CSV 경로만 읽어서 화면에 표시한다.
**다른 파트의 파이썬 코드를 직접 import 하지 않는다.** 파일만 읽는다.

## 입력
| 파일 | 만드는 곳 |
|---|---|
| `data/output/result.json` | `pipeline/run_demo.py` |
| `data/intermediate/*.png`, `*.csv` | 각 파트 |

## 실행
```bash
streamlit run ui/app.py
```

## 할 일
- [ ] `ui/app.py` 작성
- [ ] result.json 로딩 및 이미지 표시
- [ ] 편차 포인트 표 표시
- [ ] 0-Line 오버레이 토글
