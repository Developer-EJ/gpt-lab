# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | SW-AI 1반 |
| 팀명 | 5 팀 |
| 팀원 | 강지현, 고명석, 김은재, 김원우 |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 담당자 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` |  |
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` |  |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` |  |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py` |  |
| 5 | loss 계산, checkpoint, generate, train_model | `src/train.py` |  |
| 6 | NSMC 감성 분류 Dataset과 classifier | `src/finetune.py` |  |

---

## 2. 테스트 통과 현황

| 실행 명령 | 결과 | 비고 |
| --- | :---: | --- |
| `pytest tests/test_bpe.py -v` | 통과 |  |
| `pytest tests/test_dataset.py -v` | 통과 |  |
| `pytest tests/test_attention.py -v` | 통과 |  |
| `pytest tests/test_model.py -v` | 통과 |  |
| `pytest tests/test_train.py -v` | 통과 |  |
| `pytest tests/test_finetune.py -v` | 통과 |  |
| `pytest tests/ -v` | 통과 | 로컬: 28 passed, 1 warning |
| `pytest tests/ -q` | 통과 | Colab CPU: 28 passed in 3.51s |

실패한 테스트가 있다면 에러 요약을 적습니다.

| 실패한 테스트 | 에러 요약 | 해결 시도 |
| --- | --- | --- |
| (예: `test_train.py::TestGenerate::test_generate_shape`) |  |  |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | Smoke (`data/nsmc_lm_train.txt` 앞 5,000자) |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | 300 |
| 학습 corpus 크기 | `data/nsmc_lm_train.txt` 앞 5,000자 |
| 어휘 학습 시간 | Colab CPU 기준 약 0.154초 |
| vocabulary 저장 경로 | Smoke 테스트에서는 저장하지 않음 |
| 인코딩/디코딩 복원 예시 | `decode(encode("이 영화는 정말 좋았다! English 123", add_bos_eos=True), skip_special=True) == 원문` |

### 4.1 BPE Smoke 테스트

| 항목 | 결과 |
| --- | --- |
| 실행 목적 | BPE 학습, encode/decode 복원, GPT 학습 배치 구성이 정상 동작하는지 빠르게 확인 |
| corpus | `data/nsmc_lm_train.txt` 앞 5,000자 |
| vocab_size | 300 |
| 실제 vocab 크기 | 300 |
| merge 수 | 40 |
| 어휘 학습 시간 | Colab CPU 기준 약 0.154초 |
| corpus token 수 | 8,780 |
| roundtrip 결과 | 통과 |
| context_length | 32 |
| batch_size | 4 |
| input/target shape | `(4, 32)` / `(4, 32)` |
| 미니 GPT 1-batch smoke loss | 5.8433 |

### 4.2 Colab 검증 기록

| 항목 | 내용 |
| --- | --- |
| Colab URL | `https://colab.research.google.com/github/Developer-EJ/gpt-lab/blob/kms/gpt-lab.ipynb` |
| 실행 브랜치 | `kms` |
| 실행 방식 | Colab 터미널에서 `kms` 브랜치 clone 후 테스트 실행 |
| 전체 테스트 | `pytest tests/ -q` -> 28 passed in 3.51s |
| 데이터 준비 | `python download_data.py` 실행, NSMC train/val/test 파일 생성 확인 |
| BPE smoke | `corpus[:5000]`, `vocab_size=300`, `context_length=32`, `batch_size=4` |
| BPE smoke 결과 | roundtrip 통과, vocab 300, merge 40, token 8,780, loss 5.8433 |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> N x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | (예: 3000) |
| context_length | (예: 128) |
| emb_dim | (예: 192) |
| n_heads | (예: 4) |
| n_layers | (예: 4) |
| drop_rate | (예: 0.1) |
| qkv_bias | True / False |
| 총 파라미터 수 | (계산식 포함) |

---

## 6. 사전 학습

### 6.1 하이퍼파라미터

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 모델 | vocab_size |  |
| 모델 | context_length |  |
| 모델 | emb_dim |  |
| 모델 | n_heads |  |
| 모델 | n_layers |  |
| 학습 | batch_size |  |
| 학습 | num_epochs |  |
| 학습 | eval_freq, eval_iter |  |
| 최적화 | lr, weight_decay |  |

### 6.2 결과

| 항목 | 내용 |
| --- | --- |
| train loss | epoch별 표 또는 요약 |
| validation loss | epoch별 표 또는 요약 |
| 손실 그래프 | 그래프 또는 파일 경로 |
| 생성 샘플 | 같은 시작 문맥으로 epoch별 비교 |
| checkpoint 경로 | (예: `checkpoints/ckpt_epoch_5.pt`) |

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | (예: 128) |
| batch_size | (예: 16) |
| backbone learning rate |  |
| classifier learning rate |  |
| validation loss / accuracy |  |
| test loss / accuracy |  |
| 오류 예시 | 틀린 리뷰 예시와 추정 원인 |

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | Colab: Python 3.12.13 |
| PyTorch | Colab: 2.11.0+cpu |
| 실행 환경 | Colab CPU / 로컬 |
| GPU/CPU 정보 | Colab Google Compute Engine CPU 런타임 |
| 총 학습 소요 시간 | BPE smoke 기준 약 0.154초 |

---

## 9. 고찰

- 어려웠던 점
- 한국어 byte-level BPE 구현에서 조심한 점
- loss가 줄어든 이유 또는 줄어들지 않은 이유
- 과적합·과소적합 여부
- 하이퍼파라미터 변경 시도와 결과
- 다음에 개선하고 싶은 점
