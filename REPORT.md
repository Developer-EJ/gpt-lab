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
| 사용한 데이터 크기 | Smoke (`data/nsmc_lm_train.txt` 앞 5,000자), Light (`data/nsmc_lm_train.txt` 앞 500,000자) |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | Smoke 300, Light 2,000 |
| 학습 corpus 크기 | Smoke 5,000자, Light 500,000자 |
| 어휘 학습 시간 | Smoke: Colab CPU 약 0.154초, Light: 로컬 CPU 약 125.359초 |
| vocabulary 저장 경로 | Light: `data/vocab_light_2000.json` (`.gitignore` 대상) |
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

### 4.3 BPE Light 테스트

| 항목 | 결과 |
| --- | --- |
| 실행 목적 | Smoke보다 큰 코퍼스에서 BPE 학습 시간, vocab 저장, 모델 입력 연결을 확인 |
| corpus | `data/nsmc_lm_train.txt` 앞 500,000자 |
| vocab_size | 2,000 |
| 실제 vocab 크기 | 2,000 |
| merge 수 | 1,740 |
| 어휘 학습 시간 | 로컬 CPU 기준 약 125.359초 |
| vocabulary 저장 경로 | `data/vocab_light_2000.json` |
| roundtrip 결과 | 통과 |
| corpus token 수 | 326,967 |
| context_length | 64 |
| batch_size | 8 |
| input/target shape | `(8, 64)` / `(8, 64)` |
| 모델 구조 | `emb_dim=128`, `n_heads=4`, `n_layers=2`, `drop_rate=0.1` |
| 모델 파라미터 수 | 916,224 |
| 1-step 전 loss | 7.758 |
| 1-step 후 loss | 7.7248 |

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
| 모델 | vocab_size | 2,000 |
| 모델 | context_length | 64 |
| 모델 | emb_dim | 128 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 2 |
| 학습 | batch_size | 8 |
| 학습 | num_epochs | Light 100-step smoke 학습 |
| 학습 | eval_freq, eval_iter | 별도 validation 평가는 수행하지 않음 |
| 최적화 | lr, weight_decay | AdamW, lr=3e-4 |

### 6.2 결과

| 항목 | 내용 |
| --- | --- |
| train loss | 100-step 기준 7.8385 -> 7.0092 |
| validation loss | 미실행 |
| 손실 그래프 | `GRAPHS.md`, `figures/light_train_val_loss.png` |
| 생성 샘플 | Light generation smoke 기준 `"이 영화는"` prompt 사용 |
| checkpoint 경로 | `checkpoints/light_step20.pt` (`.gitignore` 대상) |

### 6.3 Light 100-step 학습 테스트

| step | train loss |
| --- | ---: |
| 1 | 7.8385 |
| 10 | 7.7068 |
| 20 | 7.6413 |
| 30 | 7.5529 |
| 40 | 7.3784 |
| 50 | 7.2707 |
| 60 | 7.1923 |
| 70 | 7.0577 |
| 80 | 7.0507 |
| 90 | 7.0164 |
| 100 | 7.0092 |

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | Light 설정에서 짧은 학습 루프가 정상 동작하고 loss가 감소하는지 확인 |
| token 수 | 326,967 |
| DataLoader batch 수 | 639 |
| 소요 시간 | 로컬 CPU 기준 약 2.425초 |
| 결과 요약 | 100 step 동안 train loss가 7.8385에서 7.0092로 감소 |

### 6.4 Light 100-step Train/Validation Loss 테스트

| step | train loss | validation loss |
| --- | ---: | ---: |
| 1 | 7.8385 | 7.7681 |
| 10 | 7.7101 | 7.7212 |
| 20 | 7.6361 | 7.6491 |
| 30 | 7.5482 | 7.5379 |
| 40 | 7.3690 | 7.3891 |
| 50 | 7.2728 | 7.2540 |
| 60 | 7.1747 | 7.1568 |
| 70 | 7.0544 | 7.0910 |
| 80 | 7.0532 | 7.0493 |
| 90 | 7.0060 | 7.0220 |
| 100 | 7.0178 | 7.0063 |

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | Light 설정에서 train loss와 validation loss가 함께 감소하는지 확인 |
| train corpus | `data/nsmc_lm_train.txt` 앞 500,000자 |
| validation corpus | `data/nsmc_lm_val.txt` 앞 120,000자 |
| train token 수 | 326,967 |
| validation token 수 | 78,479 |
| validation 평가 범위 | 매 기록 시점마다 10 batch 평균 |
| 소요 시간 | 로컬 CPU 기준 약 3.593초 |
| 결과 요약 | train loss 7.8385 -> 7.0178, validation loss 7.7681 -> 7.0063 |
| 그래프 | `GRAPHS.md`, `figures/light_train_val_loss.png` |

### 6.5 Light Generation Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | 학습 전/후 `generate()`와 tokenizer `decode()` 경로가 정상 동작하는지 확인 |
| prompt | `이 영화는` |
| 생성 방식 | greedy decoding (`temperature=0`, `max_new_tokens=30`) |
| prompt token 수 | 2 |
| 생성 token 수 | 30 |
| 학습 전 생성 샘플 | `이 영화는 중��같은에도 기대 나온� 그저이 영화에;;` ... |
| 100-step 학습 후 생성 샘플 | `이 영화는이이이이이이이이이이이이이이이이이이이이이이이이이이이이이이` |
| 생성 token 다양성 | unique token 1개 |
| 최빈 생성 token | token id 267, 30회 |
| 결과 해석 | 100-step만 학습한 모델은 생성 경로는 동작하지만 greedy 생성에서 같은 token을 반복하는 퇴화 현상이 나타남 |
| 그래프 | `GRAPHS.md`, `figures/light_generation_token_freq.png` |

### 6.6 Checkpoint Save/Load Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | `save_checkpoint()`와 `load_checkpoint()`가 model/optimizer/epoch/global_step을 정상 복원하는지 확인 |
| checkpoint 경로 | `checkpoints/light_step20.pt` |
| 학습 step | 20 |
| 복원 epoch | 0 |
| 복원 global_step | 20 |
| 저장 직전 loss | 7.626362 |
| 복원 후 loss | 7.626362 |
| loss 차이 | 0.0 |
| 최대 parameter 차이 | 0.0 |
| 소요 시간 | 로컬 CPU 기준 약 0.665초 |
| 그래프 | `GRAPHS.md`, `figures/light_checkpoint_loss.png` |

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
