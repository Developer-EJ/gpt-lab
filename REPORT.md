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

## 2.1 테스트 진행률

| 구분 | 진행률 | 완료 내용 | 남은 내용 |
| --- | ---: | --- | --- |
| 단위 테스트 | 100% | 로컬/Colab pytest 28개 통과 | 최종 제출 전 재실행 |
| BPE 검증 | 100% | Smoke, Light, Basic 시간 측정, batch smoke | 없음 |
| 사전 학습 smoke | 90% | Light/Basic loss, generation, sampling, checkpoint, resume, token cache | 더 긴 Colab 학습 선택 사항 |
| 미세 조정 smoke | 80% | classifier smoke, mini train/validation curve, mini test confusion matrix, majority baseline 비교 | full finetune Colab 학습, 전체 test 평가 |
| 보고서/그래프 | 88% | `REPORT.md`, `GRAPHS.md`, 실험별 JSON/PNG 기록 | 최종 요약/고찰 정리 |
| 전체 진행률 | 약 84% | 로컬 중심 smoke/mini 검증 대부분 완료 | 긴 finetune Colab 검증과 최종 재검증 |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | Smoke (`data/nsmc_lm_train.txt` 앞 5,000자), Light (`data/nsmc_lm_train.txt` 앞 500,000자), Basic (`data/nsmc_lm_train.txt` 전체 1,379,486자) |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | Smoke 300, Light 2,000, Basic 3,000 |
| 학습 corpus 크기 | Smoke 5,000자, Light 500,000자, Basic 1,379,486자 |
| 어휘 학습 시간 | Smoke: Colab CPU 약 0.154초, Light: 로컬 CPU 약 125.359초, Basic: 로컬 CPU 약 556.574초 |
| vocabulary 저장 경로 | Light: `data/vocab_light_2000.json`, Basic: `data/vocab_basic_3000.json` (`.gitignore` 대상) |
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

### 4.4 BPE Basic 시간 측정

| 항목 | 결과 |
| --- | --- |
| 실행 목적 | Basic 설정의 BPE 학습 시간과 vocab 저장 가능 여부 확인 |
| corpus | `data/nsmc_lm_train.txt` 전체 1,379,486자 |
| vocab_size | 3,000 |
| 실제 vocab 크기 | 3,000 |
| merge 수 | 2,740 |
| 어휘 학습 시간 | 로컬 CPU 기준 약 556.574초, 약 9.276분 |
| vocabulary 저장 경로 | `data/vocab_basic_3000.json` |
| roundtrip 결과 | 통과 |
| sample token 수 | 18 |
| corpus token 수 | 805,023 |
| 결과 해석 | Basic 설정에서는 BPE 학습 시간이 길기 때문에 학습한 vocabulary를 저장해 재사용하는 것이 필요함 |
| 그래프 | `GRAPHS.md`, `figures/basic_bpe_time.png` |

### 4.5 Basic 한 배치 Smoke 테스트

| 항목 | 결과 |
| --- | --- |
| 실행 목적 | Basic vocabulary가 Dataset/DataLoader/GPT forward까지 연결되는지 확인 |
| corpus | `data/nsmc_lm_train.txt` 전체 1,379,486자 |
| vocab_size | 3,000 |
| roundtrip 결과 | 통과 |
| corpus token 수 | 805,023 |
| context_length | 128 |
| batch_size | 4 |
| DataLoader batch 수 | 1,573 |
| input/target shape | `(4, 128)` / `(4, 128)` |
| 모델 구조 | `emb_dim=192`, `n_heads=4`, `n_layers=2`, `drop_rate=0.1` |
| 모델 파라미터 수 | 2,065,536 |
| 1-batch loss | 8.1552 |
| 실행 시간 | 전체 corpus encode 포함 로컬 CPU 기준 약 257.297초 |
| 결과 해석 | Basic 설정에서도 batch 구성과 forward/loss 계산이 정상 동작함. 다만 전체 corpus encode 비용도 커서 token ID 캐싱을 고려할 필요가 있음 |
| 그래프 | `GRAPHS.md`, `figures/basic_batch_shapes.png` |

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
| train loss | Light 100-step 기준 7.8385 -> 7.0092, Basic cached 100-step 기준 8.1775 -> 7.3897 |
| validation loss | Light 100-step 기준 7.7681 -> 7.0063, Basic cached 100-step 기준 8.1748 -> 7.3425 |
| 손실 그래프 | `GRAPHS.md`, `figures/light_train_val_loss.png`, `figures/basic_train_val_loss.png`, `figures/basic_cached_100step_loss.png` |
| 생성 샘플 | Light/Basic generation smoke 기준 `"이 영화는"` prompt 사용 |
| checkpoint 경로 | `checkpoints/light_step20.pt`, `checkpoints/basic_step50.pt` (`.gitignore` 대상) |

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

### 6.7 Basic 50-step Train/Validation Loss 테스트

| step | train loss | validation loss |
| --- | ---: | ---: |
| 1 | 8.1775 | 8.1748 |
| 10 | 8.1324 | 8.1223 |
| 20 | 8.0404 | 8.0285 |
| 30 | 7.8272 | 7.8575 |
| 40 | 7.6451 | 7.6538 |
| 50 | 7.5693 | 7.5192 |

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | Basic 설정에서 짧은 학습 루프가 train/validation loss를 함께 낮추는지 확인 |
| train corpus | `data/nsmc_lm_train.txt` 전체 1,379,486자 |
| validation corpus | `data/nsmc_lm_val.txt` 전체 120,560자 |
| train token 수 | 805,023 |
| validation token 수 | 70,388 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| 모델 구조 | `emb_dim=192`, `n_heads=4`, `n_layers=2`, `drop_rate=0.1` |
| validation 평가 범위 | 매 기록 시점마다 5 batch 평균 |
| corpus encode 시간 | 로컬 CPU 기준 약 296.009초 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 2.175초 |
| 결과 요약 | train loss 8.1775 -> 7.5693, validation loss 8.1748 -> 7.5192 |
| 그래프 | `GRAPHS.md`, `figures/basic_train_val_loss.png` |

### 6.8 Basic Generation Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | Basic 50-step 학습 후 `generate()`와 tokenizer `decode()` 경로가 정상 동작하는지 확인 |
| prompt | `이 영화는` |
| 생성 방식 | greedy decoding (`temperature=0`, `max_new_tokens=30`) |
| prompt token 수 | 2 |
| 생성 token 수 | 30 |
| train corpus | `data/nsmc_lm_train.txt` 전체 1,379,486자 |
| train token 수 | 805,023 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| corpus encode 시간 | 로컬 CPU 기준 약 263.694초 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 1.734초 |
| 50-step train loss | 8.1775 -> 7.5660 |
| 학습 전 생성 샘플 | `이 영화는 이해 고공 별 역시 나오 야 사�구. 그 해서 OO 가슴받더니 혼잡�때언립 철 한공포 나온보는 현실뭔틱 따` |
| 50-step 학습 후 생성 샘플 | `이 영화는는는는는는는을는는는이\n\n는는는이을는을는이는는을는는는는이` |
| 생성 token 다양성 | unique token 4개 |
| 최빈 생성 token | token id 272 (`는`), 20회 |
| 결과 해석 | 50-step만 학습한 Basic 모델은 생성 경로는 동작하지만 greedy 생성에서 조사 token 중심의 반복 현상이 나타남 |
| 그래프 | `GRAPHS.md`, `figures/basic_generation_token_freq.png` |

### 6.9 Basic Checkpoint Save/Load Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | Basic 50-step 학습 후 checkpoint 저장/복원이 model/optimizer/epoch/global_step을 정상 복원하는지 확인 |
| train corpus | `data/nsmc_lm_train.txt` 전체 1,379,486자 |
| train token 수 | 805,023 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| checkpoint 경로 | `checkpoints/basic_step50.pt` |
| 학습 step | 50 |
| 복원 epoch | 0 |
| 복원 global_step | 50 |
| 저장 직전 loss | 7.538605 |
| 복원 후 loss | 7.538605 |
| loss 차이 | 0.0 |
| 최대 parameter 차이 | 0.0 |
| corpus encode 시간 | 로컬 CPU 기준 약 258.934초 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 2.000초 |
| 결과 해석 | Basic checkpoint가 동일 loss와 동일 parameter로 복원됨 |
| 그래프 | `GRAPHS.md`, `figures/basic_checkpoint_loss.png` |

### 6.10 Basic Token ID Cache 성능 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | Basic 실험에서 반복적으로 발생하는 전체 corpus BPE encode 비용을 token ID cache로 줄일 수 있는지 확인 |
| train corpus | `data/nsmc_lm_train.txt` 전체 1,379,486자 |
| validation corpus | `data/nsmc_lm_val.txt` 전체 120,560자 |
| train token 수 | 805,023 |
| validation token 수 | 70,388 |
| vocab_size | 3,000 |
| cache 경로 | `data/basic_train_ids.pt`, `data/basic_val_ids.pt` (`.gitignore` 대상) |
| cache 파일 크기 | train 약 3.2MB, validation 약 0.28MB |
| BPE encode 시간 | 로컬 CPU 기준 약 280.018초 |
| cache 저장 시간 | 약 0.040초 |
| cache load 시간 | 3회 평균 약 0.011초 |
| encode 대비 load 속도 | 약 24,354.72배 빠름 |
| token ID 일치 여부 | 통과 |
| DataLoader 설정 | `context_length=128`, `batch_size=4`, `stride=128` |
| DataLoader batch 수 | train 1,573, validation 138 |
| input/target shape | `(4, 128)` / `(4, 128)` |
| DataLoader 준비 시간 | 약 0.009초 |
| 결과 해석 | Basic 반복 실험에서는 BPE encode를 매번 수행하지 말고 token ID cache를 재사용하는 것이 필요함 |
| 그래프 | `GRAPHS.md`, `figures/basic_token_cache_perf.png` |

### 6.11 Cached Basic 100-step Train/Validation Loss 테스트

| step | train loss | validation loss |
| --- | ---: | ---: |
| 1 | 8.1775 | 8.1748 |
| 10 | 8.1324 | 8.1223 |
| 20 | 8.0404 | 8.0285 |
| 30 | 7.8272 | 7.8575 |
| 40 | 7.6451 | 7.6538 |
| 50 | 7.5693 | 7.5192 |
| 60 | 7.4051 | 7.4374 |
| 70 | 7.2755 | 7.3899 |
| 80 | 7.3511 | 7.3631 |
| 90 | 7.3978 | 7.3501 |
| 100 | 7.3897 | 7.3425 |

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | token ID cache를 사용해 Basic 설정에서 100-step train/validation loss 감소를 빠르게 확인 |
| cache 경로 | `data/basic_train_ids.pt`, `data/basic_val_ids.pt` (`.gitignore` 대상) |
| cache load 시간 | 약 0.016초 |
| train token 수 | 805,023 |
| validation token 수 | 70,388 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| DataLoader batch 수 | train 1,573, validation 138 |
| validation 평가 범위 | 매 기록 시점마다 5 batch 평균 |
| 모델 구조 | `emb_dim=192`, `n_heads=4`, `n_layers=2`, `drop_rate=0.1` |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 3.881초 |
| 결과 요약 | train loss 8.1775 -> 7.3897, validation loss 8.1748 -> 7.3425 |
| 결과 해석 | cache 사용으로 준비 시간이 크게 줄었고, 100-step까지 validation loss가 계속 완만하게 감소함 |
| 그래프 | `GRAPHS.md`, `figures/basic_cached_100step_loss.png` |

### 6.12 Cached Basic 100-step Generation Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | token ID cache를 사용한 Basic 100-step 학습 후 `generate()`와 tokenizer `decode()` 경로 및 생성 반복 양상을 확인 |
| prompt | `이 영화는` |
| 생성 방식 | greedy decoding (`temperature=0`, `max_new_tokens=30`) |
| prompt token 수 | 2 |
| 생성 token 수 | 30 |
| cache 경로 | `data/basic_train_ids.pt` (`.gitignore` 대상) |
| cache load 시간 | 약 0.014초 |
| train token 수 | 805,023 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| DataLoader batch 수 | 1,573 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 3.579초 |
| 100-step train loss | 8.1775 -> 7.4030 |
| 학습 전 생성 샘플 | `이 영화는 이해 고공 별 역시 나오 야 사�구. 그 해서 OO 가슴받더니 혼잡�때언립 철 한공포 나온보는 현실뭔틱 따` |
| 100-step 학습 후 생성 샘플 | `이 영화는이이이이이이이이이이이이이이이이이이이이이이이이이이이이이이` |
| 생성 token 다양성 | unique token 1개 |
| 최빈 생성 token | token id 267 (`이`), 30회 |
| 결과 해석 | 100-step 학습 후에도 생성 경로는 동작하지만 greedy 생성은 단일 token 반복으로 퇴화함 |
| 그래프 | `GRAPHS.md`, `figures/basic_cached_100step_generation_token_freq.png` |

### 6.13 Cached Basic 100-step Sampling Generation Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | greedy decoding 반복 문제를 sampling 설정으로 완화할 수 있는지 확인 |
| prompt | `이 영화는` |
| 비교 생성 방식 | greedy (`temperature=0`) vs sampling (`temperature=0.8`, `top_k=40`, seed 123) |
| 생성 token 수 | 각 30 |
| cache 경로 | `data/basic_train_ids.pt` (`.gitignore` 대상) |
| cache load 시간 | 약 0.015초 |
| train token 수 | 805,023 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 3.514초 |
| 100-step train loss | 8.1775 -> 7.4030 |
| greedy 생성 샘플 | `이 영화는이이이이이이이이이이이이이이이이이이이이이이이이이이이이이이` |
| sampling 생성 샘플 | `이 영화는하한가고을고을지고의이.에.리기을나나로는자고는다\n라리만기리` |
| greedy token 다양성 | unique token 1개 |
| sampling token 다양성 | unique token 19개 |
| greedy 최빈 token | token id 267 (`이`), 30회 |
| sampling 최빈 token | token id 277 (`고`), 4회 |
| 결과 해석 | sampling을 적용하면 문장 품질은 아직 낮지만 단일 token 반복은 크게 줄고 token 다양성이 증가함 |
| 그래프 | `GRAPHS.md`, `figures/basic_cached_100step_sampling_token_freq.png` |

### 6.14 Basic Checkpoint Resume 학습 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | 100-step checkpoint 저장 후 로드해서 120-step까지 이어 학습되는지 확인 |
| 비교 방식 | 120-step 연속 학습 vs 100-step 저장/복원 후 20-step 추가 학습 |
| cache 경로 | `data/basic_train_ids.pt`, `data/basic_val_ids.pt` (`.gitignore` 대상) |
| checkpoint 경로 | `checkpoints/basic_resume_step100.pt` (`.gitignore` 대상) |
| cache load 시간 | 약 0.014초 |
| train token 수 | 805,023 |
| validation token 수 | 70,388 |
| vocab_size | 3,000 |
| context_length | 128 |
| batch_size | 4 |
| DataLoader shuffle | `False` |
| 복원 epoch | 0 |
| 복원 global_step | 100 |
| 연속 학습 120-step train loss | 7.2706 |
| resume 학습 120-step train loss | 7.2711 |
| 연속 학습 validation loss | 7.325830 |
| resume 학습 validation loss | 7.326025 |
| 120-step fixed batch loss 차이 | 약 0.003947 |
| 최대 parameter 차이 | 약 0.026437 |
| 결과 해석 | checkpoint에서 model/optimizer/global_step은 복원되어 이어 학습된다. 다만 dropout RNG 상태를 checkpoint에 저장하지 않아 연속 학습과 bit-exact하게 같지는 않다 |
| 그래프 | `GRAPHS.md`, `figures/basic_checkpoint_resume_loss.png` |

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | 128 |
| batch_size | Smoke 기준 8 |
| backbone learning rate | Smoke 기준 AdamW lr=3e-4 |
| classifier learning rate | Smoke 기준 AdamW lr=3e-4 |
| validation loss / accuracy | Smoke 기준 0.760974 / 0.539062 |
| test loss / accuracy | 본 학습 후 측정 예정 |
| 오류 예시 | 본 학습 후 틀린 리뷰 예시 분석 예정 |

### 7.1 Finetune Sentiment Classifier Smoke 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | 감성 분류 Dataset, DataLoader, GPT backbone, classification head, loss/accuracy 계산 경로 확인 |
| train sample 수 | 128 |
| validation sample 수 | 128 |
| vocab_size | 3,000 |
| max_length | 128 |
| batch_size | 8 |
| input shape | `(8, 128)` |
| label shape | `(8)` |
| logits shape | `(8, 2)` |
| 모델 구조 | `emb_dim=192`, `n_heads=4`, `n_layers=2`, `drop_rate=0.1`, classifier 2-class |
| optimizer | AdamW, lr=3e-4 |
| batch loss 전 | 0.772232 |
| batch loss 1-step 후 | 0.657083 |
| batch accuracy 전/후 | 0.625 / 0.625 |
| validation loss 1-step 후 | 0.760974 |
| validation accuracy 1-step 후 | 0.539062 |
| 결과 해석 | 분류용 Dataset padding, 마지막 유효 token pooling, classifier forward/loss/update/evaluate 경로가 정상 동작함 |
| 그래프 | `GRAPHS.md`, `figures/finetune_classifier_smoke_loss.png` |

### 7.2 Finetune Mini Train/Validation Curve 테스트

| epoch | train loss | train accuracy | validation loss | validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0 | - | - | 0.7513 | 0.4648 |
| 1 | 0.7590 | 0.5078 | 0.7132 | 0.4883 |
| 2 | 0.6860 | 0.5625 | 0.7254 | 0.5312 |
| 3 | 0.6432 | 0.6318 | 0.8946 | 0.5117 |

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | smoke보다 큰 subset에서 finetune train/validation loss와 accuracy 곡선을 확인 |
| train sample 수 | 1,024 |
| validation sample 수 | 256 |
| vocab_size | 3,000 |
| max_length | 128 |
| batch_size | 16 |
| epoch 수 | 3 |
| train batch 수 | 64 |
| validation batch 수 | 16 |
| 모델 구조 | `emb_dim=192`, `n_heads=4`, `n_layers=2`, `drop_rate=0.1`, classifier 2-class |
| optimizer | AdamW, lr=3e-4 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 37.834초 |
| 결과 요약 | train loss 0.7590 -> 0.6432, validation accuracy 0.4648 -> 0.5117 |
| 결과 해석 | train loss와 train accuracy는 개선됐지만 validation loss가 3 epoch에서 상승해 작은 subset 기준 과적합 신호가 나타남 |
| 그래프 | `GRAPHS.md`, `figures/finetune_mini_curve.png` |

### 7.3 Finetune Mini Test Confusion Matrix 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | finetune mini 모델을 test subset에서 평가하고 예측 편향을 confusion matrix로 확인 |
| train sample 수 | 1,024 |
| validation sample 수 | 256 |
| test sample 수 | 256 |
| vocab_size | 3,000 |
| max_length | 128 |
| batch_size | 16 |
| epoch 수 | 3 |
| test loss | 0.864738 |
| test accuracy | 0.531250 |
| confusion matrix | `[[4, 117], [3, 132]]` |
| negative class accuracy | 0.033058 |
| positive class accuracy | 0.977778 |
| 학습/평가 소요 시간 | 로컬 CPU 기준 약 40.658초 |
| 결과 해석 | 전체 accuracy는 0.531250이지만 positive class로 치우친 예측이 강하게 나타남. full finetune에서는 class balance와 validation 기준 early stopping이 필요함 |
| 그래프 | `GRAPHS.md`, `figures/finetune_mini_test_confusion.png` |

### 7.4 Finetune Mini Majority Baseline 비교 테스트

| 항목 | 내용 |
| --- | --- |
| 실행 목적 | finetune mini test accuracy가 단순 majority baseline보다 의미 있게 높은지 확인 |
| test sample 수 | 256 |
| test label 분포 | negative 121개, positive 135개 |
| all-negative baseline accuracy | 0.472656 |
| all-positive baseline accuracy | 0.527344 |
| mini finetune model accuracy | 0.531250 |
| mini model - all-positive 차이 | 0.003906 |
| mini model confusion matrix | `[[4, 117], [3, 132]]` |
| 결과 해석 | mini model은 all-positive baseline보다 0.003906만 높아 실질적인 성능 개선은 거의 없음. full finetune에서는 validation 기준 checkpoint 선택과 편향 완화가 필요함 |
| 그래프 | `GRAPHS.md`, `figures/finetune_mini_baseline_compare.png` |

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
