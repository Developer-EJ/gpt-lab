# Graph Results

이 파일은 테스트 결과 확인용 그래프를 한 곳에 모아둔 문서입니다.

## Light 100-step Train/Validation Loss

- 원본 이미지: `figures/light_train_val_loss.png`
- 결과 JSON: `figures/light_train_val_loss.json`
- 요약: train loss는 7.8385에서 7.0178로, validation loss는 7.7681에서 7.0063으로 감소했습니다.

![Light 100-step train validation loss](figures/light_train_val_loss.png)

## Light Generation Token Frequency

- 원본 이미지: `figures/light_generation_token_freq.png`
- 결과 JSON: `figures/light_generation_smoke.json`
- 요약: 100-step 학습 후 greedy generation에서 token id 267이 30회 반복 생성되었습니다.

![Light generation token frequency](figures/light_generation_token_freq.png)

## Checkpoint Smoke Train Loss

- 원본 이미지: `figures/light_checkpoint_loss.png`
- 결과 JSON: `figures/light_checkpoint_smoke.json`
- 요약: 20-step 학습 후 checkpoint 저장/복원에서 loss와 parameter가 동일하게 복원되었습니다.

![Checkpoint smoke train loss](figures/light_checkpoint_loss.png)

## Basic BPE Training Time

- 원본 이미지: `figures/basic_bpe_time.png`
- 결과 JSON: `figures/basic_bpe_time.json`
- 요약: 전체 사전 학습 train corpus 1,379,486자에서 `vocab_size=3000` BPE 학습에 약 9.276분이 걸렸습니다.

![Basic BPE training time](figures/basic_bpe_time.png)

## Basic Batch Smoke Shapes

- 원본 이미지: `figures/basic_batch_shapes.png`
- 결과 JSON: `figures/basic_batch_smoke.json`
- 요약: Basic tokenizer와 `context_length=128`, `batch_size=4` 설정에서 input/target batch shape가 모두 `(4, 128)`로 생성되었습니다.

![Basic batch smoke shapes](figures/basic_batch_shapes.png)

## Basic 50-step Train/Validation Loss

- 원본 이미지: `figures/basic_train_val_loss.png`
- 결과 JSON: `figures/basic_train_val_loss.json`
- 요약: Basic 설정에서 train loss는 8.1775에서 7.5693으로, validation loss는 8.1748에서 7.5192로 감소했습니다.

![Basic 50-step train validation loss](figures/basic_train_val_loss.png)

## Basic Generation Token Frequency

- 원본 이미지: `figures/basic_generation_token_freq.png`
- 결과 JSON: `figures/basic_generation_smoke.json`
- 요약: 50-step 학습 후 greedy generation에서 token id 272 (`는`)가 20회 반복 생성되었습니다.

![Basic generation token frequency](figures/basic_generation_token_freq.png)
