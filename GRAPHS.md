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
