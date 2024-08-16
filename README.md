## 프로젝트 개요
Attention 기법의 적용은 피부 병변에 효과적이라는 가정을 설정하여 Attention기법이 적용된 모델과 적용되지 않은 모델을 비교하여 실험을 진행하였습니다.
Attention 기법이 적용되지 않은 모델에 해당 기법을 적용하여 결과를 도출하였습니다.

## SE-block
<img width="955" alt="스크린샷 2024-08-16 오후 1 00 42" src="https://github.com/user-attachments/assets/3cdbdb8d-bba8-4fff-a3b6-9415bba5f3f4"></br>
SE Block은 채널 간의 상호 의존성을 강조하고 강조된 특징을 통합하여 모델의 성능을 향상시킵니다.
채널에 대한 평균값 Fully connected layer를 통과합니다. 
그 가중치를 원래 입력 피처맵에 곱하여 Attention이 반영된 피처맵을 생성합니다.

## 가설
Attention 기법의 적용은 피부 병변에 효과적이라는 가정을 설정하여 Attention기법이 적용된 모델과 적용되지 않은 모델을 비교하여 실험을 진행하였습니다.
<img width="917" alt="스크린샷 2024-08-16 오후 1 02 15" src="https://github.com/user-attachments/assets/2f6370ce-bed0-46fa-ab5f-28f7f43a9547"></br>

## 데이터 선정
<img width="883" alt="스크린샷 2024-08-16 오후 1 05 16" src="https://github.com/user-attachments/assets/5359a4f4-4a1e-40ed-b11f-27c01207ad26"></br>
<img width="879" alt="스크린샷 2024-08-16 오후 1 05 34" src="https://github.com/user-attachments/assets/8730f687-b75d-4420-8bd2-f6659960698d"></br>

## 데이터 전처리
<img width="887" alt="스크린샷 2024-08-16 오후 1 07 51" src="https://github.com/user-attachments/assets/de680bc4-978f-465f-83b8-05b887f43a1c"></br>

## 데이터 학습
<img width="1613" alt="스크린샷 2024-08-16 오후 1 09 58" src="https://github.com/user-attachments/assets/b91c2d3b-5606-4ee5-b28f-c3892c7ab4a4"></br>

## 가설검증
Attention 기법이 적용된 모델과 적용 안된 모델을 비교함으로써 적용된 모델이 상위를 유지하여 “Attention기법은 피부 병변에 효과적이다”라는 결과를 도출했습니다.
<img width="914" alt="스크린샷 2024-08-16 오후 1 06 58" src="https://github.com/user-attachments/assets/02dae93d-c509-47eb-82f6-fccb7ca90652"></br>

## SE-block을 추가한 DenseNet121
<img width="1721" alt="스크린샷 2024-08-16 오후 1 08 53" src="https://github.com/user-attachments/assets/0e1f9df1-4f15-4d64-b91b-230a39aea596"></br>

## Test 결과
<img width="1557" alt="스크린샷 2024-08-16 오후 1 11 05" src="https://github.com/user-attachments/assets/4ab8d578-b7ca-4e1a-a685-b287458ff89c"></br>
Test Data는 각 클래스 별 15장의 이미지 데이터로 구성되어 총 105장의 데이터로 진행하였습니다. 하지만 Validation 단계에 비해 성능이 안 좋게 나왔습니다. 해당 원인을 찾기 위해 Confusion Matrix를 생성해보았을 때
<img width="1560" alt="스크린샷 2024-08-16 오후 1 12 30" src="https://github.com/user-attachments/assets/fdcbee73-57ca-46ab-850a-35ffc803cd43"></br>
광선각화증(akiec)과 피부섬유종(df)를 탐지를 잘 못하는 것으로 보입니다. Validation에 비해 Test 과정에서의 성능이 현저히 낮았는데, 이는 데이터가 과적합 되었을 확률이 높습니다. 그러므로 향후 연구 계획으로 과적합을 줄이고 모델의 계층을 강화하여 우수한 성능을 도출 하도록 하겠습니다. 

## 결론 및 연구방향
<img width="1623" alt="스크린샷 2024-08-16 오후 1 13 09" src="https://github.com/user-attachments/assets/684ef67b-1f0e-4942-95cd-500b38025c5c">

