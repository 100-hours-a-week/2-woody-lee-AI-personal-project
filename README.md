# 🥦 채소 이미지 분류 - Pretrained 모델 비교 실험

**이동재**  
**2025년 3월 30일**

---

## 목차
1. [서론](#서론)
2. [데이터셋 설명](#데이터셋-설명)
3. [모델 설명](#모델-설명)
4. [실험 방법](#실험-방법)
5. [결과 및 분석](#결과-및-분석)
6. [결론](#결론)
7. [참고문헌 및 부록](#참고문헌-및-부록)

---

## 서론

본 연구의 목적은 다양한 채소 이미지 데이터셋을 활용하여 대표 채소(예: 토마토, 오이, 당근 등)의 종류를 자동으로 분류하는 CNN 기반 모델을 개발하는 것이다. 전통적인 채소 분류 방식은 인력 소요와 시간이 많이 들기 때문에, 자동화된 분류 시스템을 통해 농가나 유통업체의 품질 관리 및 생산 효율성을 크게 향상시킬 수 있다. 본 보고서에서는 CNN 모델과 사전 훈련된 모델(전이 학습 기법)을 적용하여 채소 분류 정확도를 극대화하는 방법을 제시한다.

---

## 데이터셋 설명

연구에 사용된 채소 이미지 데이터셋은 Kaggle 또는 기타 공개 데이터 소스에서 제공되는 채소 이미지 컬렉션을 기반으로 한다. 주요 특징은 다음과 같다.

- **이미지 해상도 및 형식:**  
  모든 이미지는 고해상도(예: 300×300 픽셀)의 PNG 또는 JPEG 형식으로 제공되어, 채소의 세부 특징(색상, 질감 등)을 명확하게 확인할 수 있다.
  
- **클래스 구성:**  
  데이터셋은 대표적인 채소 종류(예: 토마토, 오이, 당근 등)로 구성되어 있으며, 클래스 간 균형을 맞추어 데이터 불균형 문제를 최소화하였다.
  
- **촬영 조건:**  
  다양한 각도, 배경 및 조명 조건에서 촬영된 이미지들이 포함되어 있어, 실제 환경에서의 분류 성능을 높일 수 있다.
  
- **데이터 규모:**  
  총 15,000개의 이미지와 15개의 클래스가 포함되어 있다.

---

## 모델 설명

### ResNet18
- **구조:**  
  18개의 층을 가진 잔차 네트워크(Residual Network)
- **주요 특징:**  
  - 잔차 연결(Residual Connections): 깊은 네트워크에서 발생하는 기울기 소실 문제 완화  
  - 비교적 간단한 구조로 이미지 분류, 객체 검출 등 다양한 비전 작업에 활용

### MobileNetV2
- **구조:**  
  경량화된 모델로, 모바일 및 임베디드 디바이스에서 효율적으로 동작
- **주요 특징:**  
  - 깊이별 분리 합성곱(Depthwise Separable Convolutions): 연산량과 파라미터 수 감소  
  - 인버티드 잔차 구조(Inverted Residual Structure): 효율성 향상

### EfficientNetB0
- **구조:**  
  EfficientNet 계열의 기본 모델
- **주요 특징:**  
  - 컴파운드 스케일링(Compound Scaling): 네트워크의 깊이, 너비, 해상도를 동시에 균형 있게 확장  
  - 효율적인 연산으로 우수한 성능 제공

### GoogleNet
- **구조:**  
  Inception 모듈을 도입한 CNN 모델 (22층)
- **주요 특징:**  
  - Inception 모듈: 여러 크기의 필터를 병렬로 사용하여 다양한 스케일의 특징 추출  
  - 효율적인 계산으로 이미지 분류 및 객체 인식 분야에서 성과

### Deep CNN
- **구조:**  
  여러 개의 합성곱 계층을 깊게 쌓은 구조
- **주요 특징:**  
  - 계층적 특징 학습: 낮은 계층은 간단한 패턴, 깊은 계층은 복잡한 패턴 학습  
  - 다양한 컴퓨터 비전 작업에 활용 가능

---

## 실험 방법

### 데이터 전처리 및 증강
- **이미지 전처리:**  
  - 모든 이미지를 224×224로 리사이즈  
  - 픽셀 값 정규화 (0 ~ 1 범위)
  
- **데이터 증강:**  
  - `ImageDataGenerator`를 활용하여 회전, 이동, 확대/축소, 좌우 반전 등의 증강 적용

### 모델 학습
- **Pre-trained 모델 불러오기:**  
  각 모델(ResNet18, MobileNetV2, EfficientNetB0, GoogleNet, Deep CNN)을 사전 훈련된 가중치와 함께 불러옴
- **학습 기법:**  
  - Early Stopping 적용하여 최적 학습 시점 도출  
  - 하이퍼파라미터(배치 크기, 학습률, 에포크 수 등) 튜닝

### 모델 평가 및 분석
- **평가:**  
  테스트 데이터셋을 통해 최종 모델의 분류 정확도와 손실(Loss) 평가
- **분석:**  
  - Epoch 당 Accuracy, Loss 값 분석  
  - GoogleNet의 Confusion Matrix 및 모델별 Heatmap 시각화  
  - TensorBoard를 활용하여 Training/Validation Accuracy 모니터링  
- **추가:**  
  SQLite를 이용한 DB 연결 후, 결과를 DataFrame으로 확인

---

## 결과 및 분석

- **학습 결과:**  
  각 모델별로 Epoch 당 Accuracy와 Loss 값 변화 관찰
- **시각화 자료:**  
  - GoogleNet Confusion Matrix  
  - 모델별 Heatmap  
  - TensorBoard를 통한 Train Acc 및 Val Acc 그래프
- **추가 분석:**  
  - Deep CNN의 구조와 Loss, Train/Val/Test Accuracy 비교 분석

---

## 결론

본 연구에서는 Pre-trained 모델을 사용하여 CNN 기반 채소 이미지 분류 모델을 개발하고 비교 실험을 진행하였다.  
- 사전 훈련된 모델을 도입함으로써 전이 학습의 장점을 효과적으로 활용하였으며, 기본 모델에 비해 높은 분류 정확도와 빠른 수렴 속도를 보였다.  
- 일부 과적합 문제와 연산 비용 이슈가 존재하였으나, 이를 개선하기 위한 추가 연구가 필요하다.  
- 향후 FastAPI를 활용한 모델 배포 및 백엔드 AI 엔지니어링 역량 강화 과정을 진행할 예정이며, 이미지 업로드와 모델 학습 파트는 추후 완성하여 Git에 푸시할 계획이다.

---

## 참고문헌 및 부록

- [Kaggle - Vegetable Image Dataset 예시](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/code)
- [TensorFlow API Docs](https://www.tensorflow.org/api_docs)
- [TensorFlow Tutorials (한국어)](https://www.tensorflow.org/tutorials?hl=ko)
