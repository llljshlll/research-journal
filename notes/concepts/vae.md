# Variational Autoencoder (VAE)

Variational Autoencoder(VAE)는  
기존 AutoEncoder(AE)를 확장한 **확률적(latent-variable) 생성 모델**로,  
입력을 하나의 고정된 latent 코드가 아닌 **잠재 분포(latent distribution)** 로 표현한다.

본 문서는 AutoEncoder와 VAE의 차이,  
그리고 VAE가 연속적이고 의미 있는 latent space를 형성하는 이유를 정리한다.

---

## 1. AutoEncoder (AE)

AutoEncoder는 다음 두 모듈로 구성된다.

- **Encoder**
  - 입력 데이터를 latent vector로 압축
- **Decoder**
  - latent vector로부터 입력 데이터를 재구성

학습 목표는 단순하다.

- 입력과 출력이 최대한 동일하도록 재구성
- latent vector는 입력을 압축한 **단일한 표현(point representation)**

이 구조의 특징:
- 결정론적(deterministic) 인코딩
- latent space의 특정 지점만 의미를 가짐
- latent space의 임의 지점은 의미 없는 경우가 많음

---

## 2. Variational AutoEncoder (VAE)

VAE는 AutoEncoder 구조를 기반으로 하지만,  
latent 표현을 **단일 값이 아닌 확률 분포**로 모델링한다는 점에서 근본적인 차이를 가진다.

### 2.1 Latent Distribution

VAE의 encoder는 입력을 다음과 같이 변환한다.

- latent vector → **latent distribution**
- 각 latent 차원마다
  - 평균 (mean)
  - 분산 (variance)

즉, 입력 이미지는 하나의 점이 아니라  
**확률 분포의 파라미터**로 표현된다.

이 분포로부터 샘플링된 latent가  
decoder의 입력으로 사용된다.

---

## 3. Sampling and Reconstruction

VAE는 latent distribution에서 **무작위 샘플링**을 수행한 뒤,  
해당 샘플을 decoder에 입력하여 출력을 복원한다.

이 과정의 핵심 의미는 다음과 같다.

- latent space 전반에서 유효한 샘플 생성 가능
- latent 공간의 연속성과 부드러움 보장
- latent 상의 작은 변화가 출력의 점진적 변화로 연결

즉, latent space의 어느 지점을 샘플링하더라도  
의미 있는 출력이 생성될 것을 기대할 수 있다.

---

## 4. Continuous and Meaningful Latent Space

VAE의 확률적 인코딩은  
latent space를 다음과 같은 성질을 가진 공간으로 만든다.

- 연속적인(latent continuity) 표현
- 의미 구조가 공간 전체에 고르게 분포
- 가까운 latent는 유사한 재구성과 연결

이는 생성 모델에서 매우 중요한 성질이며,  
latent space 상에서의 보간(interpolation)이나 샘플링을 가능하게 한다.

---

## 5. Generative Models에서의 의미

VAE는 입력 데이터를  
**통계적 과정에 의해 생성된 샘플**로 가정한다.

이로 인해:
- 생성 과정에 내재된 불확실성 모델링 가능
- noise 기반 모델(diffusion 등)과 자연스럽게 결합 가능
- latent space에서의 확률적 조작 가능

이러한 특성 때문에  
VAE는 diffusion model, GAN, flow 기반 모델 등  
다양한 생성 모델의 구성 요소로 활용된다.
