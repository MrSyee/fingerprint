# 위조지문 판별을 위한 ANN과 CNN

## Introduction
- 위조 지문 판별을 위한 ANN 모델과 CNN 모델을 다양하게 실험함.
- 모델의 성능은 유지하면서 모델의 크기를 줄이는 방향으로 실험함.

## Prerequisites
- python 3.6
- pytorch 0.4.1

## Run the example
`python main.py --model ConvNet --mode train`

## File Description
**main**
- 실행 파일. train 또는 eval 명령어를 통해 모델을 학습하거나 학습된 모델을 테스트 함.

**ImageLoader**
- 해당 경로에 있는 data를 pytorch dataloader로 만들어 줌.

**logger**
- tensorboard 또는 txt 파일로 로그를 작성하기 위한 코드.

**model/**
- ANN, CNN 모델이 작성되있음.

## Details
- ANN
    1. 기본 6개 Layer FC 모델.
    2. 3개 Layer의 작은 FC 모델

- CNN
    1. 기본 CNN.
    2. Depthwise-separable conv 모델 ([MobileNet](https://arxiv.org/abs/1704.04861))
