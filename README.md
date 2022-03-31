# Horovod Tutorial

> Horovod 공식 docs를 기반으로 설명합니다. ([링크](https://github.com/horovod/horovod/blob/master/docs/pytorch.rst))

PyTorch와 함께 Horovod를 사용하려면 학습 스크립트를 다음과 같이 수정하십시오.

1. ``hvd.init()`` 실행
2. 각 GPU를 프로세스에 하나씩 고정합니다.
3. worker 수에 따라 lr을 조정합니다.
4. optimizer를 ``hvd.DistributedOptimizer``로 감쌉니다.
5. 초기 변수 상태를 rank 0에서 다른 모든 프로세스로 브로드캐스트합니다.
6. 다른 worker가 checkpoint를 손상시키지 않도록 worker 0만 checkpoint를 저장하도록 코드를 수정합니다.
