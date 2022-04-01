# Horovod Tutorial

> Horovod 공식 docs를 기반으로 설명합니다. ([링크](https://github.com/horovod/horovod/blob/master/docs/pytorch.rst))

PyTorch와 함께 Horovod를 사용하려면 코드를 다음과 같이 수정하십시오.

1. ``hvd.init()`` 실행

2. 각 GPU를 프로세스에 하나씩 고정합니다.
```python
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
```

3. worker 수에 따라 lr을 조정합니다.

4. 초기 변수 상태를 rank 0에서 다른 모든 프로세스로 브로드캐스트합니다.
```python
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```

5. optimizer를 ``hvd.DistributedOptimizer``로 감쌉니다.
```python
hvd.DistributedOptimizer(optimizer, model.named_parameters(), compression, op, gradient_predivide_factor)
```

6. 다른 worker가 checkpoint를 손상시키지 않도록 worker 0만 checkpoint를 저장하도록 코드를 수정합니다.

7. 쉘에서 아래 명령으로 실행합니다.
```bash
horovodrun -np 4 python train.py
```