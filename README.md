# Horovod Tutorial

> Horovod 공식 docs를 기반으로 설명합니다. ([링크](https://github.com/horovod/horovod/blob/master/docs/pytorch.rst))

PyTorch와 함께 Horovod를 사용하려면 코드를 다음과 같이 수정하십시오.

1. Horovod를 초기화합니다.
```python
hvd.init()
```

2. 각 GPU를 프로세스에 하나씩 고정합니다.
```python
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
```

3. 데이터셋을 DistributedSampler로 프로세스 개수 만큼 분할합니다.
```python
dataset = datasets.CIFAR10('data')
sampler = torch.utils.data.distributed.DistributedSampler(dataset, hvd.size(), hvd.rank())
dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
```

4. 초기 변수 상태를 root rank에서 다른 모든 프로세스(rank)로 브로드캐스트합니다.
```python
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```

5. optimizer를 ``hvd.DistributedOptimizer``로 wrapping합니다.
```python
optimizer = hvd.DistributedOptimizer(optimizer, model.named_parameters())
```

6. 파일 쓰기 작업은 root rank에서만 작동하도록 filelock을 추가합니다.
```python
with filelock.FileLock('.filelock'):
    ...
```

7. 쉘에서 아래 명령으로 Horovod를 실행합니다.
```bash
horovodrun -np 4 python main.py
```