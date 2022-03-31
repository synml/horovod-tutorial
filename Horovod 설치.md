# Horovod 설치 방법

> Horovod 공식 설치 방법을 기반으로 설명합니다. ([링크](https://github.com/horovod/horovod/blob/master/docs/gpus.rst))
Windows는 설치가 불가능합니다!
> 

## 1. CMake 설치

Reference: [https://eehoeskrap.tistory.com/397](https://eehoeskrap.tistory.com/397)

### 1.1. 소스코드 다운로드 (최신버전 확인 필수)

공식 홈페이지: [https://cmake.org/download/](https://cmake.org/download/)

```bash
wget <다운로드_링크>
```

### 1.2. 압축 풀기 및 설치

```bash
tar xzf cmake-<version>.tar.gz
cd cmake-<version>
./bootstrap --prefix=/usr/local
make
make install
```

### 1.3. 설치 확인

```bash
cmake --version
```

---

## 2. NCCL 설치

Reference: [https://kyumdoctor.tistory.com/29](https://kyumdoctor.tistory.com/29)

### 2.1. 설치파일 다운로드

다음 [링크](https://developer.nvidia.com/nccl/nccl-download)에서 시스템 CUDA 버전에 맞는 NCCL 다운로드

만약, 맞는 CUDA 버전이 없으면 다음 [링크](https://developer.nvidia.com/nccl/nccl-legacy-downloads)에서 다운로드 (NCCL은 꼭 최신버전이 아니어도 됨)

### 2.2. 압축 풀기 및 설치

```bash
tar xf nccl_<version>-1+cuda11.2_x86_64.txz
mv nccl_<version>-1+cuda11.2_x86_64 /usr/local/nccl_<version>
```

### 2.3. 환경변수 등록

```bash
cd
nano .profile
# LD_LIBRARY_PATH에 /usr/local/nccl-<version>/lib을 추가
# Ctrl + O, Ctrl + X
source .profile
```

---

## 3. OpenMPI 설치

Reference: [https://blog.naver.com/PostView.nhn?blogId=duqrlwjddns1&logNo=221995194641](https://blog.naver.com/PostView.nhn?blogId=duqrlwjddns1&logNo=221995194641)

### 3.1. 소스코드 다운로드 (최신버전 확인 필수)

공식 홈페이지: [https://www.open-mpi.org/software/ompi/v4.1/](https://www.open-mpi.org/software/ompi/v4.1/)

```bash
wget <다운로드_링크>
```

### 3.2. 압축 풀기 및 설치

```bash
tar xzf openmpi-<version>.tar.gz
cd openmpi-<version>
./configure --prefix=/usr/local
make all install
```

### 3.3. 설치 확인

```bash
mpirun --version
```

만약 libopen-rte.so.40 오류가 발생하면, LD_LIBRARY_PATH에 /usr/local/lib가 있는지 확인하고 없으면 추가한다.

```bash
cd
nano .profile
# LD_LIBRARY_PATH에 /usr/local/lib을 추가
# Ctrl + O, Ctrl + X
source .profile
```

---

## 4. (선택적) Conda에서 gxx_linux-64 설치

Tensorflow, PyTorch가 Conda로 설치된 경우 진행.

```bash
conda install gxx_linux-64
```

---

## 5. Horovod 설치

```bash
HOROVOD_NCCL_HOME=/usr/local/nccl-<version> HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
```