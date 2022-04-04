num_gpu='nvidia-smi -L | wc -l'
horovodrun -np "$num_gpu" python main.py
