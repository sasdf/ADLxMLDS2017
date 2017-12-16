import multiprocessing as mp
import random
import time

a = mp.Value('d', 0)

def worker():
    time.sleep(random.random())
    global a
    a.value += 1
    print(a)

p = [mp.Process(target=worker) for i in range(10)]
for pi in p: pi.start()
for pi in p: pi.join()
