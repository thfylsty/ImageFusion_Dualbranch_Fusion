import os
import time
while True:
    os.system("nvidia-smi")
    time.sleep(1)
# str = os.popen('gpustat')
# str = str.readlines()
# print(str)