import pynvml
import numpy as np
import time
pynvml.nvmlInit()
handles = []
# 这里的0是GPU id
for i in range(4):
    handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
# print(handles)
memInfos = []
for handle in handles:
    memInfos.append(pynvml.nvmlDeviceGetMemoryInfo(handle))
commandList = ['',
               '',
               '',]
commandFalg = np.ones(len(commandList))

def getUsedRate(memInfo):
    return memInfo.used / memInfo.total


def sendCommand(deviceID):
    print(str(deviceID) + ': command')


setRate = 0.2

while True:
    num = 0
    print("===============")
    min_use = 100
    ID = 0
    can_ues=[]
    for memInfo in memInfos:
        num = num+1
        min_use = min(min_use, getUsedRate(memInfo)*100)
        ID = num
        print(num, int(getUsedRate(memInfo)*100), "%")
        if min_use<10:
            break
    if min_use>50:
        print('显卡全特么被占啦！！！')
    else:
        print('找到空的了！！！', ID,min_use)
        sendCommand(num)
    print("===============")


    time.sleep(1)

    # print('显卡空闲')
    # print('显卡被占用')
    # while (getUsedRate(memInfo0) < setRate) | (getUsedRate(memInfo1) < setRate) | (getUsedRate(memInfo2) < setRate) | (
    #         getUsedRate(memInfo3) < setRate):
    #     print('存在显卡空闲')
    #     if getUsedRate(memInfo0) < setRate:
    #         sendCommand(0)
    #     elif getUsedRate(memInfo1) < setRate:
    #         sendCommand(1)
    #     if getUsedRate(memInfo2) < setRate:
    #         sendCommand(2)
    #     if getUsedRate(memInfo3) < setRate:
    #         sendCommand(3)
