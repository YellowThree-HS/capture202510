import serial
import time


regdict = {
    'ID' : 1000,
    'baudrate' : 1001,
    'clearErr' : 1004,
    'forceClb' : 1009,
    'angleSet' : 1486,
    'forceSet' : 1498,
    'speedSet' : 1522,
    'angleAct' : 1546,
    'forceAct' : 1582,
    'errCode' : 1606,
    'statusCode' : 1612,
    'temp' : 1618,
    'actionSeq' : 2320,
    'actionRun' : 2322
}


def openSerial(port, baudrate):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.open()
    return ser


def writeRegister(ser, id, add, num, val):
    bytes = [0xEB, 0x90]
    bytes.append(id) # id
    bytes.append(num + 3) # len
    bytes.append(0x12) # cmd
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF) # add
    for i in range(num):
        bytes.append(val[i])
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    ser.read_all() # 把返回帧读掉，不处理


def readRegister(ser, id, add, num, mute=False):
    bytes = [0xEB, 0x90]
    bytes.append(id) # id
    bytes.append(0x04) # len
    bytes.append(0x11) # cmd
    bytes.append(add & 0xFF)
    bytes.append((add >> 8) & 0xFF) # add
    bytes.append(num)
    checksum = 0x00
    for i in range(2, len(bytes)):
        checksum += bytes[i]
    checksum &= 0xFF
    bytes.append(checksum)
    ser.write(bytes)
    time.sleep(0.01)
    recv = ser.read_all()
    if len(recv) == 0:
        return []
    num = (recv[3] & 0xFF) - 3
    val = []
    for i in range(num):
        val.append(recv[7 + i])
    if not mute:
        print('读到的寄存器值依次为：', end='')
        for i in range(num):
            print(val[i], end=' ')
        print()
    return val


def write6(ser, id, str, val):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet':
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        writeRegister(ser, id, regdict[str], 12, val_reg)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')


def read6(ser, id, str):
    if str == 'angleSet' or str == 'forceSet' or str == 'speedSet' or str == 'angleAct' or str == 'forceAct':
        val = readRegister(ser, id, regdict[str], 12, True)
        if len(val) < 12:
            print('没有读到数据')
            return
        val_act = []
        for i in range(6):
            val_act.append((val[2*i] & 0xFF) + (val[1 + 2*i] << 8))
        print('读到的值依次为：', end='')
        for i in range(6):
            print(val_act[i], end=' ')
        print()
    elif str == 'errCode' or str == 'statusCode' or str == 'temp':
        val_act = readRegister(ser, id, regdict[str], 6, True)
        if len(val_act) < 6:
            print('没有读到数据')
            return
        print('读到的值依次为：', end='')
        for i in range(6):
            print(val_act[i], end=' ')
        print()
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'temp\'')




if __name__ == "__main__":
    hand = openSerial('/dev/ttyUSB1', 115200)
    print('设置灵巧手运动速度参数，-1为不设置该运动速度！')
    write6(hand, 1, 'speedSet', [100, 100, 100, 100, 100, 100])
    time.sleep(2)
    # print('设置灵巧手抓握力度参数！')
    # write6(hand, 1, 'forceSet', [500, 500, 500, 500, 500, 500])
    # time.sleep(1)

    # right hand
    write6(hand, 1, 'angleSet', [0, 0, 0, 550, 750, 50])
    time.sleep(1)

    # write6(hand, 1, 'angleSet', [0, 0, 0, 400, 600, 50])
    # time.sleep(1)

    # left hand
    # write6(hand, 1, 'angleSet', [0, 0, 0, 550, 750, 50])
    # time.sleep(1)

    # write6(hand, 1, 'angleSet', [0, 0, 0, 450, 600, 50])
    # time.sleep(1)

    # read6(hand, 1, 'angleAct')
    # time.sleep(1)