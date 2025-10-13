from inspire_hand_sdk import write6, openSerial
import time

class InspireHand:
    def __init__(self, port='/dev/ttyUSB1', baudrate=115200):
        self.hand = openSerial(port, baudrate)

    # 控制InspireHand
    def reset_hand(self,):
        # print('设置灵巧手运动速度参数，-1为不设置该运动速度！')
        write6(self.hand, 1, 'speedSet', [100, 100, 100, 100, 100, 100])
        time.sleep(2)
        # print('设置灵巧手抓握力度参数！')
        write6(self.hand, 1, 'forceSet', [500, 500, 500, 500, 500, 500])
        time.sleep(1)

    # 夹取
    def close_hand(self, num=[0, 0, 0, 400, 600, 50]):
        write6(self.hand, 1, 'angleSet', num)
        # time.sleep(1)

    # 打开
    def open_hand(self, num=[0, 0, 0, 550, 750, 50]):
        write6(self.hand, 1, 'angleSet', num)
        # time.sleep(1)

if __name__ == "__main__":
    hand = InspireHand(port='COM5', baudrate=115200)
    hand.open_hand()