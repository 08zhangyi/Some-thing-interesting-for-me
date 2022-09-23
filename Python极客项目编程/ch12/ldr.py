import serial, argparse
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnalogPlot:
    def __init__(self, strPort, maxLen):
        self.ser = serial.Serial(strPort, 9600)
        self.a0Vals = deque([0.0]*maxLen)
        self.a1Vals = deque([0.0]*maxLen)
        self.maxLen = maxLen

    def add(self, data):
        assert(len(data) == 2)
        self.addToDeq(self.a0Vals, data[0])
        self.addToDeq(self.a1Vals, data[1])

    def addToDeq(self, buf, val):
        buf.pop()
        buf.appendleft(val)

    def update(self, frameNum, a0, a1):
        try:
            line = self.ser.readline()
            data = [float(val) for val in line.split()]
            if len(data) == 2:
                self.add(data)
                a0.set_data(range(self.maxLen), self.a0Vals)
                a1.set_data(range(self.maxLen), self.a1Vals)
        except:
            pass

        return a0, a1

    def close(self):
        self.ser.flush()
        self.ser.close()


def main():
    parser = argparse.ArgumentParser(description="LDR serial")
    parser.add_argument('--port', dest='port', required=True)
    parser.add_argument('--N', dest='maxLen', required=False)
    args = parser.parse_args()
    strPort = args.port
    print('reading from serial port %s...' % strPort)
    maxLen = 100
    if args.maxLen:
        maxLen = int(args.maxLen)

    analogPlot = AnalogPlot(strPort, maxLen)

    print('plotting data...')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, maxLen), ylim=(0, 1023))
    a0 = ax.plot([], [])
    a1 = ax.plot([], [])
    anim = animation.FuncAnimation(fig, analogPlot.update, fargs=(a0, a1), interval=20)
    plt.show()

    analogPlot.close()
    print('exiting.')


if __name__ == '__main__':
    main()