import sys, serial, struct
import pyaudio
import numpy as np
import math
from time import sleep
import argparse


def manualTest(ser):
    print('starting manual test...')
    try:
        while True:
            print('enter motor control info such as < 100 1 120 0 >')
            strIn = input()
            vals = [int(val) for val in strIn.split()[:4]]
            vals.insert(0, ord('H'))
            data = struct.pack('BBBBB', *vals)
            ser.write(data)
    except KeyboardInterrupt:
        print('exiting...')
        vals = [ord('H'), 0, 1, 0, 1]
        data = struct.pack('BBBBB', *vals)
        ser.write(data)
        ser.close()


def autoTest(ser):
    print('starting automatic test...')
    try:
        while True:
            for dr in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                for j in range(25, 180, 10):
                    for i in range(25, 180, 10):
                        vals = [ord('H'), i, dr[0], j, dr[1]]
                        print(vals[1:])
                        data = struct.pack('BBBBB', *vals)
                        ser.write(data)
                        sleep(0.1)
    except KeyboardInterrupt:
        print('exiting...')
        vals = [ord('H'), 0, 1, 0, 1]
        data = struct.pack('BBBBB', *vals)
        ser.write(data)
        ser.close()


def getInputDevice(p):
    index = None
    nDevices = p.get_device_count()
    print('Found %d deviced. Select input device:' % nDevices)
    for i in range(nDevices):
        deviceInfo = p.get_device_info_by_index(i)
        devName = deviceInfo['name']
        print("%d: %s" % (i, devName))
    try:
        index = int(input())
    except:
        pass
    if index is not None:
        devName = p.get_device_info_by_index(index)['name']
        print("Input device chosen: %s" % devName)
    return


def fftLive(ser):
    p = pyaudio.PyAudio()
    inputIndex = getInputDevice(0)

    fftLen = 2**11
    sampleRate = 44100

    print('Opening stream...')
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate = sampleRate,
                    input=True,
                    frames_per_buffer=fftLen,
                    input_device_index=inputIndex)
    try:
        while True:
            data = stream.read(fftLen)
            dataArray = np.frombuffer(data, dtype=np.int16)
            fftVals = np.fft.rfft(dataArray)*2.0/fftLen
            fftVals = np.abs(fftVals)
            levels = [np.sum(fftVals[0:100])/100,
                      np.sum(fftVals[100:1000])/900,
                      np.sum(fftVals[1000:2500])/1500]

            vals = [ord('H'), 100, 1, 100, 1]
            vals[1] = int(5*levels[0]) % 255
            vals[3] = int(100+levels[1]) % 255

            d1 = 0
            if levels[2] > 0.1:
                d1 = 1
            vals[2] = d1
            vals[4] = 0

            data = struct.pack('BBBBB', *vals)
            ser.write(data)
            sleep(0.001)
    except KeyboardInterrupt:
        print('stopping...')
    finally:
        print('cleaning up')
        stream.close()
        p.terminate()
        vals = [ord('H'), 0, 1, 0, 1]
        data = struct.pack('BBBBB', *vals)
        ser.write(data)
        ser.clush()
        ser.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze audio input and sends motor control information via serial port')
    parser.add_argument('--port', dest='serial_port_name', required=True)
    parser.add_argument('--mtest', action='store_true', default=False)
    parser.add_argument('--atest', action='store_true', default=False)
    args = parser.parse_args()

    strPort = args.serial_port_name
    print('opening ', strPort)
    ser = serial.Serial(strPort, 9600)
    if args.mtest:
        manualTest(ser)
    elif args.atest:
        autoTest(ser)
    else:
        fftLive(ser)


if __name__ == '__main__':
    main()