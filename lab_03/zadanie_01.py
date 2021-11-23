import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import soundfile as sf
import scipy.fftpack
import sounddevice as sd


def quantization(signal, N):
    if signal.dtype == np.float32:
        minimalFloatValue = -1
        maximalFloatValue = 1
        zA = (signal-minimalFloatValue)/(maximalFloatValue-minimalFloatValue)
        wyn = np.round(zA*(2**N-1))/(2**N-1)
        zC = (wyn*(maximalFloatValue-minimalFloatValue))+minimalFloatValue
        return zC
    signalType = signal.dtype
    x = signal.astype(np.float32)
    minimalTypeValue = np.iinfo(signalType).min
    maximumTypeValue = np.iinfo(signalType).max
    zA = (x-minimalTypeValue)/(maximumTypeValue-minimalTypeValue)
    quant = np.round(zA*(2**N-1))/(2**N-1)
    zC = (quant*(maximumTypeValue-minimalTypeValue))+minimalTypeValue
    return zC.astype(signalType)


def decimation(signal, N):
    return signal[::N]


def interpolation(signal, linear, fs, new_fs):
    time = (len(signal))/fs
    newSignalLength = int(len(signal)*new_fs/fs)
    x = np.linspace(0, time, len(signal))
    x1 = np.linspace(0, time, newSignalLength)
    if linear == 1:
        metode_lin = interp1d(x, signal)
        y_lin = metode_lin(x1)
        return x1, y_lin
    metode_nonlin = interp1d(x, signal, kind='cubic')
    y_nonlin = metode_nonlin(x1)
    return x1, y_nonlin


def doQuantization():
    numbersOfbits = [4, 8, 16, 24]
    filesToRead = ['sin_60Hz.wav', 'sin_440Hz.wav',
                   'sin_8000Hz.wav', 'sin_combined.wav']
    for i in range(0, len(filesToRead)):
        for j in range(-1, len(numbersOfbits)):
            data, fs = sf.read(filesToRead[i], dtype=np.int32)
            if j == -1:
                plt.figure()
                time = (len(data))/fs
                x = np.linspace(0, time, len(data))
                title = filesToRead[i] + " - sygnał oryginalny"
                plt.title(title)
                plt.xlabel('t[s]')
                plt.ylabel('f(t)')
                plt.plot(x, data)
                plt.figure()
                if i == 0:
                    fsize = 2**6
                if i == 1:
                    fsize = 2**9
                if i == 2 or i == 3:
                    fsize = 2**13
                yf = scipy.fftpack.fft(data, fsize)
                title = filesToRead[i] + " - widmo"
                plt.title(title)
                plt.xlabel('f[Hz]')
                plt.ylabel('y(f)[dB]')
                plt.plot(np.arange(0, fs/2, fs/fsize), 20 *
                         np.log10(np.abs(yf[:fsize//2])))
            else:
                afterQuantization = quantization(data, numbersOfbits[j])
                plt.figure()
                time = (len(afterQuantization))/fs
                x = np.linspace(0, time, len(afterQuantization))
                title = filesToRead[i] + " - kwantyzacja " + \
                    str(numbersOfbits[j])+" bity"
                plt.title(title)
                plt.xlabel('t[s]')
                plt.ylabel('f(t)')
                plt.plot(x, afterQuantization)
                plt.figure()
                if i == 0:
                    fsize = 2**6
                if i == 1:
                    fsize = 2**9
                if i == 2 or i == 3:
                    fsize = 2**13
                yf = scipy.fftpack.fft(afterQuantization, fsize)
                title = filesToRead[i] + " - kwantyzacja " + \
                    str(numbersOfbits[j])+" bity - widmo"
                plt.title(title)
                plt.xlabel('f[Hz]')
                plt.ylabel('y(f)[dB]')
                plt.plot(np.arange(0, fs/2, fs/fsize), 20 *
                         np.log10(np.abs(yf[:fsize//2])))
            print(filesToRead[i], numbersOfbits[j])
# doQuantization()


def doDecimation():
    filesToRead = ['sin_60Hz.wav', 'sin_440Hz.wav',
                   'sin_8000Hz.wav', 'sin_combined.wav']
    new_fs = [2000, 4000, 8000, 16000, 24000]
    for i in range(0, len(filesToRead)):
        for j in range(0, len(new_fs)):
            data, fs = sf.read(filesToRead[i], dtype=np.int32)
            N = int(fs/new_fs[j])
            afterDecimation = decimation(data, N)
            plt.figure()
            time = (len(afterDecimation))/new_fs[j]
            x = np.linspace(0, time, len(afterDecimation))
            title = filesToRead[i] + " - decymacja "+str(new_fs[j])+"Hz - fs"
            plt.title(title)
            plt.xlabel('t[s]')
            plt.ylabel('f(t)')
            plt.plot(x, afterDecimation)
            plt.figure()
            if i == 0:
                fsize = 2**6
            if i == 1:
                fsize = 2**9
            if i == 2 or i == 3:
                fsize = 2**13
            yf = scipy.fftpack.fft(afterDecimation, fsize)
            title = filesToRead[i] + " - decymacja " + \
                str(new_fs[j])+"Hz - fs - widmo"
            plt.title(title)
            plt.xlabel('f[Hz]')
            plt.ylabel('y(f)[dB]')
            plt.plot(
                np.arange(0, new_fs[j]/2, new_fs[j]/fsize), 20*np.log10(np.abs(yf[:fsize//2])))
# doDecimation()


def doInterpolation():
    filesToRead = ['sin_60Hz.wav', 'sin_combined.wav']
    new_fs = [41000, 16950]
    for i in range(0, len(filesToRead)):
        for j in range(0, len(new_fs)):
            for k in range(0, 2):
                plt.figure()
                data, fs = sf.read(filesToRead[i], dtype=np.int32)
                x, afterInterpolation = interpolation(data, k, fs, new_fs[j])
                time = (len(afterInterpolation))/new_fs[j]
                xx = np.linspace(0, time, len(afterInterpolation))
                if k == 0:
                    title = filesToRead[i] + \
                        " - interpolacja nieliniowa "+str(new_fs[j])+"Hz - fs"
                if k == 1:
                    title = filesToRead[i] + \
                        " - interpolacja liniowa "+str(new_fs[j])+"Hz - fs"
                plt.title(title)
                plt.xlabel('t[s]')
                plt.ylabel('f(t)')
                plt.plot(xx, afterInterpolation)
                plt.figure()
                if i == 0:
                    fsize = 2**6
                if i == 1:
                    fsize = 2**9
                if i == 2 or i == 3:
                    fsize = 2**13
                yf = scipy.fftpack.fft(afterInterpolation, fsize)
                if k == 0:
                    title = filesToRead[i] + \
                        " - interpolacja nieliniowa "+str(new_fs[j])+"Hz - fs"
                if k == 1:
                    title = filesToRead[i] + \
                        " - interpolacja liniowa "+str(new_fs[j])+"Hz - fs"
                plt.title(title)
                plt.xlabel('f[Hz]')
                plt.ylabel('y(f)[dB]')
                plt.plot(
                    np.arange(0, new_fs[j]/2, new_fs[j]/fsize), 20*np.log10(np.abs(yf[:fsize//2])))
# doQuantization()
# doDecimation()
# doInterpolation()


def zad22_interpolacja(plik, new_fs):
    data, fs = sf.read(plik, dtype=np.float32)
    x, y = interpolation(data, 1, fs, new_fs)
    sd.play(y, new_fs)
    sd.wait()


def zad22_kwantyzacja(plik, N):
    data, fs = sf.read(plik, dtype=np.int32)
    afterQuantization = quantization(data, N)
    sd.play(afterQuantization, fs)
    sd.wait()


def zad22_decymacja(plik, N):
    data, fs = sf.read(plik, dtype=np.int32)
    afterDecimation = decimation(data, N)
    sd.play(afterDecimation, fs)
    sd.wait()


# zad22_kwantyzacja('sing_high1.wav', 24)
# zad22_decymacja('sing_low1.wav', 24)

zad22_interpolacja('sing_low1.wav', 33000)
plt.show()
