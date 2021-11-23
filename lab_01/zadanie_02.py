import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

data, fs = sf.read('./src/sound1.wav', dtype='float32')

print(fs)
print(data.dtype)
print(data.shape)

# sd.play(data, fs)
# status = sd.wait()
x = np.arange(0, data.shape[0] / fs, 1 / fs)

print(data[:, 0])
plt.subplot(2, 1, 1)
plt.plot(data[:, 0])
plt.subplot(2, 1, 2)
plt.plot(x, data[:, 0])
plt.show()


sf.write('sound_L.wav', data[:, 0], fs)
sf.write('sound_R.wav', data[:, 1], fs)
sf.write('sound_mix.wav', (data[:, 0] + data[:, 1]) / 2, fs)
