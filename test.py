import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import matplotlib.cm as cm

FIG_SIZE = (15,10)
audio_path = "C:/Users/ICT/Desktop/IoT/AI/E02/S000001/000000.wav"
y, sr = librosa.load(audio_path)
''' y : 파형의 amplitude값
    sr : sampling rate'''
print('amplitude value:' , y)

stft_result = librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=1024)
D = np.abs(stft_result)
S_dB = librosa.power_to_db(D, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, hop_length = 1024, y_axis='linear', x_axis='time', cmap = cm.jet)
plt.colorbar(format='%2.0f dB')
plt.show()