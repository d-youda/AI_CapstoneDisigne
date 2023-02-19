import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import matplotlib.cm as cm

FIG_SIZE = (15,10)
n_fft=4096
win_length = 4096
hop_length=1024
n_mels = 256

audio_path = "C:/Users/ICT/Desktop/IoT/AI/E02/S000001/000000.wav"
y, sr = librosa.load(audio_path)
''' y : 파형의 amplitude값
    sr : sampling rate'''
print('amplitude value:' , y)

D = np.abs(librosa.stft(y, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=0.00002), sr=sr, hop_length = hop_length, y_axis='mel', x_axis='time', cmap = cm.jet)
plt.colorbar(format='%2.0f dB')
plt.show()