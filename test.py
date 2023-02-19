import librosa
import matplotlib.pyplot as plt
import librosa.display

FIG_SIZE = (15,10)
audio_path = "C:/Users/ICT/Desktop/IoT/AI/E02/S000001/000000.wav"
y, sr = librosa.load(audio_path)
''' y : 파형의 amplitude값
    sr : sampling rate'''
print('amplitude value:' , y)

plt.figure(figsize=FIG_SIZE)
#librosa.display.wavplot(y,sr,alpha=0.4)
librosa.display.waveshow(y,sr,alpha=0.4)
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()