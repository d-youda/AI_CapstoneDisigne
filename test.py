import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import matplotlib.cm as cm

frame_length = 0.125 # window length
frame_stride = 0.010 # frame_stride 1ms단위로 뽑음
''' 한 칸은 0.015초(15ms) 겹침'''

def Mel_Spectrogram(wav_file):
    '''Mel Spectogram을 구하는 함수'''
    y,sr = librosa.load(wav_file)
    input_nftt = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nftt, hop_length=input_stride)

    print("Wav Lenght:{} , Mel_Spectrogram:{}".format(len(y)/sr , np.shape(S)))
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()
    return S

audio_path = "C:/Users/ICT/Desktop/IoT/AI/E02/S000001/000000.wav"
mel_spec = Mel_Spectrogram(audio_path)