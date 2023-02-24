import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import matplotlib.cm as cm

frame_length = 0.125 # window length
frame_stride = 0.010 # frame_stride 1ms단위로 뽑음
''' 한 칸은 0.015초(15ms) 겹침'''
def STFT(wav_file):
    y, sr = librosa.load(wav_file)

    stft_result = librosa.stft(y , n_fft=2048, win_length = 2048, hop_length=1024)
    return np.abs(stft_result)

def Mel_Spectrogram(wav_file):
    '''Mel Spectogram을 구하는 함수'''
    y, sr = librosa.load(wav_file)
    input_nftt = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    D = STFT(wav_file)
    S = librosa.feature.melspectrogram(S=D, n_mels=40, n_fft=input_nftt, hop_length=input_stride)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sr, n_mfcc=20)
    
    plt.figure(figsize=(10,4))
    librosa.display.specshow(librosa.power_to_db(mfcc, ref=np.max),y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    #plt.show()

for i in range(1,63):
    audio_path = f"C:/Users/ICT/Desktop/youda/AI/emotion_data/emotion/pleasure/0029_G2A4E1S0C0_KJE/0029_G2A4E1S0C0_KJE_{str(i).zfill(6)}.wav"
    mel_spec = Mel_Spectrogram(audio_path)
    plt.savefig(f'Mel-Spectrogram example{int(str(i).zfill(6))}.png')