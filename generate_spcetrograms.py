import os
import librosa
import librosa.display   # <-- fix
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)

    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))

    plt.figure(figsize=(7, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Log Magnitude Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# for all audio files in the "audio" directory, generate spectrograms
audio_dir = 'temporal-concept-dynamics-audioldm2/audio'
output_dir = 'temporal-concept-dynamics-audioldm2/spectrograms'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.wav', '.png'))
        generate_spectrogram(audio_path, output_path)


