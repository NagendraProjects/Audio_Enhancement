#pip install pyAudioAnalysis pydub matplotlib scipy sklearn hmmlearn simplejson eyed3 pydub
# to execute the below code, we need to install above libraries

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import librosa

def process_audio_files(input_folder, output_folder):
    # get all .wav files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    for file in files:
        audio_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)

        # Calculate raw audio duration
        raw_duration = librosa.get_duration(filename=audio_file_path)
        print(f"Raw audio duration: {raw_duration} seconds")

        # Read audio data
        sampling_rate, signal = audioBasicIO.read_audio_file(audio_file_path)

        # Segment audio using silence removal
        segments = aS.silence_removal(signal, sampling_rate, 0.020, 0.020, smooth_window=1.0, weight=0.3, plot=False)

        # Concatenate segments and normalize
        processed_signal = np.concatenate([signal[int(start * sampling_rate):int(end * sampling_rate)] for start, end in segments])
        audio = AudioSegment(processed_signal.tobytes(), frame_rate=sampling_rate, sample_width=processed_signal.dtype.itemsize, channels=1)
        normalized_audio = normalize(audio)

        # Export processed audio
        normalized_audio.export(output_file_path, format="wav")

        # Calculate cleaned audio duration
        cleaned_duration = librosa.get_duration(filename=output_file_path)
        print(f"Cleaned audio duration: {cleaned_duration} seconds")

    print(f"Processing completed successfully. Cleaned audio files have been saved to {output_folder}")

# Usage:
# process_audio_files('path_to_your_input_folder', 'path_to_your_output_folder')
