import os
import json
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
from sklearn.cluster import AgglomerativeClustering
from xvector_jtubespeech import XVector
from tqdm import tqdm

# 1. x-vectorモデルのロード
model = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector", trust_repo=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")

# 2. 音声ファイルのディレクトリ
audio_dir = 'callhome_japanese_audio'

# 音声ファイルのリストを作成
audio_files = [
    os.path.join(audio_dir, f)
    for f in os.listdir(audio_dir)
    if f.endswith('.wav') or f.endswith('.mp3')
]

# 音声区間検出（VAD）の準備
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

# VADモデルはCPU上にあることを確認
vad_model.eval()  # 必要に応じて評価モードに設定

# 予測結果のRTTMファイルを作成するためのリスト
all_results = []

for audio_file in audio_files:
    utt_id = os.path.splitext(os.path.basename(audio_file))[0]
    print(f"Processing {utt_id}...")

    # 2.1 音声の読み込み（wav は CPU 上）
    wav, sr = torchaudio.load(audio_file)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000
    # wav = wav.to(device)  # この行は削除またはコメントアウト

    # 2.2 音声区間検出（VAD）の適用（wav は CPU 上）
    speech_timestamps = get_speech_timestamps(wav.squeeze(0), vad_model, sampling_rate=sr)
    if not speech_timestamps:
        print(f"No speech detected in {utt_id}.")
        continue

    # 3. セグメントごとにx-vectorを抽出
    embeddings = []
    segments = []
    for ts in speech_timestamps:
        start_frame = ts['start']
        end_frame = ts['end']
        segment_wav = wav[:, start_frame:end_frame]
        segment_wav_np = segment_wav.cpu().numpy().squeeze(0)

        # 3.1 MFCCの抽出
        segment_tensor = torch.from_numpy(segment_wav_np.astype(np.float32)).unsqueeze(0).to(device)
        mfcc = kaldi.mfcc(segment_tensor, num_ceps=24, num_mel_bins=24).unsqueeze(0)

        # 3.2 x-vectorの抽出
        with torch.no_grad():
            xvector = model.vectorize(mfcc)
        xvector = xvector.cpu().numpy()[0]

        embeddings.append(xvector)
        # 時間を秒に変換
        start_time = start_frame / sr
        end_time = end_frame / sr
        segments.append((start_time, end_time))

    embeddings = np.array(embeddings)

    # 4. クラスタリングによる話者ダイアライゼーション
    num_speakers = 2  # 必要に応じて変更

    clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)

    # 予測結果を保存
    diarization_result = []
    for (segment, label) in zip(segments, labels):
        diarization_result.append([utt_id, segment[0], segment[1], label])
    all_results.extend(diarization_result)

# 5. 予測結果のRTTMファイルの作成
hypothesis_rttm = 'predicted.rttm'
with open(hypothesis_rttm, 'w', encoding='utf-8') as f:
    for entry in all_results:
        utt_id, start_time, end_time, speaker_label = entry
        duration = end_time - start_time
        f.write(f"SPEAKER {utt_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> speaker_{speaker_label} <NA> <NA>\n")


# 6. リファレンスのRTTMファイルを作成（既に作成済みの場合はこの部分をコメントアウト可能）
json_input_path = 'callhome_japanese_metadata.json'
reference_rttm = 'callhome_japanese.rttm'
with open(json_input_path, 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)
with open(reference_rttm, 'w', encoding='utf-8') as f_rttm:
    for metadata in metadata_list:
        audio_filename = metadata['audio_filename']
        uri = os.path.splitext(audio_filename)[0]
        utterances = metadata['utterances']
        for utt in utterances:
            start_time = utt['start_time']
            end_time = utt['end_time']
            duration = end_time - start_time
            speaker = utt['speaker']
            f_rttm.write(f"SPEAKER {uri} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

# 7. リファレンスと予測結果のRTTMファイルを読み込み
def load_rttm(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            uri = tokens[1]
            start_time = float(tokens[3])
            duration = float(tokens[4])
            end_time = start_time + duration
            speaker = tokens[7]
            segment = Segment(start_time, end_time)
            if uri not in annotations:
                annotations[uri] = Annotation(uri=uri)
            annotations[uri][segment] = speaker
    return annotations

reference = load_rttm(reference_rttm)
hypothesis = load_rttm(hypothesis_rttm)

metric = DiarizationErrorRate()

# 8. 評価結果を保存するテキストファイルを開く
output_file = 'xvector_jtubespeech-der_results.txt'
with open(output_file, 'w', encoding='utf-8') as result_f:

    # 各ファイルごとに評価
    for utt_id in reference:
        ref = reference[utt_id]
        hyp = hypothesis.get(utt_id, None)

        if hyp is None:
            result_line = f"Hypothesis for {utt_id} not found."
            print(result_line)
            result_f.write(result_line + '\n')
            continue

        der = metric(ref, hyp)
        result_line = f"{utt_id}: DER = {der * 100:.2f}%"
        print(result_line)
        result_f.write(result_line + '\n')

    # 全体のDERを計算
    total_der = abs(metric)
    total_result_line = f"Total DER: {total_der * 100:.2f}%"
    print(total_result_line)
    result_f.write(total_result_line + '\n')

print(f"Results saved to {output_file}")