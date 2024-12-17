# スクリプトの最初に環境変数を設定
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
# UMAPとHDBSCANをインポート
import umap
import hdbscan
# 必要に応じてPAHCクラスをインポートまたは定義
from wespeaker.diar.umap_clusterer import PAHC  # PAHCクラスを別途コピーして使用

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
vad_model.eval()

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

    # 2.2 音声区間検出（VAD）の適用
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
        segment_wav_np = segment_wav.numpy().squeeze(0)

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

    # 4. UMAPによる次元削減
    if len(embeddings) <= 2:
        labels = [0] * len(embeddings)
    else:
        umap_embeddings = umap.UMAP(
            n_components=min(32, len(embeddings) - 2),
            metric='cosine',
            n_neighbors=16,  # 必要に応じて調整
            min_dist=0.05,   # 必要に応じて調整
            random_state=2023,
            n_jobs=1
        ).fit_transform(embeddings)

        # 5. HDBSCANによるクラスタリング
        labels = hdbscan.HDBSCAN(
            allow_single_cluster=True,
            min_cluster_size=4,
            approx_min_span_tree=False,
            core_dist_n_jobs=1
        ).fit_predict(umap_embeddings)

        # 6. PAHCによるクラスタのマージと吸収
        labels = PAHC(
            merge_cutoff=0.3,
            min_cluster_size=3,
            absorb_cutoff=0.0
        ).fit_predict(labels, embeddings)

    # 予測結果を保存
    diarization_result = []
    for (segment, label) in zip(segments, labels):
        diarization_result.append([utt_id, segment[0], segment[1], label])
    all_results.extend(diarization_result)

# 7. 予測結果のRTTMファイルの作成
hypothesis_rttm = 'predicted.rttm'
with open(hypothesis_rttm, 'w', encoding='utf-8') as f:
    for entry in all_results:
        utt_id, start_time, end_time, speaker_label = entry
        duration = end_time - start_time
        f.write(f"SPEAKER {utt_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> speaker_{speaker_label} <NA> <NA>\n")

# 以下、評価コード（リファレンスのRTTMファイルの読み込みなど）を追加

# 8. リファレンスのRTTMファイルを読み込み（ご自身のコードに合わせてください）
# 例えば:
reference_rttm = 'callhome_japanese.rttm'
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

# 9. 評価結果を保存するテキストファイルを開く
output_file = 'xvector_jtubespeech-der-umap_results.txt'
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