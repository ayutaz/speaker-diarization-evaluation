import os
import json
import torch
import wespeaker
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

# モデルのパスを指定
model_dir = "wespeaker-voxceleb-resnet152-LM"
model = wespeaker.load_model_local(model_dir)

# 必要に応じてデバイスを設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.set_device(device)

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")

# 音声ファイルのディレクトリ
audio_dir = 'callhome_japanese_audio'

# 音声ファイルのリストを作成
audio_files = [
    os.path.join(audio_dir, f)
    for f in os.listdir(audio_dir)
    if f.endswith('.wav') or f.endswith('.mp3')
]

# 結果を格納するリスト
all_results = []

for audio_file in audio_files:
    utt_id = os.path.splitext(os.path.basename(audio_file))[0]
    print(f"Processing {utt_id}...")
    diarization_result = model.diarize(audio_file)
    all_results.append((utt_id, diarization_result))

# 予測結果のRTTMファイルの作成
hypothesis_rttm = 'predicted.rttm'
with open(hypothesis_rttm, 'w', encoding='utf-8') as f:
    for utt_id, result in all_results:
        for segment in result:
            # segment: [開始時間, 終了時間, 話者ラベル]
            start_time = float(segment[1])
            end_time = float(segment[2])
            speaker_label = segment[3]
            duration = end_time - start_time
            f.write(f"SPEAKER {utt_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>\n")

# リファレンスのRTTMファイルを作成（既に作成済みの場合はこの部分をコメントアウト可能）
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

# リファレンスと予測結果のRTTMファイルを読み込み
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

## 出力を保存するテキストファイルを開く
output_file = 'wespeaker-voxceleb-resnet152-LM_results.txt'
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

# 結果が 'der_results.txt' に保存されます
print(f"Results saved to {output_file}")