import wespeaker

# モデルのパスを指定
model_dir = 'ResNet34_download_dir'
model = wespeaker.load_model_local(model_dir)
# model.set_gpu(0)

# 音声ファイルのパスを指定
audio_file = 'JA_B00000_S00529_W000007.wav'

# 話者ダイアリゼーションの実行
diarization_result = model.diarize(audio_file)

# 結果の表示
for segment in diarization_result:
    # segmentの内容を確認（デバッグ用）
    print(f"Segment content: {segment}, Type: {type(segment)}")
    start_time = float(segment[1])
    end_time = float(segment[2])
    speaker_label = segment[3]
    print(f"{start_time:.3f}\t{end_time:.3f}\t{speaker_label}")