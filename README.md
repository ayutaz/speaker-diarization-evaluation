# 話者ダイアライゼーション評価コード

このリポジトリでは、話者ダイアライゼーションの評価を行うためのコードを提供しています。このREADMEでは、コードの概要、使用方法、必要な環境について説明します。

## 目次

- [概要](#概要)
- [必要な環境](#必要な環境)
- [使用方法](#使用方法)
- [評価指標](#評価指標)
- [参考文献](#参考文献)

## 概要

この評価コードは、音声データに対して話者ダイアライゼーションを実施し、その結果を評価するためのものです。以下の機能を持っています：

- **話者埋め込みの抽出**：WeSpeakerモデルやx-vectorモデルを使用して、音声区間から話者埋め込みベクトルを抽出します。
- **次元削減**：UMAPを使用して埋め込みベクトルの次元を削減します。
- **クラスタリング**：HDBSCANを使用して、次元削減された埋め込みベクトルをクラスタリングします。
- **評価**：pyannote.metricsを使用して、ダイアライゼーションの性能を評価します。

## 必要な環境

以下のパッケージが必要です：

- Python 3.x
- PyTorch
- Torchaudio
- NumPy
- SciPy
- Scikit-learn
- UMAP (`umap-learn`)
- HDBSCAN
- pyannote.metrics
- tqdm

**インストール方法**

```bash
pip install -r requirements.txt
```

または個別にインストール：

```bash
pip install torch torchaudio numpy scipy scikit-learn umap-learn hdbscan pyannote.metrics tqdm
```

# 使用方法
* 音声ファイルの準備

1. 音声データを所定のディレクトリに配置します。
2. リファレンスのRTTMファイルの準備
3. 評価用に、正解の話者セグメント情報を含むRTTMファイルを用意します。
4. 評価スクリプトの実行

```bash
python evaluate_diarization.py --audio_dir path/to/audio_files --reference_rttm path/to/reference.rttm
--audio_dir：音声ファイルが保存されているディレクトリのパス。
--reference_rttm：リファレンスのRTTMファイルのパス。
```

# 結果の確認

1. スクリプトの実行後、各音声ファイルごとのDER（Diarization Error Rate）が表示され、全体のDERも計算されます。
2. 結果はresults.txtファイルにも保存されます。

# 評価指標

Diarization Error Rate (DER)：話者ダイアライゼーションの誤り率を示す指標で、以下の3つの要素の合計で計算されます。
* ミスマッチ（スピーカーミスアサインメント）
* 話者の挿入エラー
* 話者の削除エラー

# 参考文献
* WeSpeaker：https://github.com/wenet-e2e/wespeaker
* x-vector models：https://github.com/sarulab-speech/xvector_jtubespeech
* Silero VAD：https://github.com/snakers4/silero-vad
* pyannote.metrics：https://github.com/pyannote/pyannote-metrics

# 注意事項

* 使用するモデル（WeSpeakerまたはx-vector）に応じて、スクリプト内の設定を変更してください。
* UMAPやHDBSCANのパラメータは、データセットに応じて調整することで、結果が改善する可能性があります。
* 評価結果は、データの品質やモデルの性能によって異なります。必要に応じて前処理やパラメータの最適化を行ってください。

# ライセンス
* このプロジェクトは、Apache License 2.0の下でライセンスされています。詳細については、LICENSEファイルを参照してください。

ご不明な点や問題が発生した場合は、Issueを作成するか、直接ご連絡ください。
