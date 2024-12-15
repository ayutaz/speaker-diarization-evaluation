# wespeaker-voxceleb-resnet34-LM

## setup
```bash
uv venv -p 3.11
source venv/bin/activate
```

## install
```bash
uv pip install -r requirements.txt
```

## run
```bash
 wespeaker -p ResNet34_download_dir --task diarization --audio_file .\JA_B00000_S00529_W000007.wav
```

or 

```sh
python sample.py
```
