pip3 install -e ".[model_worker,webui]"
apt install libgl1-mesa-glx
pip install flash-attn==2.5.0  --no-build-isolation
pip install openai
pip install -q mediapipe
wget -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite -P ./fastchat/llm_judge/data/mt_bench/model_judgment
pip install anthropic
pip install jsonlines
pip install pydantic==1.10.13
pip install pydantic_core==2.16.1
pip install gradio==3.35.2
pip install ray
pip install pytest
pip install gradio_client==0.8.1
pip install --upgrade transformers
