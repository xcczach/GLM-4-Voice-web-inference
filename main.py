# main.py
from ml_web_inference import expose, Request, StreamingResponse
import torch
import io
import argparse
import os
import tempfile
from threading import Thread
from queue import Queue
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from transformers.generation.streamers import BaseStreamer
import torchaudio
import sys
import uuid

# 插入必要的路径
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import soundfile as sf
import setproctitle


# 定义 TokenStreamer 类
class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


# 全局变量
model = None
glm_tokenizer = None
audio_decoder = None
whisper_model = None
feature_extractor = None
device = None
model_size_mb = 35000


async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    sample_rate = data["sample_rate"]
    audio_data = data["audio_data"]

    # 将音频数据保存为临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        sf.write(temp_audio_path, data=audio_data, samplerate=sample_rate)
        # 提取语音标记
        audio_tokens = extract_speech_token(
            whisper_model, feature_extractor, [temp_audio_path], f"cuda:{device}" if type(device) == int else device
        )[0]
    if len(audio_tokens) == 0:
        raise Exception("No audio tokens extracted")
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"

    # 准备提示词
    user_input = audio_tokens
    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens."
    inputs = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

    # 生成响应
    temperature = 0.2
    top_p = 0.8
    max_new_tokens = 2000

    params = {
        "prompt": inputs,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }

    # 定义 generate_stream 函数
    def generate_stream(params):
        prompt = params["prompt"]
        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 1.0)
        max_new_tokens = params.get("max_new_tokens", 256)

        input_ids = glm_tokenizer([prompt], return_tensors="pt").to(device)
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(
            target=model.generate,
            kwargs=dict(
                **input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                streamer=streamer,
            ),
        )
        thread.start()
        for token_id in streamer:
            yield token_id

    # 收集响应标记
    response_tokens = []
    try:
        for token_id in generate_stream(params):
            response_tokens.append(token_id)
    except Exception as e:
        raise Exception(f"Error during generation: {e}")

    # 处理响应标记
    audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
    end_token_id = glm_tokenizer.convert_tokens_to_ids("<|user|>")
    text_tokens = []
    audio_tokens = []
    is_finalize = False
    this_uuid = str(uuid.uuid4())
    prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
    for token_id in response_tokens:
        if token_id == end_token_id:
            is_finalize = True
        if not is_finalize:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)

    # 使用 audio_decoder 获取音频数据
    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
    tts_speech, _ = audio_decoder.token2wav(
        tts_token,
        uuid=this_uuid,
        prompt_token=flow_prompt_speech_token.to(device),
        prompt_feat=prompt_speech_feat.to(device),
        finalize=is_finalize,
    )
    # 将音频数据转换为字节
    result = io.BytesIO()
    torchaudio.save(result, tts_speech.cpu(), 22050, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global model, glm_tokenizer, audio_decoder, whisper_model, feature_extractor, device
    device = "cuda"

    # 初始化模型路径
    model_path = "THUDM/glm-4-voice-9b"
    tokenizer_path = "THUDM/glm-4-voice-tokenizer"
    flow_path = "./glm-4-voice-decoder"

    # 初始化 GLM 模型和分词器
    glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
    )

    # 初始化 AudioDecoder
    flow_config = os.path.join(flow_path, "config.yaml")
    flow_checkpoint = os.path.join(flow_path, "flow.pt")
    hift_checkpoint = os.path.join(flow_path, "hift.pt")

    audio_decoder = AudioDecoder(
        config_path=flow_config,
        flow_ckpt_path=flow_checkpoint,
        hift_ckpt_path=hift_checkpoint,
        device=torch.device(f"cuda:{device}") if type(device) == int else device,
    )

    # 初始化语音分词器
    whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)


def hangup():
    global model, glm_tokenizer, audio_decoder, whisper_model, feature_extractor
    del model
    del glm_tokenizer
    del audio_decoder
    del whisper_model
    del feature_extractor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    setproctitle.setproctitle("glm4voice-web-inference")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="glm4voice")
    parser.add_argument("--hangup-timeout-sec", type=int, default=900)
    parser.add_argument("--hangup-interval-sec", type=int, default=60)
    args = parser.parse_args()
    expose(
        args.api_name,
        inference,
        port=args.port,
        hangup_timeout_sec=args.hangup_timeout_sec,
        hangup_interval_sec=args.hangup_interval_sec,
        init_function=init,
        hangup_function=hangup,
    )
