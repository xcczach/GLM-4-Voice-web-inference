# GLM-4-Voice
Read this in [English](./README_en.md)

GLM-4-Voice 是智谱 AI 推出的端到端语音模型。GLM-4-Voice 能够直接理解和生成中英文语音，进行实时语音对话，并且能够根据用户的指令改变语音的情感、语调、语速、方言等属性。

## Model Architecture
![Model Architecture](./resources/architecture.jpeg)

GLM-4-Voice 由三个部分组成：
* GLM-4-Voice-Tokenizer: 通过在 [Whisper](https://github.com/openai/whisper) 的 Encoder 部分增加 Vector Quantization 并在 ASR 数据上有监督训练，将连续的语音输入转化为离散的 token。每秒音频平均只需要用 12.5 个离散 token 表示。
* GLM-4-Voice-9B: 在 [GLM-4-9B](https://github.com/THUDM/GLM-4) 的基础上进行语音模态的预训练和对齐，从而能够理解和生成离散化的语音 token。
* GLM-4-Voice-Decoder: 基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的 Flow Matching 模型结构训练的支持流式推理的语音解码器，将离散化的语音 token 转化为连续的语音输出。最少只需要 10 个语音 token 即可开始生成，降低端到端对话延迟。

更详细的技术报告将在之后公布。

## Model List

|         Model         | Type |                                                                     Download                                                                     |
|:---------------------:| :---: |:------------------------------------------------------------------------------------------------------------------------------------------------:|
| GLM-4-Voice-Tokenizer | Speech Tokenizer | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-voice-tokenizer) [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-voice-tokenizer) |
|    GLM-4-Voice-9B     | Chat Model |                                          [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-voice-9b) [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-voice-9b)                                           
| GLM-4-Voice-Decoder   | Speech Decoder |                                        [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-voice-decoder) [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-voice-decoder)                                        

## Usage
我们提供了可以直接启动的 Web Demo。用户可以输入语音或文本，模型会同时给出语音和文字回复。

![](resources/web_demo.png)

### Preparation
首先下载仓库
```shell
git clone --recurse-submodules https://github.com/THUDM/GLM-4-Voice
cd GLM-4-Voice
```
然后安装依赖。
```shell
pip install -r requirements.txt
```
由于 Decoder 模型不支持通过 `transformers` 初始化，因此 checkpoint 需要单独下载。

```shell
# git 模型下载，请确保已安装 git-lfs
git clone https://huggingface.co/THUDM/glm-4-voice-decoder
```

### Launch Web Demo
首先启动模型服务
```shell
python model_server.py --model-path glm-4-voice-9b
```
此命令会自动下载 `glm-4-voice-9b`。如果网络条件不好，也手动下载之后通过 `--model-path` 指定本地的路径。

然后启动 web 服务
```shell
python web_demo.py
```
即可在 http://127.0.0.1:8888 访问 web demo。此命令会自动下载 `glm-4-voice-tokenizer` 和 `glm-4-voice-9b`。如果网络条件不好，也可以手动下载之后通过 `--tokenizer-path` 和 `--model-path` 指定本地的路径。

### Known Issues
* Gradio 的流式音频播放效果不稳定。在生成完成后点击对话框中的音频质量会更高。

## Cases
我们提供了 GLM-4-Voice 的部分对话案例，包括控制情绪、改变语速、生成方言等。

* 用轻柔的声音引导我放松

https://github.com/user-attachments/assets/4e3d9200-076d-4c28-a641-99df3af38eb0

* 用激动的声音解说足球比赛

https://github.com/user-attachments/assets/0163de2d-e876-4999-b1bc-bbfa364b799b

* 用哀怨的声音讲一个鬼故事

https://github.com/user-attachments/assets/a75b2087-d7bc-49fa-a0c5-e8c99935b39a

* 用东北话介绍一下冬天有多冷

https://github.com/user-attachments/assets/91ba54a1-8f5c-4cfe-8e87-16ed1ecf4037

* 用重庆话念“吃葡萄不吐葡萄皮”

https://github.com/user-attachments/assets/7eb72461-9e84-4d8e-9c58-1809cf6a8a9b

* 用北京话念一句绕口令

https://github.com/user-attachments/assets/a9bb223e-9c0a-440d-8537-0a7f16e31651

  * 加快语速

https://github.com/user-attachments/assets/c98a4604-366b-4304-917f-3c850a82fe9f

  * 再快一点

https://github.com/user-attachments/assets/d5ff0815-74f8-4738-b0f1-477cfc8dcc2d

## Acknowledge
本项目的部分代码来自：
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [GLM-4](https://github.com/THUDM/GLM-4)
