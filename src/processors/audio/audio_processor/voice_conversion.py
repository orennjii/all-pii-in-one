"""
语音转换模块，基于SeedVC模型，提供零样本声音转换功能。
"""

# System modules
import os
import numpy as np
import yaml
from typing import Dict, Tuple, List, Union, Optional, Generator, Any

# External modules
import torch
import torchaudio
import librosa
from pydub import AudioSegment

# Internal modules
from src.modules.seed_vc_modules.commons import build_model, load_checkpoint, recursive_munch, str2bool
from .utils import load_custom_model_from_hf
from src.commons.device_config import get_device


class VoiceConverter:
    """
    语音转换类，提供基于SeedVC模型的声音转换功能。
    
    此类设计为易于集成到Gradio界面中，通过各种setter方法可以在界面上调整参数：
    - set_diffusion_steps: 设置扩散步数，越大质量越高但速度越慢
    - set_length_adjust: 调整输出音频的长度
    - set_inference_cfg_rate: 设置推理CFG率，控制音色与内容的平衡
    - set_auto_f0_adjust: 设置是否自动调整F0
    - set_pitch_shift: 调整音高
    - set_fp16: 设置是否使用半精度浮点数
    - set_max_context_window: 设置最大上下文窗口大小
    - set_bitrate: 设置输出音频的比特率
    - set_f0_condition: 设置是否使用F0条件（会重新加载模型）
    - set_model: 设置自定义模型路径
    
    使用示例:
        converter = VoiceConverter(device=torch.device("cuda:0"))
        # 设置参数
        converter.set_diffusion_steps(20)
        converter.set_length_adjust(1.0)
        # 转换音频
        converted_audio = converter.convert_voice("source.wav", "reference.wav")
        converter.save_converted_audio(converted_audio, "output.wav")
    """
    
    def __init__(
        self,
        device: torch.device = None,
    ):
        """
        初始化语音转换模型。

        参数:
            device: 运行设备，如果为None则自动选择
        """
        # 设置设备
        self.device = get_device() if device is None else device
        
        # 设置运行时参数
        self._overlap_frame_len = 16
        
        # 加载模型
        (
            self.model,
            self.semantic_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.to_mel,
            self.mel_fn_args
        ) = self._load_models()
        
        # 获取模型参数
        self.sr = self.mel_fn_args["sampling_rate"]
        self.hop_length = self.mel_fn_args["hop_size"]
        self.overlap_wave_len = self._overlap_frame_len * self.hop_length

    def _load_models(
        self, 
        checkpoint_path: Optional[str] = None, 
        config_path: Optional[str] = None,
        f0_condition: bool = True,
        fp16: bool = True,
        max_context_window: int = 8192
    ) -> Tuple:
        """
        加载转换所需的全部模型。
        
        参数:
            checkpoint_path: 模型检查点路径，如果为None则使用默认模型
            config_path: 配置文件路径，如果为None则使用默认配置
            f0_condition: 是否使用F0条件进行转换
            fp16: 是否使用半精度浮点数
            max_context_window: 最大上下文窗口大小
            
        返回:
            Tuple包含模型、语义函数、声码器等组件
        """
        # 检查是否使用F0条件
        if not f0_condition:
            if checkpoint_path is None:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
                    sub_dir="audio"
                )
            else:
                dit_checkpoint_path = checkpoint_path
                dit_config_path = config_path
            f0_fn = None
        else:
            if checkpoint_path is None:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
                    sub_dir="audio"
                )
            else:
                dit_checkpoint_path = checkpoint_path
                dit_config_path = config_path
                
            # F0提取器
            from src.modules.seed_vc_modules.rmvpe import RMVPE
            
            model_path = load_custom_model_from_hf(
                "lj1995/VoiceConversionWebUI", "rmvpe.pt", None, sub_dir="audio"
            )
            f0_extractor = RMVPE(model_path, is_half=fp16, device=self.device)
            f0_fn = f0_extractor.infer_from_audio

        # 加载配置和构建模型
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        sr = config["preprocess_params"]["sr"]

        # 加载检查点
        model, _, _, _ = load_checkpoint(
            model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        
        for key in model:
            model[key].eval()
            model[key].to(self.device)
            
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=max_context_window)

        # 加载CAMP+模型
        from src.modules.seed_vc_modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus",
            "campplus_cn_common.bin", 
            config_filename=None,
            sub_dir="audio"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(self.device)

        # 加载声码器
        vocoder_type = model_params.vocoder.type
        if vocoder_type == 'bigvgan':
            from src.modules.seed_vc_modules.bigvgan import bigvgan
            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            # 移除权重归一化并设置为评估模式
            bigvgan_model.remove_weight_norm()
            bigvgan_model = bigvgan_model.eval().to(self.device)
            vocoder_fn = bigvgan_model
        elif vocoder_type == 'hifigan':
            from src.modules.seed_vc_modules.hifigan.generator import HiFTGenerator
            from src.modules.seed_vc_modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            hift_path = load_custom_model_from_hf(
                "FunAudioLLM/CosyVoice-300M", 'hift.pt', None, sub_dir="audio"
            )
            hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            hift_gen.eval()
            hift_gen.to(self.device)
            vocoder_fn = hift_gen
        elif vocoder_type == "vocos":
            vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
            vocos_path = model_params.vocoder.vocos.path
            vocos_model_params = recursive_munch(vocos_config['model_params'])
            vocos = build_model(vocos_model_params, stage='mel_vocos')
            vocos_checkpoint_path = vocos_path
            vocos, _, _, _ = load_checkpoint(
                vocos, 
                None, 
                vocos_checkpoint_path,
                load_only_params=True, 
                ignore_modules=[], 
                is_distributed=False
            )
            _ = [vocos[key].eval().to(self.device) for key in vocos]
            _ = [vocos[key].to(self.device) for key in vocos]
            vocoder_fn = vocos.decoder
        else:
            raise ValueError(f"未知的声码器类型: {vocoder_type}")

        # 加载语音分词器
        speech_tokenizer_type = model_params.speech_tokenizer.type
        if speech_tokenizer_type == 'whisper':
            # 加载Whisper模型
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(
                whisper_name, 
                torch_dtype=torch.float16 if fp16 else torch.float32
            ).to(self.device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                       return_tensors="pt",
                                                       return_attention_mask=True)
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori
        elif speech_tokenizer_type == 'cnhubert':
            # 加载CNHubert模型
            from transformers import (
                Wav2Vec2FeatureExtractor,
                HubertModel,
            )
            hubert_model_name = config['model_params']['speech_tokenizer']['name']
            hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
            hubert_model = HubertModel.from_pretrained(hubert_model_name)
            hubert_model = hubert_model.to(self.device)
            hubert_model = hubert_model.eval()
            hubert_model = hubert_model.half() if fp16 else hubert_model

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy()
                    for bib in range(len(waves_16k))
                ]
                ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                      return_tensors="pt",
                                                      return_attention_mask=True,
                                                      padding=True,
                                                      sampling_rate=16000).to(self.device)
                with torch.no_grad():
                    ori_outputs = hubert_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        elif speech_tokenizer_type == 'xlsr':
            # 加载XLSR模型
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2Model,
            )
            model_name = config['model_params']['speech_tokenizer']['name']
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
            wav2vec_model = wav2vec_model.to(self.device)
            wav2vec_model = wav2vec_model.eval()
            wav2vec_model = wav2vec_model.half() if fp16 else wav2vec_model

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy()
                    for bib in range(len(waves_16k))
                ]
                ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                       return_tensors="pt",
                                                       return_attention_mask=True,
                                                       padding=True,
                                                       sampling_rate=16000).to(self.device)
                with torch.no_grad():
                    ori_outputs = wav2vec_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
            raise ValueError(f"未知的语音分词器类型: {speech_tokenizer_type}")

        # 设置梅尔频谱图生成函数
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        
        from src.modules.seed_vc_modules.audio import mel_spectrogram
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        return (
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
        )

    def _crossfade(self, chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
        """
        对两个音频片段进行交叉淡入淡出混合。
        
        参数:
            chunk1: 第一个音频片段
            chunk2: 第二个音频片段
            overlap: 重叠样本数
            
        返回:
            混合后的音频片段
        """
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice(
        self, 
        source_path: str, 
        target_path: str,
        diffusion_steps: int = 10,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        stream: bool = False,
        bitrate: str = "320k",
        max_context_window: int = 8192,
        fp16: bool = True,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0
    ) -> Union[np.ndarray, Generator[bytes, None, Tuple[int, np.ndarray]]]:
        """
        执行语音转换，将源语音转换为目标语音风格。
        
        参数:
            source_path: 源音频文件路径
            target_path: 目标参考音频文件路径
            diffusion_steps: 扩散步数
            length_adjust: 长度调整
            inference_cfg_rate: 推理CFG率
            stream: 是否使用流式处理（流式处理时返回生成器）
            bitrate: 输出音频的比特率（仅流式处理有效）
            max_context_window: 最大上下文窗口大小
            fp16: 是否使用半精度浮点数
            auto_f0_adjust: 是否自动调整F0
            pitch_shift: 音高偏移值
            
        返回:
            如果stream=False，返回转换后的音频数组
            如果stream=True，返回一个生成器，产生(mp3字节，转换进度)
        """
        
        # 加载音频
        source_audio = librosa.load(source_path, sr=self.sr)[0]
        ref_audio = librosa.load(target_path, sr=self.sr)[0]

        # 处理音频
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_audio[:self.sr * 25]).unsqueeze(0).float().to(self.device)

        # 重采样
        ref_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)
        
        # 如果源音频少于30秒，Whisper可以一次处理
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self.semantic_fn(converted_waves_16k)
        else:
            # 处理长音频，分块处理
            overlapping_time = 5  # 5秒
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:  # 第一个块
                    chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
                S_alt = self.semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        # 处理参考音频
        ori_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        S_ori = self.semantic_fn(ori_waves_16k)

        # 生成梅尔频谱图
        mel = self.to_mel(source_audio.to(self.device).float())
        mel2 = self.to_mel(ref_audio.to(self.device).float())

        # 设置目标长度
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # 提取参考音频特征
        feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                                  num_mel_bins=80,
                                                  dither=0,
                                                  sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        # F0设置
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

        # 长度调整
        cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
        prompt_condition, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

        # 设置处理窗口大小
        max_source_window = max_context_window - mel2.size(2)
        
        # 分块处理源条件并生成输出
        processed_frames = 0
        generated_wave_chunks = []
        
        # 如果不需要流式处理，收集所有结果
        if not stream:
            complete_output = []
            
        # 分块生成并流式输出
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if fp16 else torch.float32):
                # 语音转换
                vc_target = self.model.cfm.inference(cat_condition,
                                                     torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                     mel2, style2, None, diffusion_steps,
                                                     inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, mel2.size(-1):]
                
            # 生成波形
            vc_wave = self.vocoder_fn(vc_target.float())[0]
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)
            
            # 处理不同的块
            if processed_frames == 0:
                if is_last_chunk:
                    # 如果这是唯一一个块
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    if stream:
                        output_wave_int = (output_wave * 32768.0).astype(np.int16)
                        mp3_bytes = AudioSegment(
                            output_wave_int.tobytes(), frame_rate=self.sr,
                            sample_width=output_wave_int.dtype.itemsize, channels=1
                        ).export(format="mp3", bitrate=bitrate).read()
                        yield mp3_bytes, (self.sr, np.concatenate(generated_wave_chunks))
                    else:
                        complete_output = np.concatenate(generated_wave_chunks)
                    break
                else:
                    # 第一个块，但不是最后一个
                    output_wave = vc_wave[0, :-self.overlap_wave_len].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                    processed_frames += vc_target.size(2) - self._overlap_frame_len
                    
                    if stream:
                        output_wave_int = (output_wave * 32768.0).astype(np.int16)
                        mp3_bytes = AudioSegment(
                            output_wave_int.tobytes(), frame_rate=self.sr,
                            sample_width=output_wave_int.dtype.itemsize, channels=1
                        ).export(format="mp3", bitrate=bitrate).read()
                        yield mp3_bytes, None
            elif is_last_chunk:
                # 最后一个块，但不是第一个
                output_wave = self._crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), self.overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - self._overlap_frame_len
                
                if stream:
                    output_wave_int = (output_wave * 32768.0).astype(np.int16)
                    mp3_bytes = AudioSegment(
                        output_wave_int.tobytes(), frame_rate=self.sr,
                        sample_width=output_wave_int.dtype.itemsize, channels=1
                    ).export(format="mp3", bitrate=bitrate).read()
                    yield mp3_bytes, (self.sr, np.concatenate(generated_wave_chunks))
                else:
                    complete_output = np.concatenate(generated_wave_chunks)
                break
            else:
                # 中间块
                output_wave = self._crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-self.overlap_wave_len].cpu().numpy(), self.overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self._overlap_frame_len
                
                if stream:
                    output_wave_int = (output_wave * 32768.0).astype(np.int16)
                    mp3_bytes = AudioSegment(
                        output_wave_int.tobytes(), frame_rate=self.sr,
                        sample_width=output_wave_int.dtype.itemsize, channels=1
                    ).export(format="mp3", bitrate=bitrate).read()
                    yield mp3_bytes, None
        
        # 如果不是流式处理，返回完整音频
        if not stream:
            return complete_output

    def save_converted_audio(
        self, 
        converted_audio: np.ndarray, 
        output_path: str, 
        sample_rate: int = None
    ) -> str:
        """
        保存转换后的音频到文件。
        
        参数:
            converted_audio: 转换后的音频数组
            output_path: 输出文件路径
            sample_rate: 采样率，默认使用模型采样率
            
        返回:
            保存的文件路径
        """
        if sample_rate is None:
            sample_rate = self.sr
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 保存音频
        import soundfile as sf
        sf.write(output_path, converted_audio, sample_rate)
        
        return output_path

    def set_model(self, checkpoint_path: str, config_path: str) -> None:
        """
        设置自定义模型路径。
        
        参数:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
        """
        # 重新加载模型
        (
            self.model,
            self.semantic_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.to_mel,
            self.mel_fn_args
        ) = self._load_models(checkpoint_path, config_path)
        # 更新参数
        self.sr = self.mel_fn_args["sampling_rate"]
        self.hop_length = self.mel_fn_args["hop_size"]
        self.overlap_wave_len = self._overlap_frame_len * self.hop_length
        
    def set_model(self, checkpoint_path: str, config_path: str) -> None:
        """
        设置自定义模型路径。
        
        参数:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
        """
        # 重新加载模型
        (
            self.model,
            self.semantic_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.to_mel,
            self.mel_fn_args
        ) = self._load_models(checkpoint_path, config_path)
        # 更新参数
        self.sr = self.mel_fn_args["sampling_rate"]
        self.hop_length = self.mel_fn_args["hop_size"]
        self.overlap_wave_len = self._overlap_frame_len * self.hop_length

    # 这里不再使用property装饰器，这些参数都将直接通过方法参数传递
    def set_overlap_frame_len(self, value: int) -> None:
        """
        设置重叠帧长度，这是一个需要类管理的参数
        
        参数:
            value: 重叠帧长度
        """
        if value < 1:
            raise ValueError("重叠帧长度必须大于0")
        self._overlap_frame_len = value
        self.overlap_wave_len = self._overlap_frame_len * self.hop_length
        
    def reload_model_with_f0_condition(self, f0_condition: bool) -> None:
        """
        根据f0条件重新加载模型
        
        参数:
            f0_condition: 是否使用F0条件
        """
        # 重新加载模型
        (
            self.model,
            self.semantic_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.to_mel,
            self.mel_fn_args
        ) = self._load_models(f0_condition=f0_condition)
        # 更新参数
        self.sr = self.mel_fn_args["sampling_rate"]
        self.hop_length = self.mel_fn_args["hop_size"]
        self.overlap_wave_len = self._overlap_frame_len * self.hop_length
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        获取默认参数设置。
        
        返回:
            包含默认参数的字典
        """
        return {
            "diffusion_steps": 10,
            "length_adjust": 1.0,
            "inference_cfg_rate": 0.7,
            "f0_condition": True,
            "auto_f0_adjust": True,
            "default_pitch_shift": 0,
            "device": self.device,
            "fp16": True,
            "max_context_window": 8192,
            "overlap_frame_len": self._overlap_frame_len,
            "bitrate": "320k",
            "sample_rate": self.sr,
            "hop_length": self.hop_length,
        }

    def load_parameters_from_config(self, config_path: str) -> Dict[str, Any]:
        """
        从配置文件加载参数并返回参数字典。
        
        参数:
            config_path: 配置文件路径
            
        返回:
            Dict[str, Any]: 从配置文件加载的参数字典
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = self.get_default_parameters()
        voice_conversion_config = config.get('voice_conversion', {})
        
        # 从配置中更新参数
        if 'diffusion_steps' in voice_conversion_config:
            params['diffusion_steps'] = voice_conversion_config['diffusion_steps']
            
        if 'length_adjust' in voice_conversion_config:
            params['length_adjust'] = voice_conversion_config['length_adjust']
            
        if 'inference_cfg_rate' in voice_conversion_config:
            params['inference_cfg_rate'] = voice_conversion_config['inference_cfg_rate']
            
        if 'auto_f0_adjust' in voice_conversion_config:
            params['auto_f0_adjust'] = voice_conversion_config['auto_f0_adjust']
            
        if 'default_pitch_shift' in voice_conversion_config:
            params['default_pitch_shift'] = voice_conversion_config['default_pitch_shift']
            
        if 'fp16' in voice_conversion_config:
            params['fp16'] = voice_conversion_config['fp16']
            
        if 'max_context_window' in voice_conversion_config:
            params['max_context_window'] = voice_conversion_config['max_context_window']
            
        if 'overlap_frame_len' in voice_conversion_config:
            self.set_overlap_frame_len(voice_conversion_config['overlap_frame_len'])
            params['overlap_frame_len'] = self._overlap_frame_len
            
        if 'bitrate' in voice_conversion_config:
            params['bitrate'] = voice_conversion_config['bitrate']
            
        if 'model' in voice_conversion_config:
            model_config = voice_conversion_config['model']
            if 'checkpoint_path' in model_config and 'config_path' in model_config:
                self.set_model(model_config['checkpoint_path'], model_config['config_path'])
                
        return params

    def batch_convert(
        self, 
        source_paths: List[str], 
        target_dir: str,
        reference_voice: str,
        diffusion_steps: int = 10,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        bitrate: str = "320k", 
        max_context_window: int = 8192,
        **kwargs
    ) -> List[str]:
        """
        批量转换多个音频文件
        
        参数:
            source_paths: 源音频文件路径列表
            target_dir: 目标目录，用于保存转换后的文件
            reference_voice: 参考声音文件路径
            diffusion_steps: 扩散步数
            length_adjust: 长度调整
            inference_cfg_rate: 推理CFG率
            bitrate: 输出音频的比特率
            max_context_window: 最大上下文窗口大小
            **kwargs: 额外参数
            
        返回:
            List[str]: 转换后音频的文件路径列表
        """
        os.makedirs(target_dir, exist_ok=True)
        converted_paths = []
        
        print(f"开始批量转换 {len(source_paths)} 个音频文件...")
        for i, source_path in enumerate(source_paths):
            output_basename = os.path.basename(source_path)
            output_name = f"converted_{i:04d}_{output_basename}"
            output_path = os.path.join(target_dir, output_name)
            
            try:
                print(f"[{i+1}/{len(source_paths)}] 正在转换: {os.path.basename(source_path)}...")
                
                # 执行转换
                converted_audio = self.convert_voice(
                    source_path=source_path,
                    target_path=reference_voice,
                    diffusion_steps=diffusion_steps,
                    length_adjust=length_adjust,
                    inference_cfg_rate=inference_cfg_rate,
                    stream=False,
                    bitrate=bitrate,
                    max_context_window=max_context_window,
                    **kwargs
                )
                
                # 保存转换后的音频
                self.save_converted_audio(
                    converted_audio=converted_audio,
                    output_path=output_path
                )
                
                converted_paths.append(output_path)
                print(f"转换成功: {output_path}")
            except Exception as e:
                print(f"转换失败 {os.path.basename(source_path)}: {str(e)}")
                # 如果有问题，记录错误但继续处理其他文件
            
        print(f"批量转换完成，成功转换 {len(converted_paths)}/{len(source_paths)} 个文件")
        return converted_paths