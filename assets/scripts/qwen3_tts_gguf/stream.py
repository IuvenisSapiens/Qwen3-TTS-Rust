"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from .constants import PROTOCOL, map_speaker, map_language
from .result import TTSResult, Timing, LoopOutput
from .config import TTSConfig
from .predictors.talker import TalkerPredictor
from .predictors.predictor import Predictor

from . import llama, logger
from .prompt_builder import PromptBuilder, PromptData
from .utils.audio import preprocess_audio, save_temp_wav

class TTSStream:
    """
    保存 Talker, Predictor, Decoder 记忆的语音流。
    """
    def __init__(self, engine, n_ctx=4096, voice_path: Optional[str] = None):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心 (Talker & Predictor)
        self.talker = TalkerPredictor(engine.talker_model, self.talker_ctx, self.talker_batch, self.assets)
        self.predictor = Predictor(engine.predictor_model, self.predictor_ctx, self.predictor_batch, self.assets)
        
        # 3. 音色锚点 (Voice)
        self.voice: Optional[TTSResult] = None
        if voice_path:
            self.set_voice(voice_path)
            
        self.decoder = getattr(engine, 'decoder', None)

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        # engine.talker_model 是 LlamaModel 对象
        self.talker_ctx = llama.LlamaContext(self.engine.talker_model, n_ctx=self.n_ctx, embeddings=True)
        self.predictor_ctx = llama.LlamaContext(self.engine.predictor_model, n_ctx=512, embeddings=False)
        
        self.talker_batch = llama.LlamaBatch(self.n_ctx, embd_dim=2048)
        self.predictor_batch = llama.LlamaBatch(32, embd_dim=1024)

    # =========================================================================
    # 核心推理 API
    # =========================================================================

    def clone(self, 
              text: str, 
              language: str = "chinese",
              config: Optional[TTSConfig] = None,
              streaming: bool = False,
              chunk_size: int = 25,
              verbose: bool = True) -> Optional[TTSResult]:
        """
        [克隆模式] 使用当前流中已设定的音色锚点（Voice Anchor）进行语音合成。

        Args:
            text: 待合成的目标文本。
            language: 目标语言。可选:
                - 'chinese' (默认), 'english', 'japanese', 'korean'
                - 'german', 'spanish', 'french', 'russian', 'italian', 'portuguese'
                - 'beijing_dialect' (北京话), 'sichuan_dialect' (四川话)
            config: 推理配置对象 (TTSConfig)，可控制 Temperature, Top-P 等采样参数。
            streaming: 是否启用流式推理。若为 True，则边推理边向播放器推送数据。
            chunk_size: 流式推理时，每积压多少帧特征码即送去解码播放一次。越小延迟越低，但每次解码会有8帧的额外计算。
            verbose: 是否输出详细推理进度和时延统计。

        Returns:
            TTSResult 对象，包含完整音频、特征码及性能统计。
        """
        if self.voice is None:
            print("[ERR] 尚未设定音色锚点，请先调用 set_voice()。")
            return None
            
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language)
            pdata = PromptBuilder.build_clone_prompt(text, self.tokenizer, self.assets, self.voice, lang_id)
            
            # DEBUG: Dump tensor stats for alignment
            import numpy as np
            embd_flat = pdata.embd.flatten()
            print(f"DEBUG_PY: Prompt embeddings: {pdata.embd.shape} tokens (Flattened: {len(embd_flat)})")
            print(f"DEBUG_PY: Flattened prompt head: {embd_flat[:10]}")
            print(f"DEBUG_PY: Flattened prompt tail: {embd_flat[-10:]}")
            print(f"DEBUG_PY: Flattened prompt sum: {np.sum(embd_flat)}")
            print(f"DEBUG_PY: Tokenizer ids: {len(pdata.text_ids)} tokens")
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            print(f"[ERR] Clone 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def custom(self,
               text: str,
               speaker: str,
               language: str = "chinese",
               instruct: Optional[str] = None,
               config: Optional[TTSConfig] = None,
               streaming: bool = False,
               chunk_size: int = 25,
               verbose: bool = True) -> Optional[TTSResult]:
        """
        [精品音色模式] 使用官方内置的精品预设音色进行合成，支持自然语言渲染指令。

        Args:
            text: 待合成的目标文本。
            speaker: 精品音色名称。可选:
                - 女性: 'vivian', 'serena', 'ono_anna', 'sohee'
                - 男性: 'ryan', 'aiden', 'uncle_fu'
                - 方言专用: 'eric' (四川话男声), 'dylan' (北京话男声)
            language: 目标语言 (见 clone 方法)。
            instruct: 渲染指令，如 "用温柔的语气说" 或 "充满活力的播报"。
            config: 推理配置对象 (TTSConfig)。
            streaming: 是否启用流式推理。
            chunk_size: 流式推理块大小。
            verbose: 是否输出详细进度。

        Returns:
            TTSResult 对象。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            spk_id = map_speaker(speaker)
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = PromptBuilder.build_custom_prompt(text, self.tokenizer, self.assets, spk_id, lang_id, instruct)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Custom 推理失败: {e}")
            return None

    def design(self,
               text: str,
               instruct: str,
               language: str = "chinese",
               config: Optional[TTSConfig] = None,
               streaming: bool = False,
               chunk_size: int = 25,
               verbose: bool = True) -> Optional[TTSResult]:
        """
        [音色设计模式] 完全通过自然语言描述来设计并生成一个全新的音色。

        Args:
            text: 待合成的目标文本。
            instruct: 音色设计描述。例如："体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。"
            language: 目标语言 (见 clone 方法)。
            config: 推理配置对象 (TTSConfig)。
            streaming: 是否启用流式推理。
            chunk_size: 流式推理块大小。
            verbose: 是否输出详细进度。

        Returns:
            TTSResult 对象。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        try:
            lang_id = map_language(language) if language.lower() != "auto" else None
            pdata = PromptBuilder.build_design_prompt(text, self.tokenizer, self.assets, instruct, lang_id)
            
            timing = Timing()
            timing.prompt_time = pdata.compile_time
            
            lout = self._run_engine_loop(pdata, timing, cfg, streaming=streaming, chunk_size=chunk_size, verbose=verbose)
            return self._post_process(text, pdata, lout)
        except Exception as e:
            logger.error(f"❌ Design 推理失败: {e}")
            return None

    def tts(self, *args, **kwargs):
        return self.clone(*args, **kwargs)

    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig, 
                         streaming: bool = False, chunk_size: int = 25, verbose: bool = False) -> LoopOutput:
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        
        if self.decoder:
            pass # 信任 Session 自动清理逻辑
            
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            all_codes.append(step_codes) # 保持 numpy 状态，供 decoder 使用
            turn_summed_embeds.append(summed_vec)
            
            if streaming and self.decoder:
                chunk_buffer.append(step_codes)
                if len(chunk_buffer) >= chunk_size:
                    self.decoder.decode(np.array(chunk_buffer), is_final=False, stream=True)
                    chunk_buffer = []

        if streaming and self.decoder:
            self.decoder.decode(np.array(chunk_buffer) if chunk_buffer else np.zeros((0, 16)), is_final=True, stream=True)

        return LoopOutput(all_codes=all_codes, summed_embeds=turn_summed_embeds, timing=timing)

    def _create_sampler(self, do_sample: bool, temperature: float, top_p: float, top_k: int) -> llama.LlamaSampler:
        """创建原生采样器实例"""
        if do_sample:
            return llama.LlamaSampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=None
            )
        else:
            return llama.LlamaSampler(temperature=0)

    def _run_engine_loop_gen(self, pdata: PromptData, cfg: TTSConfig, timing: Timing):
        t_pre_s = time.time()
        m_hidden = self.talker.prefill(pdata.embd, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        # 1. 初始化两级原生采样器
        talker_sampler = self._create_sampler(cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k)
        predictor_sampler = self._create_sampler(cfg.sub_do_sample, cfg.sub_temperature, cfg.sub_top_p, cfg.sub_top_k)
            
        step_idx = 0
        try:
            for step_idx in range(cfg.max_steps):
                # DEBUG: Print m_hidden sum (from Talker)
                import numpy as np
                print(f"DEBUG_PY: Step {step_idx} m_hidden sum: {np.sum(m_hidden):.5f}")
                # Print first 5 values for detailed comparison
                print(f"DEBUG_PY: Step {step_idx} m_hidden[0..5]: {m_hidden[:5]}")
                
                # ---------------- Talker Stage ----------------
                code_0 = talker_sampler.sample(self.talker.ctx)
                print(f"DEBUG_PY: Step {step_idx} code_0: {code_0}")
                
                if code_0 == PROTOCOL["EOS"]:
                    print(f"DEBUG_PY: Step {step_idx} EOS reached at {code_0}")
                    break
                
                t_c_s = time.time()
                
                # ---------------- Predictor Stage ----------------
                # 显式传递复用的采样器
                step_codes, step_embeds_2048 = self.predictor.predict_frame(
                    m_hidden, 
                    code_0, 
                    sampler=predictor_sampler
                )
                
                if step_idx == 0:
                   # DEBUG_PY: Print Top 10 Logits
                   logits_ptr = self.talker.ctx.get_logits()
                   n_vocab = 152064 # Default Qwen3 vocab
                   try:
                        n_vocab = llama.llama_vocab_n_tokens(self.talker.model.vocab)
                   except:
                        pass
                   
                   # Create numpy view (copy to be safe)
                   logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
                   
                   
                   # Create numpy view (copy to be safe) - ASSUME DENSE for testing
                   # If dense, size is n_tokens * n_vocab = 111 * 3072 = 340992
                   # We need to be careful not to Segfault if sparse.
                   # Let's try to access the last token's position: 110.
                   last_idx = 110 # Hardcoded for this prompt
                   
                   try:
                       # Offset pointer
                       import ctypes
                       logit_type = ctypes.c_float
                       
                       # Index 0
                       logits_0 = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
                       top_k_0 = np.argsort(logits_0)[-5:][::-1]
                       print(f"DEBUG_PY: Step 0 Top 5 Logits (Index 0):")
                       for idx in top_k_0:
                           print(f"        Code {idx}: {logits_0[idx]:.5f}")

                       # Index 110
                       # Pointer arithmetic: ptr + (last_idx * n_vocab)
                       # Cast to void_p, add bytes? No, ctypes pointer arithmetic works by element size.
                       offset_ptr = ctypes.cast(
                           ctypes.addressof(logits_ptr.contents) + (last_idx * n_vocab * ctypes.sizeof(logit_type)),
                           ctypes.POINTER(logit_type)
                       )
                       logits_end = np.ctypeslib.as_array(offset_ptr, shape=(n_vocab,)).copy()
                       top_k_end = np.argsort(logits_end)[-5:][::-1]
                       print(f"DEBUG_PY: Step 0 Top 5 Logits (Index {last_idx}):")
                       for idx in top_k_end:
                           print(f"        Code {idx}: {logits_end[idx]:.5f}")
                   except Exception as e:
                       print(f"DEBUG_PY: Failed to access Index {last_idx}: {e}")

                   print(f"DEBUG_PY: Step 0 Codes[1..8]: {step_codes[:8]}")
                timing.predictor_loop_time += (time.time() - t_c_s)
                
                # DEBUG: Print step_embeds_2048 sum
                embeds_sum = np.sum(np.sum(step_embeds_2048, axis=0))
                print(f"DEBUG_PY: Step {step_idx} step_embeds_2048 total sum: {embeds_sum:.5f}")
                
                t_m_s = time.time()
                summed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
                
                print(f"DEBUG_PY: Step {step_idx} feedback sum: {np.sum(summed):.5f}")
                
                m_hidden = self.talker.decode_step(summed, seq_id=0)
                timing.talker_loop_time += (time.time() - t_m_s)
                
                yield step_codes, summed
                
        finally:
            talker_sampler.free()
            predictor_sampler.free()
            
        timing.total_steps = len(pdata.embd) + step_idx

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput) -> TTSResult:
        audio = None
        if self.decoder:
            t0 = time.time()
            audio = self.decoder.decode(np.array(lout.all_codes), is_final=True, stream=False)
            lout.timing.decoder_render_time = time.time() - t0

        return TTSResult(
            audio=audio,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(lout.all_codes),
            summed_embeds=lout.summed_embeds,
            stats=lout.timing
        )

    def reset(self):
        self.talker_ctx.clear_kv_cache()
        self.predictor_ctx.clear_kv_cache()
        self.voice = None
        logger.info("扫 [Stream] 记忆与音色已清除。")

    def join(self, timeout: Optional[float] = None):
        """阻塞直至当前流所有音频（解码+播报）全部完毕"""
        if self.decoder:
            # 1. 先等解码器把活干完 (Bitstream -> PCM)
            self.decoder.join_decoder(timeout)
            # 2. 再等播放器把声音放完 (PCM -> 声卡)
            self.decoder.join_speaker(timeout)

    def shutdown(self):
        # 让 Python GC 处理内存释放 (_del_ 会调用 free)
        self.talker_batch = None
        self.talker_ctx = None
        self.predictor_batch = None
        self.predictor_ctx = None

    # =========================================================================
    # 音色设置 API (Voice Management)
    # =========================================================================

    def set_voice(self, source: Union[TTSResult, str, Path], text: Optional[str] = None, **kwargs) -> Union[bool, TTSResult]:
        """统一设置当前流的音色锚点。返回生成的 TTSResult 或 False。"""
        try:
            success = False
            if isinstance(source, TTSResult):
                success = self._set_voice_from_result(source)
            else:
                source_p = Path(source)
                if source_p.suffix.lower() == ".json":
                    success = self._set_voice_from_json(source_p)
                elif source_p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".opus"]:
                    success = self._set_voice_from_audio(source_p, text or "", **kwargs)
                else:
                    # 尝试作为内置说话人处理
                    return self.set_voice_from_speaker(str(source), text or "你好", **kwargs)
            
            return self.voice if success else False
        except Exception as e:
            print(f"[ERR] 设置音色时出现无法预料的异常: {e}")
            return False

    def _set_voice_from_result(self, res: TTSResult) -> bool:
        if not res.is_valid_anchor:
            print("[ERR] 提供的 TTSResult 不是有效的音色锚点 (缺少 codes 或 spk_emb)。")
            return False
        self.voice = res
        print(f"[Voice] 音色已切换为: {res.text[:20]}...")
        return True

    def _set_voice_from_json(self, path: Path) -> bool:
        """从 JSON 文件恢复音色锚点"""
        if not path.exists():
            logger.error(f"❌ 未找到音色 JSON 文件: {path}")
            return False
        try:
            res = TTSResult.from_json(str(path))
            return self._set_voice_from_result(res)
        except Exception as e:
            print(f"[ERR] 解析音色 JSON 失败 ({path.name}): {e}")
            return False

    def _set_voice_from_audio(self, wav_path: Path, text: str) -> bool:
        """从音频文件克隆音色：使用 pydub 标准化输入"""
        if self.engine.encoder is None:
            print("[ERR] 编码器模块未加载，无法执行音色克隆。")
            return False
            
        print(f"[Voice] 正在从音频提取音色特征: {wav_path.name}")
        
        # 1. 万能格式转换与预处理
        samples = preprocess_audio(wav_path)
        if samples is None:
            return False
            
        # 2. 交互过渡: 编码器目前只接受文件路径，因此我们需要一个临时 wav
        temp_wav = save_temp_wav(samples)
        try:
            codes, spk_emb = self.engine.encoder.encode(temp_wav)
            res = TTSResult(text=text, text_ids=self.tokenizer.encode(text).ids, spk_emb=spk_emb, codes=codes.tolist())
            return self._set_voice_from_result(res)
        except Exception as e:
            print(f"[ERR] 音声特征提取失败: {e}")
            return False
        finally:
            if os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

    def set_voice_from_speaker(self, speaker_id: str, text: str, **kwargs) -> Optional[TTSResult]:
        """从内置说话人生成音色锚点并设置"""
        try:
            logger.info(f"📍 正在从内置说话人初始化音色核心: {speaker_id}")
            # kwargs 包含 language, streaming, verbose 等
            res = self.custom(text, speaker_id, **kwargs)
            if res:
                self._set_voice_from_result(res)
                return res
            return None
        except Exception as e:
            print(f"[ERR] 内置音色初始化失败: {e}")
            return None
