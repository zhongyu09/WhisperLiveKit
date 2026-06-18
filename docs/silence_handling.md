# 静音处理与断句机制详解

本文档基于 `main` 分支代码，梳理 WhisperLiveKit 中**静音（silence）如何触发断句、控制 ASR 解码上下文重置、以及输出静音段**，并对比两种常见启动配置的差异：

- `--backend faster-whisper --backend-policy simulstreaming`
- `--backend qwen3-vllm`（即 in-process vLLM 的 Qwen3-ASR 实时后端，俗称 "vllm-realtime"）

> 注意：CLI 的 `--backend` 没有 `vllm-realtime` 这个值，可选项见 `whisperlivekit/parse_args.py`，实际用 `qwen3-vllm` / `qwen3-vllm-metal`。

---

## 0. 公共前提：静音事件由 VAD 产生

无论哪种 backend，"何时存在一个静音事件"都由 Silero VAD 决定，默认 `min_silence_duration_ms=100ms`：

```197:200:whisperlivekit/silero_vad_iterator.py
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
```

也就是说，停顿 ≥ ~100ms 才会被上报，进而在 `audio_processor` 里触发 `_begin_silence` / `_end_silence`：

```804:825:whisperlivekit/audio_processor.py
            if "start" in event and self.current_silence:
                ...
                await self._end_silence(at_sample=start_sample_eff)
                last_offset = start_offset

            if "end" in event and not self.current_silence:
                ...
                await self._begin_silence(at_sample=end_sample_eff)
                last_offset = end_offset
```

下文的所有差异，都是"拿到静音事件之后"的处理。

---

## 1. 静音触发断句（输出分行）—— 两种配置相同

断句逻辑在 `audio_processor` 与 formatter 中，**与 backend 无关**。

第一步：`_end_silence` 计算静音时长，只有 `duration > MIN_DURATION_REAL_SILENCE` 时才把 `Silence` 放进 `new_tokens`：

```186:189:whisperlivekit/audio_processor.py
        if self.current_silence.duration is not None:
            self.metrics.total_silence_duration_s += self.current_silence.duration
        if self.current_silence.duration and self.current_silence.duration > MIN_DURATION_REAL_SILENCE:
            self.state.new_tokens.append(self.current_silence)
```

```26:26:whisperlivekit/audio_processor.py
MIN_DURATION_REAL_SILENCE = 0.5
```

第二步：formatter 一旦在 `new_tokens` 中遇到 `Silence`，就把当前累积的词收尾成一行（Segment），并另起新行：

```233:248:whisperlivekit/tokens_alignment.py
            for token in self.new_tokens:
                if isinstance(token, Silence):
                    if self.current_line_tokens:
                        self.validated_segments.append(Segment.from_tokens(self.current_line_tokens))
                        self.current_line_tokens = []

                    end_silence = token.end if token.has_ended else _silence_now
                    if self.validated_segments and self.validated_segments[-1].is_silence():
                        self.validated_segments[-1].end = end_silence
                    else:
                        self.validated_segments.append(SilentSegment(
                            start=token.start,
                            end=end_silence
                        ))
                else:
                    self.current_line_tokens.append(token)
```

> **结论**：`> 0.5s` 的停顿即断句，这条规则对 simulstreaming 和 qwen3-vllm 完全一致，由 `audio_processor.py:26` 唯一控制。

---

## 2. 输出静音段（SilentSegment）—— 两种配置相同

如上所示，断句与"生成可见静音段"是**同一机制的副产物**：只要 `Silence` 进入 `new_tokens`（即 `duration > 0.5s`），formatter 就会追加一个 `SilentSegment`。两种 backend 无差异。

> 无法只要断句、不要静音段——两者由同一处触发。如确需分离，要改 formatter 逻辑。

---

## 3. ASR 解码上下文是否/何时重置 —— 核心差异

这是两种配置**最不同**的地方。转写线程在取到静音事件时分别调用 `start_silence()`（静音开始）与 `end_silence(duration)`（静音结束）：

```388:401:whisperlivekit/audio_processor.py
                if isinstance(item, Silence):
                    if item.is_starting:
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(
                            self.transcription.start_silence
                        )
                        asr_processing_logs += " + Silence starting"
                    if item.has_ended:
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                        cumulative_pcm_duration_stream_time += item.duration
                        current_audio_processed_upto = cumulative_pcm_duration_stream_time
                        self.transcription.end_silence(item.duration, self.state.tokens[-1].end if self.state.tokens else 0)
```

### 3.1 faster-whisper + simulstreaming（`simul_whisper/backend.py`）

重置发生在**静音结束时** `end_silence(duration)`，且**按时长分流**：

```77:93:whisperlivekit/simul_whisper/backend.py
    def end_silence(self, silence_duration, offset):
        """Handle silence period."""
        self.end += silence_duration
        long_silence = silence_duration >= MIN_DURATION_REAL_SILENCE
        if not long_silence:
            gap_len = int(16000 * silence_duration)
            if gap_len > 0:
                if self.asr.use_full_mlx:
                    gap_silence = np.zeros(gap_len, dtype=np.float32)
                else:
                    gap_silence = torch.zeros(gap_len)
                self.model.insert_audio(gap_silence)
        if long_silence:
            self.model.refresh_segment(complete=True)
            self.model.global_time_offset = silence_duration + offset
            self._last_committed_end = max(self._last_committed_end, self.model.global_time_offset)
            self._recent_words = []
```

```37:37:whisperlivekit/simul_whisper/backend.py
MIN_DURATION_REAL_SILENCE = 0.5
```

- **短静音（< 0.5s）**：插入等长静音样本，时间轴连续，**不重置**，AlignAtt 在同一上下文继续解码。
- **长静音（≥ 0.5s）**：`refresh_segment(complete=True)` + 重设 `global_time_offset` + 清空近期词历史 → 翻篇，下一句从干净上下文开始。

### 3.2 qwen3-vllm（`qwen3_vllm_asr.py`）

重置发生在**静音开始时** `start_silence()`，且**无条件、不看时长**：

```575:584:whisperlivekit/qwen3_vllm_asr.py
    def start_silence(self) -> Tuple[List[ASRToken], float]:
        tokens = self._drain_uncommitted(self._commit_available(flush=True))
        logger.info("[qwen3-vllm] start_silence: flushed %d words", len(tokens))
        self._reset_for_next_utterance()
        return tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self._buffer_time_offset += silence_duration
        self._last_committed_time += silence_duration
        self.end += silence_duration
```

```568:573:whisperlivekit/qwen3_vllm_asr.py
    def _reset_for_next_utterance(self):
        self._buffer_time_offset += self._audio_duration()
        self._last_committed_time = self._buffer_time_offset
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._current_tokens = []
```

- 任何 VAD 检测到的停顿（≥ ~100ms）**一开始**就 `flush` 当前内容 + `_reset_for_next_utterance()`（清空 buffer、重置时间）。
- `end_silence` 内**没有任何阈值判断**，只把时间偏移往后推。
- 没有"短静音插零保持连续"这个概念。

> **结论**：qwen3-vllm 的上下文重置比 simul **激进得多**——每个 ≥100ms 的停顿都会切成新 utterance（哪怕只有 0.2s，不会断句但内部已重置）；而 simul 对 0.1~0.5s 的小停顿是"插零续上"、不重置，只有 ≥0.5s 才重置。

---

## 4. pending 词的提交时机

- **simulstreaming**：静音开始 → `start_silence()` → `process_iter(is_last=True)` 强制 flush 已对齐的词。
- **qwen3-vllm**：静音开始 → `start_silence()` → `_commit_available(flush=True)` flush。

两者都是"停顿一开始就把攒着的词吐出来"，行为相近。

---

## 5. 稳定性保护 / 防幻觉

- **simulstreaming**：**有**专门 guard——`_filter_stable_words`（丢弃倒带/陈旧/非法时间戳的词）、rewind 重置、重复循环检测 `_has_repetition_loop`。AlignAtt 注意力对齐容易倒带/循环，需要这些保护。典型日志 `all emitted words rewound behind committed time ...` 即来自此处。

```240:248:whisperlivekit/simul_whisper/backend.py
            stable_words = self._filter_stable_words(timestamped_words)
            if not stable_words:
                max_end = max(float(token.end or 0.0) for token in timestamped_words)
                if self._last_committed_end - max_end > self._REWIND_RESET_SECONDS:
                    self._reset_after_unstable_output(
                        f"all emitted words rewound behind committed time "
                        f"{self._last_committed_end:.2f}s"
                    )
                self.buffer = []
                return [], self.end
```

- **qwen3-vllm**：**没有**这类 guard，靠节流参数控制——`_HOLDBACK_SECONDS=0.25`（尾部 250ms 不提交）、`_MIN_NEW_SECONDS=1.0`（攒满 1s 新音频才推理一次）、`_MAX_BUFFER_SECONDS=30`。

```439:443:whisperlivekit/qwen3_vllm_asr.py
    _HOLDBACK_SECONDS = 0.250
    _MIN_NEW_SECONDS = 1.0
    _MAX_BUFFER_SECONDS = 30.0
    _TRIM_BEFORE_COMMITTED_SECONDS = 2.0
    _COMMITTED_EPSILON = 0.05
```

---

## 6. 阈值常量分布（易踩坑）

`MIN_DURATION_REAL_SILENCE` 在代码里有**多份拷贝**，作用不同，改一处往往不够：

| 位置 | 值 | simulstreaming 是否生效 | qwen3-vllm 是否生效 | 作用 |
|---|---|---|---|---|
| `audio_processor.py:26` | 0.5 | ✅ | ✅ | **断句** + 输出静音段（与 backend 无关） |
| `simul_whisper/backend.py:37` | 0.5 | ✅ | ❌ | simulstreaming 的 **ASR 上下文重置**阈值 |
| `local_agreement/online_asr.py:11` | 0.5 | ❌ | ❌ | 仅 LocalAgreement 策略使用（两种配置都走不到） |

> qwen3-vllm 内部**不看静音时长**，只受 `audio_processor.py:26`（断句）影响。

---

## 7. 汇总对照表

| 维度 | faster-whisper + simulstreaming | qwen3-vllm |
|---|---|---|
| 断句阈值（分行） | > 0.5s（`audio_processor.py:26`） | > 0.5s（同，backend 无关） |
| 输出静音段 | 同断句机制，> 0.5s | 同断句机制，> 0.5s |
| ASR 上下文重置**时机** | 静音**结束**时 | 静音**开始**时 |
| ASR 重置**条件** | 仅 ≥ 0.5s（< 0.5s 插零续上） | **无条件**，任何检测到的停顿都重置 |
| 重置激进程度 | 较温和（小停顿保连续） | 激进（每个 ≥100ms 停顿都切新句） |
| pending 词提交 | 静音开始 flush | 静音开始 flush |
| 防幻觉/倒带 guard | 有（filter_stable/rewind/repetition） | 无，靠 holdback+min_new 节流 |
| 受静音阈值常量影响 | `audio_processor.py:26` + `simul_whisper/backend.py:37` | 仅 `audio_processor.py:26` |

---

## 8. 调参建议

- **只想改断句灵敏度**：改 `audio_processor.py:26`（对两种 backend 都生效）。
- **simulstreaming 下觉得上下文重置太频繁**（如 `all emitted words rewound...` 变多、偶发丢词）：把 `simul_whisper/backend.py:37` 单独调大（如 1.5~2s），它与断句互不影响。
- **改完务必重启 `wlk` 进程**：常量在 import 时读一次，热改源码不会生效。
- **确认线上加载的是哪份代码**（尤其非 editable 安装时）：

```bash
python -c "import whisperlivekit.audio_processor as a; print(a.__file__, a.MIN_DURATION_REAL_SILENCE)"
python -c "import whisperlivekit.simul_whisper.backend as b; print(b.__file__, b.MIN_DURATION_REAL_SILENCE)"
```

若打印出的路径在 `site-packages` 下且值仍是旧值，说明运行的是安装副本——需改该文件或在目标机 `pip install -e .`。
