# Elio Agent v3

本地运行的具身桌面智能体。Llama 当"单刻大脑"理解此刻屏幕，可训练的 TTT 海马体当"记忆"——通过持续预测下一刻，预测错了就实时改写记忆权重，长期内化你的电脑使用习惯。

## 项目结构

```
Elio_Agent_v3/
├── elio/                         # 数据采集 + 预处理
│   ├── preprocess.py             # 6-step 预处理流水线 → 12 .npy/session
│   ├── record.py                 # 录屏 + 音频 + 键鼠事件
│   ├── validate.py               # 数据校验
│   ├── verify_env.py             # 环境自检
│   └── download.py               # 下载模型权重
├── trainer/                      # 训练骨架 (云端 meta-training)
│   ├── dataset.py                # K 帧 Dataset (mmap 懒加载)
│   ├── model.py                  # ElioModel: AttentionPool + Projector → inputs_embeds
│   ├── heads.py                  # IntentPool + 5 预测头 + loss
│   ├── verify_forward.py         # 端到端前向验证 (Step 1-4)
│   └── train.py                  # (待实现) TBPTT 训练循环
├── models/                       # 冻结权重 (~5GB)
│   ├── Llama-3.2-1B-Instruct-abliterated/
│   ├── siglip_base/
│   └── dinov2_base/
├── data/
│   ├── raw/                      # 原始录屏 session_*/
│   │   ├── video.mp4
│   │   └── events.jsonl
│   └── processed/                # 预处理输出 session_*/processed/
│       ├── spec.json             # 元数据 (N, screen_w/h, 各文件 shape)
│       ├── visual_siglip.npy     # [N, 196, 768] float16
│       ├── visual_dinov2.npy     # [N, 257, 768] float16
│       ├── visual_siglip_fovea.npy  # [N, 196, 768] float16
│       ├── visual_dinov2_fovea.npy  # [N, 257, 768] float16
│       ├── audio_clap.npy        # [N, 512] float16
│       ├── actions.npy           # [N, 178] float16
│       ├── gaze_pseudo.npy       # [N, 2] float32
│       ├── events_flat.npy       # [M, 10] float32 (CSR)
│       ├── events_offsets.npy    # [N+1] int64
│       ├── frame_targets_siglip.npy  # [N, 196, 768]
│       └── frame_targets_dinov2.npy  # [N, 257, 768]
└── scripts/                      # 辅助脚本
```

## 架构：单 tick 数据流

```
[1] 感知 (冻结, 离线预计算)
    z_siglip  = SigLIP(frame)              # [196, 768]
    z_dinov2  = DINOv2(frame)              # [257, 768]
    z_sig_fov = SigLIP(crop@gaze)          # [196, 768]  中央凹
    z_din_fov = DINOv2(crop@gaze)          # [257, 768]  中央凹
    z_audio   = CLAP(audio_chunk)          # [512]
    z_action  = actions snapshot           # [178]

[2] 压缩 + 投影 (慢权重, 可训练)
    AttentionPool(concat(z_siglip, z_dinov2))  → [16, 768]  全屏 453→16
    AttentionPool(concat(z_sig_fov, z_din_fov)) → [16, 768]  中央凹 453→16
    Projector(16×768)                           → [16, 2048]
    Projector(audio)                            → [1, 2048]
    Projector(action)                           → [1, 2048]

[3] 拼帧窗口
    tokens_per_frame = 1(audio) + 16(full) + 16(fovea) + 1(action) = 34
    inputs_embeds = [B, K×34, 2048]    (K=4 → 136 tokens)

[4] Llama 推理 (冻结, bfloat16/4bit)
    h_t = Llama(inputs_embeds=inputs_embeds)   # [B, 136, 2048]

[5] IntentPool → 5 意图 (慢权重)
    h_frames = h_t.view(B, K, 34, 2048)
    5 learnable queries 对每帧 34 个 h 做 cross-attn
    → intents [B, K, 5, 2048]

[6] 5 个预测头 (慢权重, 各吃专属 intent)
    audio_head:   [B,K,2048] → [B,K,512]    预测下帧 CLAP
    gaze_head:    [B,K,2048] → [B,K,2]      预测下帧注视坐标
    frame_head:   [B,K,2048] → [B,K,196,768] + [B,K,257,768]  预测下帧 patch 残差
    kb_head:      [B*K,2048] → autoregressive → 键盘事件序列
    mouse_head:   [B*K,2048] → autoregressive → 鼠标事件序列
```

## 关键设计

### 视觉：AttentionPool 压缩

SigLIP (196 patch) + DINOv2 (257 patch) = **453 token** 进 Llama 太贵。16 个可学习 query 通过 cross-attention 压缩到 16 token，全屏/中央凹各一路不共享 query。`tokens_per_frame = 1 + 16 + 16 + 1 = 34`，K=4 帧 = 136 token，远在 Llama 2048 窗口内。

### 事件：CSR + 自回归头

键鼠事件变长序列 → CSR 格式 (`events_flat [M,10]` + `events_offsets [N+1]`)。每事件 10 维编码：
`[type_id, key_id, button_id, x, y, dx, dy, path_len, scroll_dy, dt_ms]`

两个 decoder-only transformer 头（键盘/鼠标各一），teacher forcing 展开 `[BOS, ev0, ..., ev_{n-1}] → [ev0, ..., ev_{n-1}, EOS]`。字段有效性 mask 按 type 屏蔽无关字段 loss（如 move 事件无 key_id）。

### 画面预测：完整 patch 残差

预测 Δz = z_{t+1} - z_t（不是绝对 z，避免平凡解）。两个独立子头（SigLIP 196×768 + DINOv2 257×768，latent 空间不同）。变化区加权：`weight = 1 + |true_Δz|`，强制模型关注动的区域。仅全屏路（焦点路内容随 gaze 跳，无因果意义）。

### 总 Loss：detach 归一化

```
L = L_audio/L_audio.detach() + L_frame/L_frame.detach()
  + L_gaze/L_gaze.detach()   + L_kb/L_kb.detach()
  + L_mouse/L_mouse.detach()
```

每项值恒≈1，5 头梯度自动同量级。日志单独打印原始 loss 看真实收敛。

### inputs_embeds 机制

投影层的 2048-dim 输出通过 `llama(inputs_embeds=...)` 直接替代 Llama 的词嵌入表，绕过 tokenizer。梯度穿过 Llama 流回慢权重，但 Llama 参数 `requires_grad=False` 不更新。

## 组件归属

| 冻结 | 慢权重 (云端学) |
|------|----------------|
| Llama-3.2-1B (bf16/4bit) | ElioModel: AttentionPool ×2 + Projector ×4 |
| SigLIP, DINOv2 | IntentPool (5 query + cross-attn) |
| CLAP (音频编码) | audio_head, gaze_head, frame_head |
| | AutoregHead ×2 (键盘/鼠标, 含 EventEmbedder) |

慢权重总计 **~33.6M** 参数。快权重 (W_fast) 待 Step 5 插入。

## 已验证 (Step 1-4)

```
python -m trainer.verify_forward --batch-size 1 --K 4
```

| 检查项 | 状态 |
|--------|------|
| Dataset K-frame 窗口 + mmap 懒加载 | ✅ |
| ElioModel → inputs_embeds [B,136,2048] | ✅ |
| Llama → h_t [B,136,2048] | ✅ |
| IntentPool → intents [B,K,5,2048] | ✅ |
| audio/gaze/frame 头 shape + loss | ✅ |
| kb/mouse AR 头 teacher forcing + field_mask + 空帧 | ✅ |
| 5 头 detach 归一化 backward | ✅ |
| ElioModel 45 params grad ≠ 0 | ✅ |
| Llama 146 params grad = None | ✅ |
| 显存峰值 9.95GB (bf16) | ✅ |

## 实施路线

| 步骤 | 内容 | 状态 |
|------|------|------|
| preprocess | 6-step 流水线 → 12 .npy/session | ✅ 完成 |
| Step 1-3 | Dataset + ElioModel + Llama forward | ✅ 完成 |
| Step 4 | IntentPool + 5 预测头 + total_loss | ✅ 完成 |
| Step 5 | TTT 内层循环 + TBPTT 外层回传 | 待实现 |
| Step 6 | 训练循环 + checkpoint | 待实现 |
| Step 7 | 本地部署闭环 | 待实现 |
