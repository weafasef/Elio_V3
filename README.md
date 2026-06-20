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
│   ├── ttt.py                    # TTTMemory: W_fast 黑板记忆 (慢权重 θ_meta)
│   ├── loop.py                   # run_ttt_segment() 单帧记忆循环
│   ├── verify_forward.py         # 端到端前向验证 (Step 1-4 批处理模式)
│   ├── verify_ttt.py             # TTT 单帧循环 + 二阶 backward 验证 (Step 5)
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

## 架构：单刻处理器 + 记忆循环 (Step 5)

每帧 t 的数据流：

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

[3] 单帧编码
    ft = [audio 1][full 16][fovea 16][action 1] = 34 token [2048]

[4] READ: 用 ft 均值查 W_fast 黑板 (慢权重 q_proj/read_proj)
    pre_summary = ft.mean()               # [2048]
    q = q_proj(pre_summary)               # [256]
    m = W @ q                              # [256] 黑板读出
    mem_token = norm(read_proj(m))         # [1, 2048]

[5] 拼帧 + Llama 推理 (冻结, bf16)
    seq = [mem_token, ft]                  # [35, 2048]
    h = Llama(inputs_embeds=seq)           # [35, 2048]

[6] IntentPool → 5 意图 (慢权重)
    5 learnable queries × cross-attn over 35 token
    → intents [5, 2048]
    split → intent_a, intent_g, intent_f, intent_k, intent_m

[7] 5 个预测头预测帧 t+1 (慢权重)
    audio_head:   [2048] → [512]         预测下帧 CLAP → L_a
    gaze_head:    [2048] → [2]           预测下帧注视 → L_g
    frame_head:   [2048] → [196,768]+[257,768]  预测 patch Δz → L_f
    kb_head:      [2048] → autoregressive → 键盘事件序列 → L_k
    mouse_head:   [2048] → autoregressive → 鼠标事件序列 → L_m

[8] WRITE: 用 h 均值写黑板 (慢权重 k_proj/v_proj, 闭式二阶)
    post_summary = h.mean()               # [2048]
    k, v = k_proj(summary), v_proj(summary)  # [256]
    grad_W = 2(Wk - v)kᵀ                   # [256,256] 闭式梯度
    W_new = W - exp(log_lr) * grad_W      # 一步 SGD
```

K 帧 = 内层 unroll K 步。段内 W 带梯度到底 (二阶路径由 grad_W 可微回传)。段间 W.detach() 截断梯度 (TBPTT)。

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

### TTT 海马体：W_fast 黑板 + 闭式二阶

W_fast = `[256, 256]` 矩阵 (batch 维度扩展为 `[B, 256, 256]`)。d_mem=256 时黑板大小 256KB/sample。

**READ** (Llama 前)：感知摘要 → q_proj → 黑板读出 → read_proj → memory_token 拼入帧序列。

**WRITE** (Llama 后)：Llama 输出摘要 → k_proj / v_proj → 闭式重建梯度 `grad_W = 2(Wk-v)kᵀ` → 一步 SGD。

**二阶路径**：grad_W 由 k_proj/v_proj (θ_meta) 可微算出。外层 task loss 经 `W_new` → `grad_W` → `k_proj/v_proj` 反传时 autograd 自然再求导 = 二阶。不需要 `create_graph` 嵌套，显存友好。

**对照实验**：把 `grad_W` 换成 `grad_W.detach()` → k_proj.grad / v_proj.grad 立即变 None（二阶断链），而 q_proj 一阶读路径仍有梯度。这是"二阶真的接通了"的铁证。

### inputs_embeds 机制

投影层的 2048-dim 输出通过 `llama(inputs_embeds=...)` 直接替代 Llama 的词嵌入表，绕过 tokenizer。梯度穿过 Llama 流回慢权重，但 Llama 参数 `requires_grad=False` 不更新。

## 组件归属

| 冻结 | 慢权重 (云端学) |
|------|----------------|
| Llama-3.2-1B (bf16/4bit) | ElioModel: AttentionPool ×2 + Projector ×4 |
| SigLIP, DINOv2 | IntentPool (5 query + cross-attn) |
| CLAP (音频编码) | audio_head, gaze_head, frame_head |
| | AutoregHead ×2 (键盘/鼠标, 含 EventEmbedder) |
| | **TTTMemory: W_init, log_lr, q/k/v/read_proj, norm** |

慢权重总计 **~35.8M** 参数 (ElioModel 33.6M + TTTMemory 2.2M)。
W_fast [B, 256, 256] 是快权重 — 内层一步 SGD 更新，外层二阶反传 θ_meta。

## 已验证 (Step 1-5)

```bash
# Step 1-4: 批处理模式
python -m trainer.verify_forward --batch-size 1 --K 4

# Step 5: TTT 单帧记忆循环 + 二阶 backward
python -m trainer.verify_ttt --batch-size 1 --K 4
```

### Step 1-4 检查项

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

### Step 5 检查项 (K=4, 3 步预测 + 3 次 W 更新)

| 检查项 | 期望 | 结果 |
|--------|------|------|
| `run_ttt_segment` 形状全过 | ✅ | W_final [1,256,256], 3 prediction steps |
| `outer.backward()` 不报错 | ✅ | 二阶图反传成功 |
| `ttt.k_proj.grad ≠ 0` | ✅ | max=4.66e-02 (二阶路径核心) |
| `ttt.v_proj.grad ≠ 0` | ✅ | max=2.69e-02 (二阶路径核心) |
| `ttt.W_init.grad ≠ 0` | ✅ | max=8.32e-02 |
| `ttt.log_lr.grad ≠ 0` | ✅ | max=3.83e-03 |
| `ttt.q_proj/read_proj.grad ≠ 0` | ✅ | 一阶读路径正常 |
| IntentPool / 5头 / ElioModel grad ≠ 0 | ✅ | Elio 45/45, Heads 118/130 (12 kb 预期 None) |
| Llama grad = None | ✅ | 0/146 |
| 显存峰值 | ≤12GB | 11.29GB |
| 对照实验: detach `grad_W` → k_proj/v_proj None | ✅ | 二阶断链证明 |
| 闭式数值核对: `allclose(grad_W_auto, 2(Wk-v)kᵀ)` | ✅ | max_diff=0.00e+00 |

## 实施路线

| 步骤 | 内容 | 状态 |
|------|------|------|
| preprocess | 6-step 流水线 → 12 .npy/session | ✅ 完成 |
| Step 1-3 | Dataset + ElioModel + Llama forward | ✅ 完成 |
| Step 4 | IntentPool + 5 预测头 + total_loss | ✅ 完成 |
| Step 5 | TTT 内层循环 + TBPTT 外层回传 | ✅ 完成 (5A) |
| Step 5B | 多段 TBPTT + dataset 连续段取样 | 待实现 |
| Step 6 | 训练循环 + checkpoint | 待实现 |
| Step 7 | 本地部署闭环 | 待实现 |
