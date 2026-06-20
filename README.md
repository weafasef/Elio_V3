# Elio 模型总结

## 一句话概括

本地运行的具身桌面智能体。Llama 当"单刻大脑"理解此刻屏幕，可训练的 TTT 海马体当"记忆"——通过持续预测下一刻、预测错了就实时改写记忆权重，长期内化你的电脑使用习惯。没有人工规则，记什么忘什么全靠预测误差倒逼。

## 核心结构：双时间尺度

```
跨时间 (慢, O(1))     TTT 海马体 W_fast
                           ↕ prefix 注入 / 内层 backward 写入
此刻理解 (快, 窗口恒定)  Llama-3.2-1B (冻结, 4bit)
```

- **Llama 不记历史**：只读"此刻 token + 记忆 prefix"，窗口恒定 → O(1) 显存
- **所有跨时间信息压在 W_fast**：序列再长它不涨，拒绝 KV cache 膨胀
- **记忆不靠堆积，靠压缩**

## 时间定义

一刻 tick = **100ms**（10 ticks/秒）。音频按窗切片、帧取 tick 末、动作用事件编码器压成定长 token。

## 感知层

### 编码器（全部冻结）

| 编码器 | 型号 | 输入 | 输出维度 | 确认来源 |
|---|---|---|---|---|
| CLAP | laion/clap-htsat-unfused | 100ms 音频窗 | **512** | `audio_config.projection_dim` |
| SigLIP | google/siglip-base-patch16-384 | 全屏 384² / 中央凹 448² | **768** | ViT-B/16 vision hidden |
| DINOv2 | facebook/dinov2-base | 同上，与 SigLIP 共用 | **768** | `hidden_size` |

### 中央凹 (foveation)

```
frame_t (原始全屏)
  ├─ 下采样 384² → SigLIP + DINO → z_fglob   (周边视觉)
  └─ gaze_{t-1} 处裁 448² → SigLIP + DINO → z_ffov  (中央凹细节)
```

- 两路共用同一套冻结编码器权重，省显存
- 裁切位置来自上一刻 gaze 预测，硬裁（框外像素不进编码器）
- 全局 384² 约 16 patch token，中央凹 448² 约 16 patch token，token 数恒定

### 投影维度（精确值）

```
进 Llama (模态投影，3 个独立线性层):
  proj_a:    Linear(512  → 2048)    # CLAP → Llama
  proj_fg:   Linear(1536 → 2048)    # SigLIP+DINO concat (768+768) → Llama
  proj_ff:   与 proj_fg 共用权重      # 中央凹与周边共享投影矩阵
  proj_act:  Linear(TBD  → 2048)    # 动作编码, 维度待定

出 Llama (预测头，4 个线性层):
  audio_head:   Linear(2048 → 512)   # 预测下刻 CLAP latent
  frame_head:   Linear(2048 → 1536)  # 预测下刻 latent 残差 Δz_f
  action_head:  Linear(2048 → TBD)   # 预测下刻动作
  gaze_head:    Linear(2048 → 2)     # 预测下刻中央凹坐标 (x, y)
```

## 单 tick 完整数据流

```
[1] 感知 (冻结, no_grad)
    z_a   = CLAP(audio_t)                          # 512
    z_fg  = [SigLIP(frame_t↓384); DINO(frame_t↓384)]  # 1536
    z_ff  = [SigLIP(crop_t@448); DINO(crop_t@448)]    # 1536
    z_act = ActEnc(action_t)                        # TBD, 无动作 → no-op token

[2] 投影 → 拼窗口
    tok   = [proj_a(z_a), proj_fg(z_fg), proj_ff(z_ff), proj_act(z_act)]
    mem   = read(W_fast)                            # prefix token(s)
    window = [mem] + tok                             # 长度恒定

[3] Llama 推理 (冻结)
    h_t = Llama(window)                              # 2048

[4] 记忆写入 (TTT 内层, 可训练)
    ŝ     = f(W_fast; h_t)
    L_in  = ||ŝ - target_latent_t||²
    W_fast ← W_fast - η · ∇L_in                      # 单步梯度, 激活即弃

[5] 预测下一刻
    ẑ_a_{t+1}  = audio_head(h_t)                     # 下刻音频
    Δẑ_f_{t+1} = frame_head(h_t)                     # 下刻画面残差
    â_{t+1}    = action_head(h_t)                    # 下刻动作 (部署时执行)
    gaze_{t+1} = gaze_head(h_t)                      # 下刻中央凹坐标
```

## 关键设计决策

### 动作：双重角色 + 双并行自回归

动作既是 Elio 要执行的**输出**，也是世界模型要预测的**对象**（建模"你怎么操作电脑"）。

**输入侧**（变长事件流 → 定长 token）：
```
单事件编码:
  键盘:    [KB,    down/up, 键码]
  鼠标:    [MS,    down/up, 左/右/中]
  鼠标移动: [MS,    move,   起点, 终点, 轨迹长度]

→ 各字段过独立 embedding → 拼成事件向量
→ 小序列编码器(几层 Transformer) pool 成定长
→ 键盘流、鼠标流各 1 个 token = 共 2 个动作 token
```

**输出侧**（双并行自回归）：
- 键盘流、鼠标流各自回归吐事件直到 eos
- 拆 down/up → 长按、拖拽、组合键自然涌现
- 自回归只在刻内展开，守住 100ms 节奏

**输入输出共享事件词表**：两侧复用同一套类型/键码 embedding，模型不用学两遍"什么是按下 A"。

### 画面预测：残差目标

GUI 大部分区域静止，直接预测整帧会学到"复制上一帧"。改成：
```
target = z_f_{t+1} - z_f_t               # 预测变化量
w(p)   = 1 + α·|Δz_f(p)|                # 变化大的 patch 权重高
L_frame = Σ_p w(p)·||Δẑ_f(p) - Δz_f(p)||²
```
逼模型学"我这个动作改变了什么"——因果倒逼。

### Gaze：硬裁 + 伪标签解耦

crop 用硬裁真省 token，坐标回归用伪标签直接监督保持可微：
```
伪标签 = argmax_region |frame_t - frame_{t-1}|   # motion saliency 峰值
L_gaze = ||gaze_pred - 伪标签||²
λ_gaze: 0.5 起 → 后期降到 0.1                      # 先拄拐杖, 后自主
```

## 训练：两阶段

### 阶段一：云端 meta-training (24GB)

双层优化，TBPTT 截断。学的不是"知识"，是**如何用微型梯度下降快速记住东西**。

```
for batch in dataloader:
    W_fast = init_from(θ_meta)
    for window in chunk(batch, K):      # K=4 起步
        for t in window:
            h = forward(W_fast, perceive(t))
            W_fast = inner_update(W_fast, h)  # 保留计算图
            L_out += weighted_loss(predict(h), target_{t+1})
        L_out.backward()                # BPTT 展开 K 步 → θ_meta
        opt_meta.step()
        W_fast = W_fast.detach()        # 跨窗口梯度截断
```

θ_meta = {内层学习率 η, W_fast 初始化, 投影矩阵, 预测头, 双自回归头, 事件编码器, 共享 embedding}

### 阶段二：本地部署 (8GB)

冻结 θ_meta，只留内层闭环：

```
W_fast = load(trained_init)
while running:                          # 无窗口、无截断、无穷长
    h = forward(W_fast, perceive(now))
    L_in = self_sup_loss(h)
    W_fast = inner_update(W_fast, h)    # 单步 backward, 激活即弃
    execute(action_head(h))
```

训练时截断，部署时连续。

## 组件归属

| | 组件 |
|---|---|
| **冻结** | CLAP、SigLIP、DINOv2、Llama-3.2-1B (4bit) |
| **慢权重** (云端学, 本地固定) | TTT 初始化、内层学习率、投影层、预测头、双自回归头、事件编码器、共享 embedding |
| **快权重** (本地实时进化) | W_fast 海马体 —— 唯一在用户电脑上持续改变的东西 |

## 实施路线（按风险倒序）

| 阶段 | 内容 | 验证目标 | 依赖真数据 |
|---|---|---|---|
| **P0 命脉** | TTT 内层更新 + TBPTT 外层回传, Llama 用 MLP stub, 模态用随机张量 | θ_meta 梯度非零、K=4 不爆、截断正确 | 否 |
| P1 感知接入 | 接真 CLAP/SigLIP/DINO, gaze 伪标签硬裁 | 单帧编码显存、token 数 | 否 |
| P2 接真 Llama | 1B 4bit + 记忆 prefix 注入 | 端到端显存 ≤8GB | 否 |
| P3 数据与训练 | 录屏+音频+动作, 跑阶段一 | loss 下降、动作预测准确率 | 是 |
| P4 部署闭环 | 阶段二本地连续运行 | 长期显存平稳、记忆生效 | 是 |

## 待实测参数

- **W_fast 容量**：2 块 2048×2048 够不够装个人习惯，P0 实测定
- **K 上限**：24GB 能展开多深，P0 给出
- **动作编码维度**：128~256 起步，P3 看效果
- **记忆 prefix 长度**：从 W_fast 读出几个 token 注入 Llama
- **λ 权重**：初值给好，P3 看各模态 loss 曲线调
