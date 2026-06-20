#!/usr/bin/env python3
"""
TTT 单帧记忆循环 — run_ttt_segment()。

每帧 t 的执行流程:
  1. READ:  帧 t 的 34 感知 token 均值 → W_fast → memory_token
  2. 拼帧:  [memory_token, audio, full×16, fovea×16, action] = 35 token
  3. Llama: 35 token 因果前向 → h [B, 35, 2048]
  4. IntentPool → 5 意图 → 5 头预测帧 t+1 → 累加 loss
  5. WRITE: h 的 35 token 均值 → W_fast 一步 SGD → W_{t+1}

K 帧 = 内层 unroll K-1 步预测 (最后一帧无 target)。
W 一路带梯度到底 (段内), backward 时二阶路径经 grad_W 反传。
"""

import torch
import torch.nn.functional as F

from trainer.heads import audio_loss, gaze_loss, frame_loss, total_loss


def run_ttt_segment(
    elio,
    ttt,
    llama,
    heads,
    batch: dict,
    device,
    detach_state=None,
):
    """单段 K 帧记忆循环。

    Args:
        elio:          ElioModel (含 encode_frame_tokens + intent_pool)
        ttt:           TTTMemory
        llama:         冻结的 LlamaForCausalLM
        heads:         nn.ModuleDict with keys: audio, gaze, frame, kb, mouse
        batch:         collate_fn 输出, 各张量 [B, K, ...]
        device:        "cuda" / "cpu"
        detach_state:  跨段传入的 W_fast (已 detach), None 则 init_state

    Returns:
        outer_loss:   scalar tensor (detach-normalized total, ready for backward)
        raw_losses:   dict of float (未归一化原始 loss, 仅监控)
        W_final:      Tensor [B, d_mem, d_mem] (带梯度, 供下一段 detach 传入)
    """
    B, K = batch["siglip"].shape[0], batch["siglip"].shape[1]

    # ── 1. 一次性编码所有帧的 34 token (纯投影, 不进 Llama) ──
    frame_tokens = elio.encode_frame_tokens(batch)                # [B, K, 34, 2048]

    # ── 2. 初始化黑板 ──
    W = detach_state if detach_state is not None else ttt.init_state(B, device)

    accum = {"audio": 0.0, "gaze": 0.0, "frame": 0.0, "kb": 0.0, "mouse": 0.0}
    n_pred = 0

    for t in range(K - 1):                                        # 用帧 t 预测帧 t+1
        ft = frame_tokens[:, t]                                   # [B, 34, 2048]

        # ── READ (Llama 前): 用 34 token 均值查黑板 ──
        pre_summary = ft.mean(dim=1)                              # [B, 2048]
        mem_token = ttt.read(W, pre_summary)                      # [B, 1, 2048]

        # ── 拼 35 token: memory token 放第 0 位 (因果 mask 下感知都能看到它) ──
        seq = torch.cat([mem_token, ft], dim=1)                   # [B, 35, 2048]
        seq_bf16 = seq.to(torch.bfloat16)
        h = llama(inputs_embeds=seq_bf16, output_hidden_states=True
                  ).hidden_states[-1].float()                     # [B, 35, 2048]

        # ── IntentPool (35 token): 复用, 塞 K=1 维 ──
        intents = elio.intent_pool(h.unsqueeze(1))                # [B, 1, 5, 2048]
        intents = intents[:, 0]                                   # [B, 5, 2048]
        ia, ig, iff, ik, im = intents.unbind(dim=1)               # 各 [B, 2048]

        # ── audio head: 预测帧 t+1 CLAP ──
        pred_a = heads["audio"](ia)                               # [B, 512]
        tgt_a = batch["audio"][:, t + 1]                          # [B, 512]
        L_a = audio_loss(pred_a.unsqueeze(1), tgt_a.unsqueeze(1)) # K'=1 给 loss

        # ── gaze head: 预测帧 t+1 注视坐标 ──
        pred_g = torch.sigmoid(heads["gaze"](ig))                 # [B, 2]
        tgt_g = batch["gaze"][:, t + 1]                           # [B, 2]
        L_g = gaze_loss(pred_g, tgt_g)

        # ── frame head: 预测帧 t+1 patch 残差 ──
        # FrameHead 签名 [B, K, 2048], K=1 补维
        ps, pd = heads["frame"](iff.unsqueeze(1))                 # [B,1,196,768], [B,1,257,768]
        true_s = (batch["siglip"][:, t + 1] - batch["siglip"][:, t])
        true_d = (batch["dinov2"][:, t + 1] - batch["dinov2"][:, t])
        L_f = frame_loss(ps, true_s.unsqueeze(1)) + frame_loss(pd, true_d.unsqueeze(1))

        # ── AR 头: 预测帧 t+1 的键鼠事件 ──
        ev_tp1 = batch["events"][:, t + 1]                        # [B, max_ev, 10]
        emsk_tp1 = batch["events_mask"][:, t + 1]                 # [B, max_ev]

        # 键盘头
        k_preds = heads["kb"](ik, ev_tp1, emsk_tp1)
        L_k = heads["kb"].loss(k_preds, ev_tp1, emsk_tp1)

        # 鼠标头
        m_preds = heads["mouse"](im, ev_tp1, emsk_tp1)
        L_m = heads["mouse"].loss(m_preds, ev_tp1, emsk_tp1)

        accum["audio"] += L_a
        accum["gaze"] += L_g
        accum["frame"] += L_f
        accum["kb"] += L_k
        accum["mouse"] += L_m
        n_pred += 1

        # ── WRITE (Llama 后): 用 35 token 均值更新黑板 ──
        post_summary = h.mean(dim=1)                              # [B, 2048]
        W, ssl = ttt.update(W, post_summary)                      # W_{t+1}

    # ── 平均后 detach 归一化总 loss ──
    raw = {k: v / max(n_pred, 1) for k, v in accum.items()}
    outer = total_loss(raw)
    return outer, {k: v.item() for k, v in raw.items()}, W
