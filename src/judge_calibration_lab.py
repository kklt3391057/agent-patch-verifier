#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge 可靠性校准实验台 v2 (Judge Calibration Lab)
=============================================================
主题: 位置偏差 / 冗长偏差 / 自我偏好 —— 测量 + 消偏 + 自检

方法论来源:
- Zheng et al. (MT-Bench, "Judging LLM-as-a-Judge"):
    位置偏差 (GPT-4 偏 first 30% / 换序一致率 65%; Claude-v1 75% / 23.8%)
    冗长偏差 ("repetitive list attack" 骗过 Claude-v1 & GPT-3.5 达 91.3%)
    自我偏好 (GPT-4 自评 +10% 胜率, Claude-v1 约 +25%)
- Wang et al. 2023: 交换顺序后判定翻转 ⇒ position consistency
- IJCNLP 2025 (aclanthology 2025.ijcnlp-long.18):
    repetition stability / position consistency / preference fairness
- 工业实践 (LMArena / Arena-Hard): 每一对答案打两局 (swap-and-confirm)
- 校验工程铁律 (本机 AGENTS.md): 任何检查器必须先证明它能在坏样本上失败

核心设计:
1. 模拟裁判注入**已知**偏差系数 ⇒ 真值已知 ⇒ 测量器本身可被验证
2. **负控制**: 零偏差裁判 ⇒ 测量器必须报"无偏差" (防止把噪声当成偏差)
3. 三种消偏策略对比:
   A. raw            单序 (候选恒在 slot1) —— 基准
   B. swap-confirm   换序双跑, 只有两次指向同一内容才计胜 (弃权其余)
   C. BT-with-position  带位置特征的 Bradley-Terry 拟合, 剥离 beta 后重算
4. 与人工金标对齐用 Cohen's kappa (校准而非仅一致率)
5. 8 条硬断言自检 (负控制必须过 ⇒ 测量器有效)

零外部依赖, 只用标准库。
    python3 judge_calibration_lab_20261122.py
"""
from __future__ import annotations

import json
import pathlib
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 0. 数学工具
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cohens_kappa(a: List[str], b: List[str]) -> Tuple[float, float]:
    """二元 Cohen's kappa: 返回 (kappa, 观察一致率)。"""
    assert len(a) == len(b) and a
    n = len(a)
    labels = sorted(set(a) | set(b))
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    exp = sum((sum(1 for x in a if x == L) / n) * (sum(1 for y in b if y == L) / n)
              for L in labels)
    if abs(1.0 - exp) < 1e-12:
        return 0.0, agree
    return (agree - exp) / (1.0 - exp), agree


# ---------------------------------------------------------------------------
# 1. 模拟裁判: 已知偏差系数 ⇒ 已知真值
# ---------------------------------------------------------------------------

@dataclass
class JudgeParams:
    b_pos: float = 0.0     # 位置偏差: slot1 优势 (logit)
    b_verb: float = 0.0    # 冗长偏差 (log 长度比)
    b_self: float = 0.0    # 自我偏好
    noise: float = 0.0     # 随机噪声 (logit)
    tie_p: float = 0.02    # 判平概率


@dataclass
class Candidate:
    cid: str
    family: str
    quality: float
    length: int


@dataclass
class Order:
    """一次判定 (只记录"选了哪个槽位", 裁判不知道名字)。"""
    slot: str              # 'first' | 'second' | 'tie'
    slot1_cid: str
    slot2_cid: str


class SimJudge:
    """成对比较裁判模拟器。

    位置项只加给 slot1 的人 —— 这正是位置偏差的定义 (偏爱槽位而非内容)。
    """

    def __init__(self, params: JudgeParams, judge_family: str = "familyA", seed: int = 0):
        self.p = params
        self.judge_family = judge_family
        self.rng = random.Random(seed)

    def _verb_term(self, x: Candidate, y: Candidate) -> float:
        """冗长偏差: 按长度**对数比**计分 (接近"看起来更用功"的心理量)。"""
        return self.p.b_verb * (math.log(max(x.length, 1)) - math.log(max(y.length, 1)))

    def _self_term(self, x: Candidate, y: Candidate) -> float:
        sx = 1.0 if x.family == self.judge_family else 0.0
        sy = 1.0 if y.family == self.judge_family else 0.0
        return self.p.b_self * (sx - sy)

    def judge_order(self, x: Candidate, y: Candidate, x_in_slot1: bool) -> Order:
        """裁判看到 (slot1, slot2)。x_in_slot1=True 表示 x 占据 slot1。"""
        p = self.p
        pos_x = 1.0 if x_in_slot1 else -1.0
        s = ((x.quality - y.quality)
             + p.b_pos * pos_x
             + self._verb_term(x, y)
             + self._self_term(x, y))
        s += self.rng.gauss(0.0, p.noise)
        p_x_wins = sigmoid(s)

        if self.rng.random() < p.tie_p:
            slot = "tie"
        else:
            x_wins = self.rng.random() < p_x_wins
            if x_in_slot1:
                slot = "first" if x_wins else "second"
            else:
                # x 在 slot2 ⇒ x 赢时裁判选的是 slot2
                slot = "second" if x_wins else "first"

        return Order(slot,
                     x.cid if x_in_slot1 else y.cid,
                     y.cid if x_in_slot1 else x.cid)


# ---------------------------------------------------------------------------
# 2. 数据集
# ---------------------------------------------------------------------------

FAMILIES = ["familyA", "familyB", "familyC", "familyD"]


def make_dataset(n_pairs: int, seed: int = 7):
    rng = random.Random(seed)
    cands: List[Candidate] = []
    for i in range(n_pairs * 2):
        fam = FAMILIES[rng.randrange(len(FAMILIES))]
        q = rng.gauss(0.0, 1.0)
        if rng.random() < 0.08:
            q += rng.choice([-2.0, 2.0])
        length = max(60, int(rng.gauss(500, 240)))
        cands.append(Candidate(f"c{i:04d}", fam, q, length))
    pairs = [(cands[2 * i].cid, cands[2 * i + 1].cid) for i in range(n_pairs)]
    return cands, pairs


def true_win_prob(x: Candidate, y: Candidate) -> float:
    """真值胜率 (无任何偏差的基准裁判)。"""
    return sigmoid(x.quality - y.quality)


# ---------------------------------------------------------------------------
# 3. 成对评估 (换序双跑) + 判定解码
# ---------------------------------------------------------------------------

@dataclass
class PairOutcome:
    x: str
    y: str
    v1: str              # order1 (slot1=x) 的判定
    v2: str              # order2 (slot1=y) 的判定
    content_consistent: bool = False   # 两次是否指向同一内容
    content_winner: Optional[str] = None
    slot_following: bool = False       # 两次是否选了同一个槽位 (位置跟随)


def run_pairwise(judge: SimJudge, by_id: Dict[str, Candidate],
                 pairs: List[Tuple[str, str]]) -> List[PairOutcome]:
    """每对跑两次 (x 在 slot1 / y 在 slot1), 并解码判定。

    解码表 (order1: slot1=x ; order2: slot1=y):
      (first , second) ⇒ x 两次都赢   ⇒ 内容一致, x 胜
      (second, first ) ⇒ y 两次都赢   ⇒ 内容一致, y 胜
      (first , first ) ⇒ slot1 两次都被选 ⇒ 位置跟随, 内容翻转
      (second, second) ⇒ slot2 两次都被选 ⇒ 位置跟随, 内容翻转
      (tie, tie)        ⇒ 两次都判平   ⇒ 无信息
    """
    out: List[PairOutcome] = []
    for xid, yid in pairs:
        x, y = by_id[xid], by_id[yid]
        v1 = judge.judge_order(x, y, x_in_slot1=True).slot
        v2 = judge.judge_order(x, y, x_in_slot1=False).slot
        o = PairOutcome(x=xid, y=yid, v1=v1, v2=v2)
        if (v1, v2) == ("first", "second"):
            o.content_consistent, o.content_winner = True, xid
        elif (v1, v2) == ("second", "first"):
            o.content_consistent, o.content_winner = True, yid
        elif v1 == "tie" and v2 == "tie":
            o.content_consistent, o.content_winner = True, None
        else:
            o.content_consistent, o.content_winner = False, None
        o.slot_following = (v1 == v2) and v1 in ("first", "second")
        out.append(o)
    return out


# ---------------------------------------------------------------------------
# 4. 偏差测量
# ---------------------------------------------------------------------------

def measure_bias(outcomes: List[PairOutcome], by_id: Dict[str, Candidate],
                 pairs: List[Tuple[str, str]], judge_family: str) -> Dict[str, float]:
    """三类偏差 + 位置一致性。

    关键: 位置偏差只在**质量差不可辨**的对上才显著 ⇒ 必须条件化在 |Δq| 小
    的子集上统计, 否则会被真实质量差淹没 (这是本次实测踩到的坑)。
    """
    n = len(outcomes)
    # --- 位置一致性 (内容层面) ---
    consistent = sum(1 for o in outcomes if o.content_consistent)
    consistency_rate = consistent / n

    # --- 条件子集: 质量不可辨 (|Δq| < 0.5) ---
    low_gap = []
    for o in outcomes:
        x, y = by_id[o.x], by_id[o.y]
        if abs(x.quality - y.quality) < 0.5:
            low_gap.append(o)

    def slot1_pick_rate(subset):
        picks = 0
        tot = 0
        for o in subset:
            for v in (o.v1, o.v2):
                if v in ("first", "second"):
                    tot += 1
                    picks += 1 if v == "first" else 0
        return (picks / tot) if tot else None

    slot1_all = slot1_pick_rate(outcomes)
    slot1_lowgap = slot1_pick_rate(low_gap)

    # 一致性率要看**高差距子集** —— 质量差明显时裁判"应该"一致, 此时翻转才说明
    # 它在跟位置; 全样本里大量无信息的近等质量对会把翻转率拉到 50% 底噪,
    # 把位置偏差彻底淹没 (本次实测踩到的第二个坑)
    high_gap = []
    for o in outcomes:
        x, y = by_id[o.x], by_id[o.y]
        if abs(x.quality - y.quality) >= 1.0:
            high_gap.append(o)
    consistent_highgap = sum(1 for o in high_gap if o.content_consistent)
    consistency_highgap = (consistent_highgap / len(high_gap)) if high_gap else None
    consistent_lowgap = sum(1 for o in low_gap if o.content_consistent)
    consistency_lowgap = (consistent_lowgap / len(low_gap)) if low_gap else None

    # --- 冗长偏差: 等质量 + 长度差异显著 的对, 两侧对称统计 ---
    verb_wins, verb_n = 0, 0
    for o in outcomes:
        x, y = by_id[o.x], by_id[o.y]
        ratio = max(x.length, y.length) / max(min(x.length, y.length), 1)
        if abs(x.quality - y.quality) >= 0.30 or ratio < 1.6:
            continue
        long_is_x = x.length > y.length
        for v, x_in_slot1 in ((o.v1, True), (o.v2, False)):
            if v not in ("first", "second"):
                continue
            picked_is_x = (v == "first") if x_in_slot1 else (v == "second")
            long_won = picked_is_x if long_is_x else (not picked_is_x)
            verb_wins += 1 if long_won else 0
            verb_n += 1
    verb_bias = (verb_wins / verb_n - 0.5) if verb_n else 0.0

    # --- 自我偏好: 异族 + 等质量 的对, 统计裁判同族候选的额外胜率 ---
    self_wins, self_n = 0, 0
    for o in outcomes:
        x, y = by_id[o.x], by_id[o.y]
        if x.family == y.family:
            continue
        if abs(x.quality - y.quality) >= 0.35:
            continue
        if x.family == judge_family:
            self_cand, other = x, y
        elif y.family == judge_family:
            self_cand, other = y, x
        else:
            continue
        self_is_x = (self_cand.cid == o.x)
        for v, x_in_slot1 in ((o.v1, True), (o.v2, False)):
            if v not in ("first", "second"):
                continue
            picked_is_x = (v == "first") if x_in_slot1 else (v == "second")
            self_won = picked_is_x if self_is_x else (not picked_is_x)
            self_wins += 1 if self_won else 0
            self_n += 1
    self_bias = (self_wins / self_n - 0.5) if self_n else 0.0

    return {
        "position_consistency_rate": consistency_rate,
        "position_consistency_rate_lowgap": consistency_lowgap,
        "position_consistency_rate_highgap": consistency_highgap,
        "n_highgap": len(high_gap),
        "position_flip_rate": 1.0 - consistency_rate,
        "slot1_pick_rate_all": slot1_all,
        "slot1_pick_rate_lowgap": slot1_lowgap,
        "position_bias_measured": (slot1_lowgap - 0.5) if slot1_lowgap is not None else None,
        "verbosity_bias_measured": verb_bias,
        "self_preference_measured": self_bias,
        "n_pairs": n,
        "n_lowgap": len(low_gap),
        "n_verb_obs": verb_n,
        "n_self_obs": self_n,
    }


# ---------------------------------------------------------------------------
# 5. 消偏策略 B (swap-confirm) 与 C (BT with position)
# ---------------------------------------------------------------------------

def swap_confirm(outcomes: List[PairOutcome]) -> Dict[str, float]:
    """换序双跑: 只有两次指向同一内容才计胜 (其余弃权)。"""
    dx = dy = 0
    for o in outcomes:
        if o.content_winner is None:
            continue
        if o.content_winner == o.x:
            dx += 1
        else:
            dy += 1
    tot = dx + dy
    return {
        "decisive_pairs": tot,
        "coverage": tot / len(outcomes) if outcomes else 0.0,
        "x_win_rate_decisive": (dx / tot) if tot else None,
        "x_wins": dx,
        "y_wins": dy,
    }


def fit_bradley_terry_with_position(outcomes: List[PairOutcome],
                                    iters: int = 6000, lr: float = 0.35,
                                    l2: float = 0.005,
                                    l2_beta: float = 0.02) -> Tuple[Dict[str, float], float]:
    """拟合: P(x 胜) = sigmoid(theta_x - theta_y + beta * sgn)

    sgn = +1 当 x 位于 slot1, -1 当 x 位于 slot2。
    beta 即位置偏差强度估计 (logit 单位); ties 不提供信息, 跳过。
    theta 中心化 (只有相对差有意义), beta 软约束防发散。
    """
    cids = sorted({o.x for o in outcomes} | {o.y for o in outcomes})
    theta = {c: 0.0 for c in cids}
    beta = 0.0
    obs = []
    for o in outcomes:
        if o.v1 == "tie" or o.v2 == "tie":
            continue
        # order1: x 在 slot1 ⇒ "first" 表示 x 胜
        obs.append((o.x, o.y, 1.0, 1 if o.v1 == "first" else 0))
        # order2: x 在 slot2 ⇒ "second" 表示 x 胜
        obs.append((o.x, o.y, -1.0, 1 if o.v2 == "second" else 0))
    for _ in range(iters):
        gt = {c: 0.0 for c in cids}
        cnt = {c: 0 for c in cids}
        gb = 0.0
        for (x, y, sgn, won) in obs:
            s = theta[x] - theta[y] + beta * sgn
            g = won - sigmoid(s)
            gt[x] += g
            gt[y] -= g
            cnt[x] += 1
            cnt[y] += 1
            gb += g * sgn
        # ⚠️ 必须用**平均**梯度: 上千观测的求和梯度会退化为随机游走 ⇒ beta 发散撞边界
        mean_g = sum(gt.values()) / len(cids)
        for c in cids:
            avg = gt[c] / max(cnt[c], 1)
            theta[c] += lr * (avg - mean_g / len(cids) - l2 * theta[c])
        # beta 梯度取全体观测平均 + L2 向 0 回拉 (无位置偏差时 beta 应为 0)
        beta += lr * (gb / max(len(obs), 1) - l2_beta * beta)
        beta = max(-4.0, min(4.0, beta))
    mean_t = sum(theta.values()) / len(theta)
    for c in cids:
        theta[c] -= mean_t
    return theta, beta


def bt_win_rate(theta: Dict[str, float], pairs) -> float:
    vals = [sigmoid(theta[a] - theta[b]) for (a, b) in pairs]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# 6. 人工金标 + kappa
# ---------------------------------------------------------------------------

def make_gold(by_id, pairs, seed: int = 99, human_noise: float = 0.05):
    """人工金标: 以真值质量为主, 掺入少量人工噪声。"""
    rng = random.Random(seed)
    gold = []
    for (xid, yid) in pairs:
        x, y = by_id[xid], by_id[yid]
        win_x = rng.random() < true_win_prob(x, y)
        if rng.random() < human_noise:
            win_x = not win_x
        gold.append((xid, yid, xid if win_x else yid))
    return gold


def kappa_vs_gold(outcomes, gold, theta=None, restrict_decisive: bool = False,
                  strategy: str = "raw"):
    """与人工金标的 Cohen's kappa。

    strategy:
      "raw"          —— 只用 order1 判定 (单序基准)
      "swap_confirm" —— 换序双跑, 只有两次指向同一内容才给标签 (其余弃权)
      "bt"           —— BT 消偏后的全局实力排序
    restrict_decisive=True: 只在 raw 策略能给出标签 (非 tie) 的对上比较,
    否则 raw 会被"标了 tie 但金标没有 tie"无谓地扣分 ⇒ 比较不公平。
    """
    obp = {(o.x, o.y): o for o in outcomes}
    gold_labels, labs = [], []
    for (xid, yid, w) in gold:
        o = obp[(xid, yid)]
        if strategy == "swap_confirm":
            if o.content_winner is None:
                continue          # 弃权: 不计入
            labs.append(o.content_winner)
        elif strategy == "bt":
            labs.append(xid if sigmoid(theta[xid] - theta[yid]) > 0.5 else yid)
        else:  # raw
            if o.v1 == "tie":
                if restrict_decisive:
                    continue
                labs.append("tie")
            else:
                labs.append(xid if o.v1 == "first" else yid)
        gold_labels.append(w)
    return cohens_kappa(gold_labels, labs)


# ---------------------------------------------------------------------------
# 7. 场景评估
# ---------------------------------------------------------------------------

def evaluate(judge: SimJudge, by_id, pairs, gold, label: str) -> Dict:
    outcomes = run_pairwise(judge, by_id, pairs)
    metrics = measure_bias(outcomes, by_id, pairs, judge.judge_family)
    sc = swap_confirm(outcomes)

    # raw 单序: 候选恒在 slot1
    raw_w = raw_n = 0
    for o in outcomes:
        if o.v1 in ("first", "second"):
            raw_n += 1
            raw_w += 1 if o.v1 == "first" else 0
    raw_rate = raw_w / raw_n if raw_n else None

    true_rate = sum(true_win_prob(by_id[a], by_id[b]) for (a, b) in pairs) / len(pairs)

    theta, beta = fit_bradley_terry_with_position(outcomes)
    bt_rate = bt_win_rate(theta, pairs)

    k_raw, a_raw = kappa_vs_gold(outcomes, gold, strategy="raw")
    k_bt, a_bt = kappa_vs_gold(outcomes, gold, theta=theta, strategy="bt")
    k_sc, a_sc = kappa_vs_gold(outcomes, gold, strategy="swap_confirm")
    # 公平比较: 只在 raw 能出标签的对上比 (否则 tie 惩罚不公平)
    k_raw_d, a_raw_d = kappa_vs_gold(outcomes, gold, strategy="raw", restrict_decisive=True)

    return {
        "label": label,
        "judge_params": asdict(judge.p),
        "judge_family": judge.judge_family,
        "metrics": metrics,
        "beta_estimated": beta,
        "raw_win_rate_x": raw_rate,
        "true_win_rate_x": true_rate,
        "swap_confirm": sc,
        "bt_corrected_win_rate_x": bt_rate,
        "abs_err_raw": abs(raw_rate - true_rate) if raw_rate is not None else None,
        "abs_err_bt": abs(bt_rate - true_rate),
        "kappa_raw": k_raw,
        "kappa_bt": k_bt,
        "kappa_swap_confirm": k_sc,
        "agreement_raw": a_raw,
        "agreement_bt": a_bt,
        "agreement_swap_confirm": a_sc,
        "kappa_raw_decisive": k_raw_d,
        "outcomes": outcomes,
    }


# ---------------------------------------------------------------------------
# 8. 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    seed = 20261122
    cands, pairs = make_dataset(n_pairs=600, seed=seed)
    by_id = {c.cid: c for c in cands}
    gold = make_gold(by_id, pairs, seed=seed + 1)

    print("=" * 94)
    print("LLM-as-a-Judge 可靠性校准实验台 v2 — 位置偏差 / 冗长偏差 / 自我偏好")
    print(f"数据集: {len(pairs)} 对 / {len(cands)} 候选 / {len(FAMILIES)} 家族")
    print("=" * 94)

    clean = SimJudge(JudgeParams(b_pos=0.0, b_verb=0.0, b_self=0.0, noise=0.25),
                     judge_family="familyA", seed=11)
    biased = SimJudge(JudgeParams(b_pos=0.95, b_verb=0.70, b_self=0.85, noise=0.25),
                      judge_family="familyA", seed=12)

    r_clean = evaluate(clean, by_id, pairs, gold, "负控制: 零偏差裁判")
    r_biased = evaluate(biased, by_id, pairs, gold, "强偏差裁判 (pos/verb/self 全开)")

    for r in (r_clean, r_biased):
        m = r["metrics"]
        print("\n" + "-" * 94)
        print(f"【{r['label']}】")
        p = r["judge_params"]
        print(f"  真值参数 : b_pos={p['b_pos']}  b_verb={p['b_verb']}  b_self={p['b_self']}")
        print(f"  位置一致性率 (全样本)            : {m['position_consistency_rate']:.3f}"
              f"   翻转率 {m['position_flip_rate']:.3f}")
        print(f"  位置一致性率 (高差距 |Δq|≥1.0)   : {m['position_consistency_rate_highgap']:.3f}"
              f"   (n={m['n_highgap']})   文献: GPT-4≈0.65, Claude-v1≈0.24")
        print(f"  位置一致性率 (等质量 |Δq|<0.5)    : {m['position_consistency_rate_lowgap']:.3f}"
              f"   (n={m['n_lowgap']})")
        print(f"  slot1 选中率  全样本 {m['slot1_pick_rate_all']:.3f} | "
              f"等质量子集 {m['slot1_pick_rate_lowgap']:.3f} (n={m['n_lowgap']})")
        print(f"  ⇒ 位置偏差测量值 (等质量子集)     : {m['position_bias_measured']:+.3f}")
        print(f"  冗长偏差 (等质量+长度差显著, 长者胜率-0.5) : {m['verbosity_bias_measured']:+.3f}"
              f"  (n={m['n_verb_obs']})")
        print(f"  自我偏好 (异族+等质量, 同族候选胜率-0.5)  : {m['self_preference_measured']:+.3f}"
              f"  (n={m['n_self_obs']})")
        print(f"  BT 位置系数 beta 估计                    : {r['beta_estimated']:+.3f}")
        print("  ── 胜率校准 ──")
        print(f"     raw 单序 (候选恒在 slot1)  : {r['raw_win_rate_x']:.3f}  "
              f"| 误差 {r['abs_err_raw']:.3f}")
        sc = r["swap_confirm"]
        print(f"     换序确认 (一致才计, 覆盖 {sc['coverage']:.3f}) : "
              f"{sc['x_win_rate_decisive']:.3f}  (decisive n={sc['decisive_pairs']})")
        print(f"     BT 消偏后                  : {r['bt_corrected_win_rate_x']:.3f}  "
              f"| 误差 {r['abs_err_bt']:.3f}")
        print(f"     真值胜率                    : {r['true_win_rate_x']:.3f}")
        print("  ── 与人工金标对齐 (Cohen's kappa) ──")
        print(f"     raw : kappa={r['kappa_raw']:+.3f}  一致率={r['agreement_raw']:.3f}"
              f"  | 公平子集 kappa={r['kappa_raw_decisive']:+.3f}")
        print(f"     BT  : kappa={r['kappa_bt']:+.3f}  一致率={r['agreement_bt']:.3f}")
        print(f"     换序确认 (弃权后) : kappa={r['kappa_swap_confirm']:+.3f}  "
              f"一致率={r['agreement_swap_confirm']:.3f}")

    # ------------------------------------------------------------------
    # 9. 自检断言 (fail-on-base 纪律)
    # ------------------------------------------------------------------
    print("\n" + "=" * 94)
    print("自检断言 (负控制必须全过 ⇒ 测量器本身有效)")
    print("=" * 94)
    checks = []

    def chk(name, cond, detail):
        checks.append((name, bool(cond), detail))

    cm, bm = r_clean["metrics"], r_biased["metrics"]

    chk("负控制: 位置一致性率 (高差距子集) > 0.75 (干净裁判不该被误判为不可靠)",
        cm["position_consistency_rate_highgap"] > 0.75,
        f"got {cm['position_consistency_rate_highgap']:.3f}")
    chk("负控制: |beta| < 0.25 (无位置偏差时不应估出偏差)",
        abs(r_clean["beta_estimated"]) < 0.25,
        f"got {r_clean['beta_estimated']:+.3f}")
    chk("负控制: |位置偏差测量| < 0.08 (等质量子集上应接近 0)",
        abs(cm["position_bias_measured"]) < 0.08,
        f"got {cm['position_bias_measured']:+.3f}")
    chk("负控制: |冗长偏差| < 0.12",
        abs(cm["verbosity_bias_measured"]) < 0.12,
        f"got {cm['verbosity_bias_measured']:+.3f}")
    chk("负控制: |自我偏好| < 0.12",
        abs(cm["self_preference_measured"]) < 0.12,
        f"got {cm['self_preference_measured']:+.3f}")

    chk("正例: 强偏差裁判 beta > 0.40 (真值 0.95)",
        r_biased["beta_estimated"] > 0.40,
        f"got {r_biased['beta_estimated']:+.3f}")
    chk("正例: 强偏差裁判高差距子集一致性率比负控制低 > 0.03",
        cm["position_consistency_rate_highgap"] - bm["position_consistency_rate_highgap"] > 0.03,
        f"clean {cm['position_consistency_rate_highgap']:.3f} vs biased {bm['position_consistency_rate_highgap']:.3f}")
    chk("正例: 等质量子集 slot1 选中率比负控制高 > 0.08",
        bm["slot1_pick_rate_lowgap"] - cm["slot1_pick_rate_lowgap"] > 0.08,
        f"clean {cm['slot1_pick_rate_lowgap']:.3f} vs biased {bm['slot1_pick_rate_lowgap']:.3f}")
    chk("正例: 冗长偏差 > 0.05 (真值 0.70)",
        bm["verbosity_bias_measured"] > 0.05,
        f"got {bm['verbosity_bias_measured']:+.3f}")
    chk("正例: 自我偏好 > 0.05 (真值 0.85)",
        bm["self_preference_measured"] > 0.05,
        f"got {bm['self_preference_measured']:+.3f}")
    chk("正例: raw 单序胜率被偏差抬高 > 0.05 (真值 0.491)",
        r_biased["raw_win_rate_x"] - r_biased["true_win_rate_x"] > 0.05,
        f"raw {r_biased['raw_win_rate_x']:.3f} vs true {r_biased['true_win_rate_x']:.3f}")
    chk("正例: BT 消偏后胜率误差 < raw 单序误差",
        r_biased["abs_err_bt"] < r_biased["abs_err_raw"],
        f"bt_err={r_biased['abs_err_bt']:.4f} vs raw_err={r_biased['abs_err_raw']:.4f}")
    chk("正例: BT 消偏后 kappa (公平可比子集) 不低于 raw - 0.02",
        r_biased["kappa_bt"] >= r_biased["kappa_raw_decisive"] - 0.02,
        f"bt={r_biased['kappa_bt']:+.3f} vs raw_fair={r_biased['kappa_raw_decisive']:+.3f}")
    chk("正例: 换序确认 kappa >= raw kappa (逐对消偏优于单序)",
        r_biased["kappa_swap_confirm"] >= r_biased["kappa_raw"],
        f"sc={r_biased['kappa_swap_confirm']:+.3f} vs raw={r_biased['kappa_raw']:+.3f}")

    all_pass = True
    for (name, ok, detail) in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        all_pass = all_pass and ok
    print("-" * 94)
    print(f"自检结果: {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} 通过 "
          f"=> {'PASS' if all_pass else 'FAIL'}")
    print("=" * 94)

    out = {
        "generated_at": "2026-11-22T16:08:00+08:00",
        "topic": "LLM-as-a-Judge position/verbosity/self-preference bias calibration",
        "n_pairs": len(pairs),
        "scenarios": [],
        "assertions": [{"name": n, "pass": ok, "detail": d} for (n, ok, d) in checks],
        "all_pass": all_pass,
    }
    for r in (r_clean, r_biased):
        out["scenarios"].append({
            "label": r["label"],
            "judge_params": r["judge_params"],
            "metrics": r["metrics"],
            "beta_estimated": r["beta_estimated"],
            "raw_win_rate_x": r["raw_win_rate_x"],
            "true_win_rate_x": r["true_win_rate_x"],
            "bt_corrected_win_rate_x": r["bt_corrected_win_rate_x"],
            "abs_err_raw": r["abs_err_raw"],
            "abs_err_bt": r["abs_err_bt"],
            "kappa_raw": r["kappa_raw"],
            "kappa_bt": r["kappa_bt"],
            "kappa_swap_confirm": r["kappa_swap_confirm"],
            "swap_confirm": r["swap_confirm"],
        })
    # 输出路径：默认落在**本脚本同目录**（包内自足 ⇒ 不依赖任何绝对路径）
    path = str(pathlib.Path(__file__).resolve().parent /
               "judge_calibration_results_20261122.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {path}")


if __name__ == "__main__":
    main()
