#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_saturation_contamination_lab_20261122.py

主题：Benchmark 饱和度（S-index）与训练集污染（13-gram overlap）联合体检器
依据：ICML 2026 "When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation"
      (arXiv:2602.16763) + 2026 污染检测方法综述（Nourbakhsh et al., 55 项研究）

核心公式（论文 §2）：
  SE(s)   = sqrt(s(1-s) / n_eff),  n_eff = n_test ** alpha   (alpha 默认 0.5)
  SE_D    = sqrt((s1(1-s1) + sk(1-sk)) / n_eff)
  Delta   = s1 - sk                    (k 默认 5 = top-5 模型)
  R_norm  = Delta / SE_D               (信噪比：分数差 / 评测噪声)
  S_index = exp(-R_norm ** 2)          ∈ [0,1]，越接近 1 越饱和

饱和 vs 停滞（论文 §2）：
  stagnation  = top 模型统计上不可区分 (Delta <= z * SE_D)
  saturation  = stagnation AND 分数逼近经验天花板 (s1 >= ceiling_ratio * s_max_possible)
  论文自身的 S_index 只用了"不可区分"部分 => 存在定义/测度内部不一致（pith.science 批评）

污染检测：
  13-gram 精确重叠（GPT-3 标准）→ 抓逐字泄漏
  归一化 n-gram 命中率 → 定量污染指数
  语义近似的漏检 → 用"最小编辑距离"作粗略同义改写探针

无第三方依赖，纯标准库。
"""

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple

ALPHA = 0.5          # 有效样本量压缩指数（论文默认）
K = 5                # top-k 模型（论文默认）
Z = 1.96             # 95% 置信
NGRAM = 13           # 13-gram：GPT-3 论文标准
CEILING_RATIO = 0.98  # s1 >= 0.98 * 满分 => 逼近经验天花板


# ---------------------------------------------------------------- 数据结构
@dataclass
class Benchmark:
    name: str
    domain: str
    n_test: int
    scores: List[float]           # top-k 降序分数
    max_possible: float = 1.0
    private_test: bool = False
    expert_curated: bool = False
    age_months: int = 0
    contamination_ratio: float = 0.0


@dataclass
class SaturationReport:
    name: str
    s_index: float
    bin: str
    r_norm: float
    delta: float
    se_delta: float
    n_eff: float
    stagnation: bool
    saturated: bool
    stale_measurement: bool       # S_index 高但未逼近天花板 => 论文定义内部不一致区
    contamination_ratio: float
    trust: str                   # TRUST / CAUTION / DISCARD


# ---------------------------------------------------------------- 饱和度计算
def se_score(s: float, n_test: int, alpha: float = ALPHA) -> float:
    """单分数标准误 SE(s) = sqrt(s(1-s)/n_eff)"""
    n_eff = max(n_test ** alpha, 1.0)
    s = min(max(s, 0.0), 1.0)
    return math.sqrt(s * (1.0 - s) / n_eff)


def saturation_index(scores: List[float], n_test: int,
                     k: int = K, alpha: float = ALPHA) -> Tuple[float, float, float, float]:
    """返回 (S_index, R_norm, Delta, SE_Delta)"""
    ss = sorted(scores, reverse=True)[:k]
    if len(ss) < 2:
        return 0.0, 0.0, 0.0, 0.0
    s1, sk = ss[0], ss[-1]
    delta = s1 - sk
    n_eff = max(n_test ** alpha, 1.0)
    se_delta = math.sqrt((s1 * (1 - s1) + sk * (1 - sk)) / n_eff)
    if se_delta <= 0:
        return 0.0, 0.0, delta, 0.0
    r_norm = delta / se_delta
    s_index = math.exp(-(r_norm ** 2))
    return s_index, r_norm, delta, se_delta


def classify_bin(s_index: float) -> str:
    """论文五档分箱"""
    if s_index < 0.01:
        return "very_low"
    if s_index < 0.3:
        return "low"
    if s_index < 0.7:
        return "moderate"
    if s_index < 0.9:
        return "high"
    return "very_high"


def assess(b: Benchmark) -> SaturationReport:
    s_index, r_norm, delta, se_delta = saturation_index(b.scores, b.n_test)
    stagnation = se_delta > 0 and delta <= Z * se_delta
    near_ceiling = b.scores[0] >= CEILING_RATIO * b.max_possible
    saturated = stagnation and near_ceiling
    # 论文的 S_index 只度量 stagnation；两者不一致的区域 = 测量已失效但问题未解决
    stale = (s_index >= 0.7) and (not near_ceiling)

    trust = "TRUST"
    if s_index >= 0.7 and b.contamination_ratio > 0.10:
        trust = "DISCARD"
    elif s_index >= 0.7:
        trust = "DISCARD" if near_ceiling else "CAUTION"
    elif s_index >= 0.3:
        trust = "CAUTION"
    if b.contamination_ratio > 0.20:
        trust = "DISCARD"

    return SaturationReport(
        name=b.name, s_index=round(s_index, 4), bin=classify_bin(s_index),
        r_norm=round(r_norm, 3), delta=round(delta, 4),
        se_delta=round(se_delta, 5), n_eff=round(b.n_test ** ALPHA, 1),
        stagnation=stagnation, saturated=saturated, stale_measurement=stale,
        contamination_ratio=b.contamination_ratio, trust=trust,
    )


# ---------------------------------------------------------------- 污染检测
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: List[str], n: int = NGRAM) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def build_corpus_index(docs: List[str], n: int = NGRAM) -> set:
    idx = set()
    for d in docs:
        for g in ngrams(tokenize(d), n):
            idx.add(hashlib.blake2b(" ".join(g).encode(), digest_size=8).digest())
    return idx


def contamination_ratio(item: str, corpus_idx: set, n: int = NGRAM) -> float:
    """13-gram 命中率：benchmark 条目中有多少比例的 n-gram 出现在训练语料里"""
    gs = ngrams(tokenize(item), n)
    if not gs:
        return 0.0
    hit = 0
    for g in gs:
        h = hashlib.blake2b(" ".join(g).encode(), digest_size=8).digest()
        if h in corpus_idx:
            hit += 1
    return hit / len(gs)


def paraphrase_probe(item: str, corpus_idx: set) -> bool:
    """改写逃逸探针：把 benchmark 条目同义改写后，若改写版出现在语料里
    => 污染项通过"换形"存活，逐字 n-gram 过滤器对它无效。"""
    alt = _paraphrase(item)
    gs = ngrams(tokenize(alt), NGRAM)
    if not gs:
        return False
    for g in gs:
        h = hashlib.blake2b(" ".join(g).encode(), digest_size=8).digest()
        if h in corpus_idx:
            return True
    return False


# ---------------------------------------------------------------- 合成世界
def _paraphrase(text: str) -> str:
    """同义改写：模拟污染项以改写形式存活，绕过逐字 n-gram 过滤器"""
    syn = {"the": "a", "is": "was", "are": "were", "of": "in", "and": "plus",
           "to": "for", "in": "within", "with": "alongside", "that": "which"}
    toks = tokenize(text)
    return " ".join(syn.get(t, t) for t in toks)


def synthesize_world(seed: int = 20261122) -> Tuple[List[Benchmark], List[str], List[str]]:
    """生成 60 个 benchmark + 训练语料 + benchmark 条目，模拟真实排行榜生态"""
    rng = random.Random(seed)
    names = ["MMLU", "HumanEval", "GSM8K", "HellaSwag", "ARC-Challenge", "BigBench-Hard",
             "SWE-bench Verified", "MATH", "GPQA", "AIME-2025", "MMLU-Pro", "LiveBench",
             "WebArena", "OSWorld", "AgentBench", "τ-bench", "BFCL-v3", "IFEval",
             "MGSM", "TheoremQA", "DROP", "SQuAD-2", "TriviaQA", "HotpotQA",
             "BoolQ", "WinoGrande", "PIQA", "SocialIQa", "CommonsenseQA", "WinoGrande-H",
             "RiddleSense", "LogiQA", "ReClor", "hellaswag-zh", "ARC-Easy",
             "OpenBookQA", "SciQ", "QASC", "MedMCQA", "MMLU-Pro-Redux", "Global-MMLU",
             "HumanEval+", "MBPP+", "DS-1000", "MultiPL-E", "LeetCode-Contest",
             "SWE-Lancer", "Terminal-Bench", "GAIA", "BrowseComp", "SimpleQA",
             "FRAMES", "LongBench", "RULER", "InfiniteBench", "HELMET", "CRAG"]
    domains = ["knowledge", "code", "math", "agent", "reasoning", "reading"]
    benchmarks: List[Benchmark] = []
    items: List[str] = []
    corpus: List[str] = []

    for i, nm in enumerate(names[:60]):
        dom = domains[i % len(domains)]
        n_test = rng.choice([100, 200, 500, 1000, 2000, 5000, 10000, 14042])
        age = rng.choice([12, 24, 36, 48, 60, 72, 96, 120])
        # 年龄越大 => 越饱和（论文实证趋势）；小测试集 => 噪声更大 => 越易饱和
        private = rng.random() < 0.35
        curated = rng.random() < 0.45
        sat_bias = min(0.95, 0.12 + age / 150.0 + (0.35 if n_test <= 500 else 0.0))
        # 论文实证：专家策展是唯一有效保护因子；私有测试集几乎无保护
        if curated:
            sat_bias = max(0.05, sat_bias - 0.30)
        base = rng.uniform(0.55, 0.80)
        top = base + (1.0 - base) * rng.uniform(0.45, 0.95) * sat_bias
        spread = (1.0 - sat_bias) * rng.uniform(0.03, 0.22) + rng.uniform(0.0, 0.01)
        k = 5
        scores = sorted([top - spread * (j / (k - 1)) * rng.uniform(0.6, 1.4)
                         for j in range(k)], reverse=True)
        scores = [min(max(s, 0.0), 1.0) for s in scores]

        # 污染：老 benchmark + 大语料 => 更容易逐字泄漏
        contam = rng.choice([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.40])
        benchmarks.append(Benchmark(
            name=nm, domain=dom, n_test=n_test, scores=scores,
            private_test=private,
            expert_curated=curated,
            age_months=age, contamination_ratio=contam,
        ))
        # 条目与语料
        for j in range(4):
            salt = hashlib.blake2b(f"{nm}-{j}".encode(), digest_size=5).hexdigest()
            body = (f"{nm} item {salt} question {j} about {dom} reasoning with multiple "
                    f"candidate answers and a long tail of distractors that models "
                    f"frequently get wrong on section {salt[:6]}")
            items.append(body)
            if rng.random() < contam:
                if rng.random() < 0.5:
                    corpus.append(body)          # 逐字泄漏（13-gram 过滤器能抓到）
                else:
                    corpus.append(_paraphrase(body))  # 改写泄漏（逐字过滤器抓不到）
                corpus.append(body + " see also " + " ".join(
                    random.Random(j).choice(names) for _ in range(6)))
            else:
                corpus.append(f"unrelated discussion of {dom} in general practice "
                              f"and history section {salt[:6]}")
    # 少量自然语料噪声
    corpus += ["the quick brown fox jumps over the lazy dog " * 8] * 20
    return benchmarks, corpus, items


# ---------------------------------------------------------------- 主流程
def main() -> None:
    bms, corpus, items = synthesize_world()
    corpus_idx = build_corpus_index(corpus)

    print("=" * 72)
    print("Benchmark 饱和度 × 训练污染 联合体检器")
    print("来源：ICML 2026 arXiv:2602.16763 (S_index) + 2026 污染检测方法综述")
    print("=" * 72)

    # --- 真实污染测量（用合成条目验证检测器本身有效） ---
    ratios = [contamination_ratio(it, corpus_idx) for it in items]
    probe_hits = sum(paraphrase_probe(it, corpus_idx) for it in items)
    print("\n[1] 13-gram 污染检测器自检")
    print(f"    语料文档数        : {len(corpus)}")
    print(f"    检测条目数        : {len(items)}")
    print(f"    命中率 mean/max   : {sum(ratios)/len(ratios):.3f} / {max(ratios):.3f}")
    print(f"    同义改写逃逸数    : {probe_hits}/{len(items)}  "
          f"(改写后仍逐字命中 = {probe_hits}；>0 说明改写可绕过逐字过滤器)")

    reports = [assess(b) for b in bms]

    # --- 分布统计 ---
    bins: Dict[str, int] = {}
    for r in reports:
        bins[r.bin] = bins.get(r.bin, 0) + 1
    high = sum(1 for r in reports if r.bin in ("high", "very_high"))
    vhigh = sum(1 for r in reports if r.bin == "very_high")
    print("\n[2] 饱和度分布（n=60）")
    print(f"    high+very_high : {high}/60   (论文实测 29/60)")
    print(f"    very_high      : {vhigh}/60   (论文实测 14/60)")
    for k in ("very_low", "low", "moderate", "high", "very_high"):
        print(f"    {k:<12}: {bins.get(k, 0):>2}  {'#' * bins.get(k, 0)}")

    # --- 饱和 vs 停滞 的定义内部不一致 ---
    inconsistent = [r for r in reports if r.stale_measurement]
    print("\n[3] 定义一致性检查：S_index 高但未逼近经验天花板")
    print(f"    落入不一致区的 benchmark: {len(inconsistent)}/{len(reports)}")
    for r in inconsistent[:5]:
        print(f"      - {r.name:<24} S={r.s_index:.3f}  top1<ceiling  => "
              f"'饱和'实为'停滞'（测量失效 ≠ 任务解决）")

    # --- 年龄/规模/私有集/策展 与饱和的相关性 ---
    def corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / (dx * dy) if dx and dy else 0.0

    ages = [b.age_months for b in bms]
    sidx = [r.s_index for r in reports]
    nt = [math.log10(b.n_test) for b in bms]
    print("\n[4] 饱和度驱动因子相关性 (Pearson r)")
    print(f"    r(age, S_index)      = {corr(ages, sidx):+.3f}   (论文: 年龄↑ ⇒ 饱和↑)")
    print(f"    r(log n_test, S)     = {corr(nt, sidx):+.3f}   (论文: 小测试集 ⇒ 更饱和)")
    priv = [1.0 if b.private_test else 0.0 for b in bms]
    cur = [1.0 if b.expert_curated else 0.0 for b in bms]
    print(f"    r(private_test, S)   = {corr(priv, sidx):+.3f}   (论文: 私有集几乎无保护)")
    print(f"    r(expert_curated, S) = {corr(cur, sidx):+.3f}   (论文: 专家策展是唯一有效保护)")

    # --- 决策建议 ---
    trust = {"TRUST": 0, "CAUTION": 0, "DISCARD": 0}
    for r in reports:
        trust[r.trust] += 1
    print("\n[5] 选型决策：哪些 benchmark 还能用来做模型选型")
    for k, v in trust.items():
        print(f"    {k:<8}: {v:>2}  {'#' * v}")
    good = [r for r in reports if r.trust == "TRUST"]
    print(f"    可信集（建议作为唯一选型依据）: "
          f"{', '.join(r.name for r in good[:8])}{' ...' if len(good) > 8 else ''}")

    # --- 最差的三个：饱和 + 高污染 ---
    worst = sorted(reports, key=lambda r: -(r.s_index + r.contamination_ratio))[:3]
    print("\n[6] 最危险的三个（高饱和 + 高污染 => 分数几乎全是记忆）")
    for r in worst:
        print(f"    {r.name:<24} S={r.s_index:.3f} contam={r.contamination_ratio:.2f} "
              f"=> {r.trust} (必须弃用)")

    out = {
        "meta": {"alpha": ALPHA, "k": K, "ngram": NGRAM, "z": Z,
                 "ceiling_ratio": CEILING_RATIO, "n_benchmarks": len(bms)},
        "distribution": bins,
        "high_or_very_high": high, "very_high": vhigh,
        "definition_inconsistency": len(inconsistent),
        "correlations": {
            "age_vs_sindex": round(corr(ages, sidx), 3),
            "logn_test_vs_sindex": round(corr(nt, sidx), 3),
            "private_vs_sindex": round(corr(priv, sidx), 3),
            "curated_vs_sindex": round(corr(cur, sidx), 3),
        },
        "trust_buckets": trust,
        "contam_probe": {"mean": round(sum(ratios) / len(ratios), 4),
                         "max": round(max(ratios), 4),
                         "paraphrase_hits": probe_hits, "items": len(items)},
        "reports": [asdict(r) for r in reports],
    }
    path = "benchmark_saturation_contamination_results_20261122.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[7] 结果已写入 {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
