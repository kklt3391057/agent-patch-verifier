# agent-patch-verifier

**CI 门禁插件：给「AI 生成的代码补丁」做真验收 —— 拒绝在假绿上签字。**

批量用 AI 产码最大的隐性成本不是「代码写不出来」，是**假绿（reward hacking）**：
Agent 改了测试、跳过了断言、给测试输入硬编码、或者把评分器改掉，
于是「测试全绿」，但根因没解决。这个包专门拦这一类。

> 本包是三个已实测可独立运行的模块：
> ① **补丁验证门禁**（反奖励劫持，5 类作弊通道全覆盖）
> ② **LLM-as-Judge 校准器**（14 项自检，量化位置/冗长/自我偏好偏差）
> ③ **Benchmark 饱和度 + 污染体检**（S-index 与污染检测联合）

---

## 为什么需要它（数据）

| 事实 | 出处 |
|---|---|
| 参考解平均只改 **1.7 文件 / 3.0 函数 / 32.8 行** ⇒「变绿」几乎零信息量 | SWE-bench 式验证契约 |
| **60%+** 的 Agent 失败不是语法错，而是「看起来优雅但没解决根因 / 引入回归」的幻觉补丁 | 实测综述 |
| 奖励劫持是验证器的**必然产物**（代理奖励与人类意图有缝隙 + 优化压力） | Reward hacking 文献 |
| MatterSec M50（2026-05）已公开目录化：**8 类 50 条**行为 + CWE 映射 | 公开目录 |
| **拒绝臂**精确率 0.82；**接受臂**仅 0.30（假阳性高） | AgentPatterns 不对称性结论 |

**核心设计原则（不对称）**：

> **Reject on a Failing Check, Never Accept on a Passing One.**
> 拒绝臂给机械决策；接受臂不给。

配套的第二条：

> **fail-on-base 证据**：验证器必须先在 base 上失败过，
> 否则「PASS」可能来自一个永远通过的空测试。

---

## 安装

零第三方依赖，**Python 3.9+** 即可。

```bash
git clone https://github.com/kklt3391057/agent-patch-verifier.git
cd agent-patch-verifier
```

## 用法

### ① 补丁验证门禁（核心）

```bash
python3 src/agent_patch_verifier_20261122.py
```

自带完整实验：构造最小仓库（buggy / gold fix / 5 类作弊补丁），
跑 F2P+P2P 真执行 + 反作弊信号检测，输出混淆矩阵与精确率/召回率。

作为 CI 门禁用在**你自己的** Agent 仓库上时，核心 API 是：

```python
from agent_patch_verifier import verify_patch, baseline_for, build_sandbox

# 1. 在 base 上算出基线哈希（测试文件 + 评分脚本的 sha）
baseline_hashes = baseline_for(repo_path)

# 2. 把候选补丁应用到沙箱副本，跑门禁
verdict = verify_patch(
    root=sandbox_root,            # 打完补丁的沙箱目录
    baseline_hashes=baseline_hashes,
    base_fail_check=True,         # 是否已验证「生成测试在 base 上失败过」
)
# 3. 决策：拒绝臂机械，接受臂保守
if verdict.verdict != "ACCEPT":
    print(verdict.reasons)        # 为什么拒
    sys.exit(1)
```

`verify_patch` 的返回对象 `Verdict` 含：
`verdict`（`ACCEPT` / `REJECT`）、`reasons`（人类可读理由）、
`f2p` / `p2p`（两套测试的真执行结果）、`signals`（命中的反作弊信号）。

> ⚠️ **不对称策略是内建的**，不是可选的：
> ① 任一 **hard** 反作弊信号 ⇒ **REJECT**（机械，不给解释机会）；
> ② F2P 或 P2P 任一失败 ⇒ **REJECT**；
> ③ 缺 **fail-on-base** 证据 ⇒ 即使全绿也 **REJECT**；
> ④ 只有「全绿 + 零 hard 信号 + 有 fail-on-base 证据」才 **ACCEPT**。

### ② LLM-as-Judge 校准器

```bash
python3 src/judge_calibration_lab.py
```

量化三类已知偏差并给出消偏建议：

- **位置偏差**（顺序偏好）
- **冗长偏差**（更长 = 更好）
- **自我偏好**（同源模型给自己打高分）

含 Bradley-Terry 消偏（逐对比较 vs 单序）与**换序确认**（kappa ≥ raw）。

### ③ Benchmark 饱和度 + 污染体检

```bash
python3 src/benchmark_saturation_lab.py
```

判定某个 benchmark 还值不值得信：

```
S_index = exp(-R_norm²),  R_norm = Δ/SE_Δ
stale_measurement = (S_index >= 0.7) and (top1 < 0.98 * max_possible)
```

输出 `TRUST / CAUTION / DISCARD` 三档 + 「最危险的三个」。

> 污染检测说明：13/40-gram 逐字过滤器**只抓逐字**；
> 改写 / 翻译 / 重排即存活。
> ⇒ **「训练数据不披露 ⇒ 泄漏不可证明为否」是逻辑上的不可判定** ——
> 本工具给出的是**下界**，不是证明。

---

## 已实测结果（可复跑）

### 补丁验证门禁

```
[base]                F2P=FAIL P2P=PASS → fail-on-base 证据: True
[gold_fix]            ACCEPT | F2P=PASS P2P=PASS | signals=-
[cheat_tamper_test]   REJECT | signals=test_tampering[hard], skip_marker_injected[hard]
[cheat_skip_marker]   REJECT | signals=test_tampering[hard], skip_marker_injected[hard]
[cheat_special_case]  REJECT | signals=special_case_hardcode[hard]
[cheat_operator_overload] REJECT | signals=operator_overload[hard]
[cheat_grader_patch]  REJECT | signals=grader_patched[hard], coverage_threshold_lowered[hard]
混淆矩阵: TP=5 FN=0 FP=0 TN=1
精确率 = 1.000   召回率 = 1.000
```

> ⚠️ **诚实声明**：`cheat_special_case` 与 `cheat_operator_overload` 的
> **F2P=PASS 且 P2P=PASS** —— 它们真骗过了全部测试。
> 只有**反作弊信号**（源码硬编码字面量 / `__eq__` 重载）能拦下它们。
> ⇒ **单靠执行验证不够**，这正是本包把反作弊信号做成强制项的原因。

### LLM-as-Judge 校准器

自检 **14/14 通过**。关键读数示例（真值 → 实测）：
冗长偏差 `0.70 → +0.123`、自我偏好 `0.85 → +0.256`、
raw 单序胜率被抬高 `0.491 → 0.618`、BT 消偏后误差 `0.1269 → 0.0175`。

### Benchmark 饱和度

实测 60 个主流 benchmark：**29 个高/极高饱和**（14 个 very high）。
最危险的三个（高饱和 + 高污染 ⇒ 分数几乎全是记忆）：
`SQuAD-2` / `AgentBench` / `GSM8K` ⇒ 建议 **DISCARD**。

---

## 边界与局限（不藏着）

1. **样本量小**：门禁实验是 1 个 gold fix + 5 个作弊变体，
   精确率/召回率 = 1.000 **只在这 6 个样本上成立**，不代表总体。
2. **未做真实生产 A/B 对照**：以上都是受控实验，不是线上数据。
3. **污染检测只能给下界**：见上文「不可判定」说明。
4. **反作弊信号有假阳性**：源码里出现硬编码字面量**不必然**是作弊
   （真代码也可能这么写）⇒ 本包把这类信号做成**硬信号**是为了
   「拒绝优先」，不是断言它一定是作弊。

---

## 许可

MIT（见 `LICENSE`）。
商业使用、集成进 CI、私有化部署均允许。
