# REPRODUCE — Qwen3-32B EAGLE-3 域训 draft head 复现指南

面向零上下文新人:从 conda 环境重建到复现 phase2 训练与部署,一步一命令。本文件与环境备份都在 **`backup-env` 分支**(你现在看到本文,说明已经在这条分支上)。

产物概述:Qwen3-32B 的 deep-research summary 域训 EAGLE-3 draft head(phase2:TP4 @ max-length 12288 + 双裁剪 + 3 epoch),训练框架 = 本 SpecForge fork(上游 [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge))。

前提:Linux + CUDA GPU(完整训练需 **4x 48GB**;冒烟测试 CPU / 单卡即可)、conda、能访问 HuggingFace(其中一个数据仓私有,需授权 token)。

---

## 1. 分支地图(先看这个)

fork 仓 [julyanghar/SpecForge](https://github.com/julyanghar/SpecForge) 共 10 条工作分支 + `main`。`main` 只是 fork 时的上游快照(`40d8fef`),**别在 main 上找我们的改动**。下表"独有 commit 数"以 sgl-project/SpecForge main(截至 2026-08-16,`e6440f0`)为参照:

| 分支 | 独有 commits | head | 一句话 |
|---|--:|---|---|
| `backup-env` | — | (随修订漂移,以 `git log` 实见为准) | **本指南所在**:= `pr-trim-a-v3` + `env/` conda 环境备份 + 本文件;9 个打补丁前 `.bak` 原件快照恒在 commit `81e02c1` |
| `pr-trim-a` / `pr-trim-a-v3` | 2 | `efbc096` | **上游 PR [#705](https://github.com/sgl-project/SpecForge/pull/705) 现行线**(head=`julyanghar:pr-trim-a`,v3 是同 commit 的定版别名):`training.trim_loss_positions`(A 级裁剪,unified-runtime 版)+ SP-native USP 兼容 |
| `pr-trim-a-v2` | 1 | `ba20730` | #705 中间版存档(rebase 到上游 #665 基座) |
| `pr-trim-a-v1` | 2 | `f31dce1` | #705 初版存档(旧 `scripts/train_eagle3.py` 时代 + 首轮 review 修改) |
| `pr-ropebuf` / `pr-ropebuf-v2` | 0(已并入上游) | `7a1040f` | **上游 PR [#703](https://github.com/sgl-project/SpecForge/pull/703)**:ropebuf nan 修复,2026-07-24 已 merge(上游 merge commit `fec8f85`) |
| `pr-ropebuf-v1` | 2 | `66825ad` | #703 合并前存档 |
| `trim-usp-dev` | 4 | `605d56c` | trim x USP 兼容开发线(wip + 探针 + cleanup + golden/analytic/4-rank 测试),v3 的孵化线 |
| `yilin-trim-mem-probes` | 2 | `78a6432` | **phase2 训练实际代码线(复现训练用这条)**:训练时代旧基座(仍有 `scripts/train_eagle3.py`)+ 双裁剪 A/B-i(另含 B-ii 开关 `--trim-step1`,默认关、phase2 未用)+ 四补丁 + MCMEM 显存探针 |

关键澄清(git 实核过,新人最容易在这里绕晕):

- **跑 phase2 训练命令必须用 `yilin-trim-mem-probes`**。上游后来把 `scripts/train_eagle3.py` 重构掉了("consolidate training on unified runtime",`d5883d5`),所以 `pr-trim-a-v3` 上没有这个脚本、也没有 B-i 裁剪(`--trim-prompt-rows`);v3 只是把 A 级裁剪面向上游 PR 化的贡献线。
- `backup-env` 的 `.bak` 快照(commit `81e02c1`,如 `specforge/core/lk_loss.py.bak_chunkacc`)是各补丁**打前**的原件,用于和 `yilin-trim-mem-probes` 上的补丁版 diff 对照,不是可运行代码。

## 2. 资源位置

| 资源 | 位置 | 说明 |
|---|---|---|
| 训好的 phase2 head + 二期训练数据 | HF model repo [julyanghar/Efficient-DRAgent](https://huggingface.co/julyanghar/Efficient-DRAgent)(公开) | `model.safetensors`(1.4G)+ `config.json`(训练 config)+ `config_deploy.json`(部署 config)+ `training_state.pt`(续训)+ `data/train_main.jsonl`(**7,681 条**)+ `data/heldout.jsonl`(46 条);其 README = phase2 权威文档(训练参数 + 部署命令出处) |
| 大数据备份 | HF dataset repo [julyanghar/Efficient-DRAgent-data](https://huggingface.co/datasets/julyanghar/Efficient-DRAgent-data)(**私有**) | `train-Eagle3-data/` = 一期训练数据(raw/main/longtail/heldout/manifest)+ `tars/` 12 个 tar.zst(search_cache、benchmark 结果、实验物证等);取用前按 `tars/SHA256SUMS` 校验 |
| warm-start 起点权重 | HF [AngelSlim/Qwen3-32B_eagle3](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3)(公开,已核实存在) | 腾讯开源第三方 EAGLE-3 head(`config.json` + `pytorch_model.bin`),非本项目产出 |
| target 模型 | HF [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)(公开,已核实存在) | 训练与部署的 target |
| agent 本体 + 实验文档 | [julyanghar/deep_researcher_demo](https://github.com/julyanghar/deep_researcher_demo)(公开) | DRAgent 代码、`exp-docx/` 实验档案(见 §8 导航) |
| serving 侧 vLLM patch 线 | `julyanghar/LMCache-yilin`(**私有**,工作分支名就叫 `branch`) | 仓内 `vllm_patch_backup/README.md`(suffix/eagle3 router 等 serving 改动,与本仓训练无关) |
| 本 fork | [julyanghar/SpecForge](https://github.com/julyanghar/SpecForge)(公开) | 分支见 §1 |

## 3. 环境重建(conda env `specforge`)

细节以 [env/README.md](env/README.md) 为准(py3.12,torch 2.11.0+cu130,sglang 0.5.14;flash-attn 社区轮说明也在那里),这里只给主干:

```bash
git clone https://github.com/julyanghar/SpecForge.git
cd SpecForge
git checkout backup-env          # env/ 备份只在这条分支

conda create -n specforge python=3.12 -y
conda activate specforge
grep -v '^specforge==' env/requirements-specforge.txt > /tmp/requirements-no-self.txt   # specforge 本身走 editable 安装,不从 freeze 装
pip install -r /tmp/requirements-no-self.txt
pip install -e .
```

注意:

- freeze 里的 `torch==2.11.0` 在本机实为 **cu130** 构建;若默认 PyPI 源装到的 CUDA 版本与你机器不符,按 pytorch.org 安装页用对应 CUDA index 重装(如 `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`,此命令为通用惯例、未在本机回归验证)。
- **flash-attn 不是训练必需**(训练用 `--attention-backend flex_attention`,缺 flash_attn 时会打 warning 并回退,无害);只有 §5 的 4 卡 USP 等价性测试需要它,装法见 [env/README.md](env/README.md)。

## 4. 数据与权重就位

```bash
export HF_TOKEN=<your-hf-token>   # 私有 dataset 仓需有权限的 token;公开仓可不设

hf download Qwen/Qwen3-32B                 --local-dir <models>/Qwen3-32B
hf download AngelSlim/Qwen3-32B_eagle3     --local-dir <models>/Qwen3-32B-eagle3-angelslim
hf download julyanghar/Efficient-DRAgent   --local-dir <models>/Efficient-DRAgent   # 含训练数据 data/train_main.jsonl

# 可选:一期训练数据 / 实验数据备份(私有仓)
hf download julyanghar/Efficient-DRAgent-data --repo-type dataset --local-dir <data>/DRAgent-data
```

(`hf` CLI 随 `huggingface_hub` 已在 env 里;`<models>`、`<data>` 换成你的本地路径。)

### 数据生成链路(可选,只在想从原始轨迹重建数据时需要)

二期 7,681 条(`data/train_main.jsonl`,zh 3935 / en 3725 / mix 21,<=12288 token)由
[convert_harvest_to_eagle3.py](https://github.com/julyanghar/deep_researcher_demo/blob/main/train/Eagle3/convert_harvest_to_eagle3.py)
(公开仓,已 force-add 入 git)从 DRAgent 跑批轨迹(harvest 的 `RESEARCH_SUMMARY_TEXT`)生成:按 user 内容去重 + 过滤(finish_reason=stop 且 output>=50 字符)+ 题面隔离 held-out(md5 排序确定性,重跑一致)+ 长度分桶。输入轨迹在 dataset 仓 `tars/`(注意其中 `drbench_trash.tar.zst` 是训练集 49.5% 的来源——目录名叫 TRASH 但**不可弃**,谱系见 deep_researcher_demo 仓 [PROVENANCE.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/PROVENANCE.md))。直接用 HF 上的现成 jsonl 不受影响。

## 5. 最小冒烟(不动大模型,验证环境 + 裁剪数学)

在 `backup-env`(或 `pr-trim-a-v3`)上,以下三条 2026-08-16 已实测通过:

```bash
conda activate specforge
cd SpecForge   # backup-env 分支

# CPU,~3s,4 tests OK:trim 行选择数学的手推 golden 表
python -m unittest -v tests.test_runtime.test_equiv_trim_usp.TestTrimPackGolden

# 1 GPU,~10s,1 test OK:trim on/off 逐步 loss 等价
python -m unittest -v tests.test_runtime.test_equiv_trim_loss_positions

# 1 GPU,~2s,1 test OK:闭式解析常数校验
python -m unittest -v tests.test_runtime.test_equiv_trim_usp.TestTrimLossAnalytic
```

可选(4x GPU + flash-attn,缺 flash-attn 会自动 skip):`python -m unittest -v tests.test_runtime.test_equiv_trim_usp.TestEquivTrimUspFourRank`。

## 6. 完整 phase2 训练(4x 48GB)

切到训练线并重装 editable 包:

```bash
git checkout yilin-trim-mem-probes
pip install -e .
```

训练命令。参数逐条照抄自 [julyanghar/Efficient-DRAgent 的 README](https://huggingface.co/julyanghar/Efficient-DRAgent)"训练配置(复现)"一节(= 本地 phase2-12k-3ep/README.md);`--output-dir` 是脚本必填参数,原文省略、此处补上:

```bash
torchrun --nproc_per_node=4 scripts/train_eagle3.py \
  --target-model-path <models>/Qwen3-32B \
  --ckpt-dir <models>/Qwen3-32B-eagle3-angelslim \
  --train-data-path <models>/Efficient-DRAgent/data/train_main.jsonl \
  --chat-template qwen3-instruct \
  --target-model-backend sglang --tp-size 4 --sglang-mem-fraction-static 0.44 \
  --max-length 12288 --ttt-length 3 --attention-backend flex_attention \
  --shard-target-output --trim-loss-positions --trim-prompt-rows \
  --learning-rate 2e-5 --num-epochs 3 --warmup-ratio 0.05 --max-grad-norm 0.5 \
  --draft-accumulation-steps 16 --save-interval 120 \
  --output-dir <out>/qwen3-32b-summary-12k-trim
```

- 想先小跑一轮:`head -n 24 <models>/Efficient-DRAgent/data/train_main.jsonl > /tmp/smoke24.jsonl`,把 `--train-data-path` 换成它、`--num-epochs 1`,其余不动。
- 参考指标(原 run,epoch_2_step_5760 即发布权重):逐 epoch 训练 acc **0.44 -> 0.47 -> 0.48**,全程 0 OOM,显存峰值 ~46-47GB/卡。
- 双裁剪(`--trim-loss-positions --trim-prompt-rows`)是 12288 能开起来的前提(省 7.05 GiB/卡),别省略。

## 7. 四补丁 + 双裁剪速查

均已提交在 `yilin-trim-mem-probes`(commit `77f4a0f`;打补丁前原件见 `backup-env` 的 `.bak` 快照)。详细踩坑账本:deep_researcher_demo 仓 [eagle3-domain-training-plan.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/exp-docx/eagle-spec-decode/eagle3-domain-training-plan.md) §四c #8-#14。

四补丁(训练稳定性/显存):

1. **ropebuf**(`scripts/train_eagle3.py`):transformers meta-device 加载不回填 non-persistent buffer,warm-start(`--ckpt-dir`)后 RoPE `inv_freq/cos/sin` 是未初始化内存导致 loss=nan;修复 = `from_pretrained` 后对每个 attention 重跑 `_init_rope()`。已单独 PR 进上游(= [#703](https://github.com/sgl-project/SpecForge/pull/703),已合并)。
2. **chunk-acc**(`specforge/core/lk_loss.py`):acceptance_rate 指标的整条 fp32 softmax 瞬时 ~3.2GiB@13.5K token,按 2048 位置分块,逐元素等价。
3. **nocompile**(`specforge/modeling/draft/llama3_eagle.py`):摘除 RMSNorm / `apply_rotary_pos_emb` / RotaryEmbedding 三处 `@torch.compile(dynamic=True)`——变长 batch 重编译 x FSDP x 梯度累积组合触发 `InternalTorchDynamoError`。
4. **ckpt-norm**(同上文件):nocompile 使 eager norm 的 fp32 中间量回到 autograd 账本(+~4GB@12K),给 5 处 norm 套 `torch.utils.checkpoint(use_reentrant=False)` 反向重算,省 ~4.3GB。

双裁剪(位置维,phase2 解锁 12288 的主力,全档见 [phase2-trimming-work.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/exp-docx/eagle-spec-decode/phase2-trimming-work.md)):

- **A 级 `--trim-loss-positions`**:teacher target_p + draft logits/loss 只在监督(loss-mask)位置计算并重标定 mean 分母,数学等价;实测省 3.37 GiB/卡@8192。上游贡献版 = PR [#705](https://github.com/sgl-project/SpecForge/pull/705)(`pr-trim-a-v3`,unified-runtime 配置化 + USP 兼容)。
- **B-i `--trim-prompt-rows`**:TTT 步 2..k 只前向监督行(compact flex mask + RoPE 绝对位),prompt 行仅作步 1 KV 上下文;再省 3.68 GiB/卡。仅在 `yilin-trim-mem-probes`。

随行小修(同在 `77f4a0f`,非"四补丁"但训练必需):warm-start 时跳过 `load_vocab_mapping` 覆盖(保留 ckpt 的 t2d/d2t);sglang `ServerArgs` 加 `chunked_prefill_size=-1`(防 `_extend` 整批提交与 chunked buffer 尺寸崩溃)。

## 8. 部署(vLLM 零转换)

训好的 head(或直接用 HF 下载的 `<models>/Efficient-DRAgent`)可被 vLLM 直接加载(实测 vLLM 0.18 零转换)。命令出处同 §6 的 HF README"部署"节:

```bash
# config_deploy.json = AngelSlim 版 config,避开新 transformers 的 rope 解析差异;
# HF 目录自带可加载 config,保险起见按 README 用 deploy 版替换:
cp <models>/Efficient-DRAgent/config_deploy.json <models>/Efficient-DRAgent/config.json

vllm serve <models>/Qwen3-32B --tensor-parallel-size 4 \
  --speculative-config '{"method":"eagle3","model":"<models>/Efficient-DRAgent","num_speculative_tokens":3}'
```

(vLLM 环境不在本仓 env 备份内;agent 端到端 serving 与 suffix/eagle3 router 属 deep_researcher_demo + LMCache-yilin 私有仓的 `vllm_patch_backup/README.md`,超出本指南范围。)

## 9. 相关文档导航

本仓内:

- [env/README.md](env/README.md) — conda env 重建细节 + flash-attn 社区轮
- `*.bak_*` 9 个快照(`scripts/`、`specforge/` 下)— 各补丁打前原件
- `tests/test_runtime/test_equiv_trim_usp.py` / `test_equiv_trim_loss_positions.py` — 裁剪等价性测试(§5)

deep_researcher_demo 仓(公开):

- [phase2-trimming-work.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/exp-docx/eagle-spec-decode/phase2-trimming-work.md) — 双裁剪动机/原理/验收全档
- [eagle3-domain-training-plan.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/exp-docx/eagle-spec-decode/eagle3-domain-training-plan.md) — 训练配方演化 + §四c 踩坑账本(四补丁出处)
- [phase2-results-summary.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/exp-docx/eagle-spec-decode/phase2-results-summary.md) / [experiment-data-summary.md](https://github.com/julyanghar/deep_researcher_demo/blob/main/exp-docx/paper-submission/experiment-data-summary.md) — 训练与评测结果(DRGym 干净 held-out 1.303x 等)
- [exp-docx/spec-train-optimization/](https://github.com/julyanghar/deep_researcher_demo/tree/main/exp-docx/spec-train-optimization) — PR #703/#705 的审查往来与定位

上游 PR:[#703](https://github.com/sgl-project/SpecForge/pull/703)(ropebuf,已合并)、[#705](https://github.com/sgl-project/SpecForge/pull/705)(trim-A + USP,open)。
