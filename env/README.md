# conda env 备份:specforge

EAGLE-3 域训环境(target backend = sglang)。

**注意**:本快照拍摄于 **2026-08-16**,不是 phase2 训练(2026-07-14)当时的环境——
中间有包漂移(如 flash-attn 系列版本;训练时 flash_attn 实际 import 失败,
fallback 到 flex_attention,证据见 deep_researcher_demo 仓库
train/Eagle3/run-artifacts/retrain.log 的 UserWarning 行)。训练当时的权威参数记录是
training_state.pt 的 args 字段(79 项,HF julyanghar/Efficient-DRAgent),
代码版本 = 本仓库 commit `77f4a0f`(yilin-trim-mem-probes 分支)。

| 文件 | 说明 |
|---|---|
| `environment-specforge.yml` | `conda env export --no-builds`(conda 层) |
| `requirements-specforge.txt` | `pip list --format=freeze`(真实 pip 版本,以这份为准) |

- Python 3.12.13,torch 2.11.0+cu130。
- `specforge==0.2.0` 是本仓库的 editable 安装(`pip install -e .`),重建时在仓库根执行,别从 freeze 里装。

## flash-attn

官方预编译轮滞后 torch 时用 mjun0812 社区轮:
https://github.com/mjun0812/flash-attention-prebuild-wheels
(ABI 不匹配的症状 = import 即 undefined symbol;本地轮文件名不可改)。

## 重建

```bash
conda create -n specforge python=3.12 -y
conda activate specforge
pip install -r requirements-specforge.txt   # 先手动删掉 specforge 那行
pip install -e /path/to/SpecForge
```
