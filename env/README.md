# conda env 备份:specforge

EAGLE-3 域训环境(phase2-12k-3ep 权重的训练环境,target backend = sglang)。

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
