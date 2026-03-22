# 手写词语识别（PaddleOCR v5）

这个小程序用于输入一张图片，输出识别到的词语数组（JSON）。

## 1. 建立虚拟环境

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 运行

```bash
python extract_words.py --image ./your_image.jpg --pretty
```

可选参数：

- `--lang`：语言，默认 `ch`
- `--min-score`：最低置信度阈值（0~1），例如 `0.5`
- `--segment-cn / --no-segment-cn`：中文分词开关，默认开启（会把连写中文拆成词语）
- `--multi-pass / --no-multi-pass`：多路预处理识别开关，默认开启（提升漏字召回）
- `--pretty`：格式化 JSON 输出

## 3. 输出示例

```json
["苹果", "香蕉", "学校"]
```
