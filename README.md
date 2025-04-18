# TemplateGenerate: 现病史自动生成系统

**TemplateGenerate** 是一个基于大型语言模型 (LLM) 的系统，旨在根据患者的主诉自动生成符合规范的现病史。本项目利用 Qwen2.5-7B-Instruct 模型，并通过 LoRA 技术进行微调，以适应医疗领域的特定需求。

✨ **主要功能:**

*   基于主诉自动生成结构化的现病史。
*   支持使用 LoRA 微调的模型或原始预训练模型进行推理。
*   提供 Gradio Web 界面进行交互式演示和测试。
*   提供 Flask API 构建的 API 接口，方便集成。
*   支持模型训练、参数配置和历史记录导出。

## 🚀 快速开始

### 1. 环境设置

**a. 克隆仓库:**

```bash
git clone <your-repository-url>  # 替换为您的仓库URL
cd <your-repository-directory> # 替换为您的项目目录名
```

**b. 安装依赖:**

确保您已安装 Python (推荐 3.8+) 和 pip。

```bash
pip install -r requirements.txt
```

**c. 下载预训练模型:**

本项目使用 [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 作为基础模型。

```bash
# 安装 Git LFS (如果尚未安装)
# 对于 Debian/Ubuntu:
sudo apt install git-lfs
# 对于其他系统，请参考 Git LFS 官方文档安装

# 初始化 Git LFS
git lfs install

# 克隆模型仓库
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

**d. 移动模型文件:**

将下载的 `Qwen2.5-7B-Instruct` 文件夹移动到项目根目录下的 `model` 文件夹中。

```bash
# 假设模型下载在当前目录
mv Qwen2.5-7B-Instruct ./model/
```

### 2. 配置

所有重要的配置参数（如模型路径、训练参数、设备设置等）都集中在 `config.py` 文件中。请在运行前根据您的环境和需求检查并修改此文件。

### 3. 运行

#### a. 模型训练 (可选)

如果您想使用自定义数据重新训练或继续训练 LoRA 模型：

*   **数据集:** 训练和验证数据分别位于 `./data/train_template.json` 和 `./data/val_template.json`。请确保数据格式符合要求。
*   **配置:** 在 `config.py` 中调整训练轮数、学习率等超参数。
*   **开始训练:**

    ```bash
    python train_template.py
    ```

训练完成后，LoRA 适配器权重将保存在 `config.py` 中指定的输出目录（默认为 `./output/lora`）。

#### b. 模型推理 (命令行)

*   **使用 LoRA 微调模型 (默认):**

    `predict_template.py` 默认加载基础模型和最新的 LoRA 适配器进行推理。

    ```bash
    python predict_template.py
    ```

*   **使用合并后的模型:**

    如果您希望将 LoRA 权重合并到基础模型中以可能提高推理速度或简化部署，请先运行合并脚本：

    ```bash
    python merge.py
    ```

    合并后的模型将保存在 `config.py` 中指定的目录（默认为 `./output/merged_model`）。然后，使用 `--use_lora=False` 参数运行推理脚本（请确认 `predict_template.py` 已支持此参数）：

    ```bash
    # 运行 predict_template.py 并指定不使用 LoRA (即使用合并后的模型)
    python predict_template.py --use_lora=False
    ```

*   **使用原始预训练模型 (无微调):**

    如果您想直接使用未经微调的 Qwen2.5-7B-Instruct 模型进行推理，请运行 `predict_without_lora.py` 脚本（**注意:** 请确保此脚本确实是加载 `./model` 目录下的原始模型）。

    ```bash
    python predict_without_lora.py
    ```

#### c. Web 交互界面

项目提供了基于 Gradio 的 Web 界面，用于演示和测试现病史生成功能。

*   **启动 Web 服务 (使用微调模型):**

    ```bash
    python web/web_demo.py
    ```
    *您也可以通过命令行参数 `--use_lora=False` 来指定使用合并后的模型（如果 `web_demo.py` 支持）。*

*   **启动 Web 测试服务 (使用静态数据):**

    如果您想快速查看界面布局和功能，可以运行测试版本，它不加载实际模型，而是使用预设的静态数据。

    ```bash
    python web/test_web.py
    ```

    启动后，根据命令行提示，在浏览器中打开指定的 URL (通常是 `http://127.0.0.1:7860` 或 `http://0.0.0.0:7860`)。Web 界面支持模型参数调整、历史记录查看和导出（JSON/Excel）。

#### d. API 服务

项目还提供了基于 FlaskAPI 的 API 接口。

*   **启动 API 服务:**

    (请确保 `api/app.py` 文件存在且配置正确)


## 📝 文件结构 (概览)

```
.
├── api/                  # API 相关代码
│   └── app.py            # API 应用主文件
├── data/                 # 数据集
│   ├── train_template.json # 训练数据
│   └── val_template.json   # 验证数据
├── model/                # 存放预训练基础模型
│   └── Qwen2.5-7B-Instruct/ # 下载的模型文件夹
├── output/               # 输出目录 LoRA 适配器权重
├── eval/                 # 测试代码
│   ├── evaluate_model.py       # 评估原始模型和微调后模型，打印相关指标
│   ├── calculate_metrics.py    # 对每一句单独计算指标，生成txt文件
│   └── generate_excel.py       # 根据txt文件生成excel文件
├── lora_output/     	  # 合并后的模型 (如果运行了 merge.py)
├── web/                  # Gradio Web 界面代码
│   ├── web_demo.py       # 主要 Web 应用 (加载模型)
│   ├── test_web.py       # Web 测试版本 (静态数据)
│   └── styles.css        # Web 界面样式
├── config.py             # 项目配置文件
├── merge.py              # LoRA 模型合并脚本
├── predict_template.py   # 使用 LoRA 或合并模型的推理脚本
├── predict_without_lora.py # 使用原始预训练模型的推理脚本 (待确认其具体加载逻辑)
├── requirements.txt      # Python 依赖库列表
├── train_template.py     # 模型训练脚本
└── README.md             # 本文档
```



