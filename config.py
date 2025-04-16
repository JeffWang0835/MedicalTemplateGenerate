from dataclasses import dataclass
from typing import List
import os

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path):
    """将相对路径转换为绝对路径"""
    return os.path.join(get_project_root(), relative_path)

@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name: str = "model/Qwen2.5-7B-Instruct"
    lora_dir: str = "output"
    lora_output: str = "lora_output"
    device: str = "cuda:0"  # 或 "cpu"
    trust_remote_code: bool = True
    use_vllm: bool = False  # 是否使用vllm加速
    vllm_tensor_parallel_size: int = 1  # vllm张量并行大小
    vllm_max_num_batched_tokens: int = 32768  # vllm最大批处理token数
    vllm_max_num_seqs: int = 256  # vllm最大序列数
    vllm_trust_remote_code: bool = True  # vllm是否信任远程代码

    def get_model_path(self):
        """获取模型的绝对路径"""
        return get_absolute_path(self.model_name)

    def get_lora_dir(self):
        """获取LoRA目录的绝对路径"""
        return get_absolute_path(self.lora_dir)
    def get_lora_output(self):
        return get_absolute_path(self.lora_output)

@dataclass
class TrainingConfig:
    """训练相关配置"""
    train_json_path: str = "./data/train_template.json"
    val_json_path: str = "./data/val_template.json"
    max_source_length: int = 64
    max_target_length: int = 256
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 16
    num_workers: int = 0

@dataclass
class LoRAConfig:
    """LoRA相关配置"""
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class SystemPrompt:
    """系统提示词配置"""
    content: str = """
你是一位资深医疗专家，精通各科室病历书写，现需根据患者主诉和科室信息生成符合医学规范的现病史。

输入内容：患者主诉 + 科室信息
输出要求：结构完整、专业规范的现病史文本

关于时间处理：
- 若主诉中包含具体时间（如"3天"、"2周"），直接使用该具体时间
- 若主诉中包含占位符（如"d天"、"h小时"、"m月"），保留占位符格式，如"d天前"

现病史撰写要点：
1. 症状起始：精确描述首发症状的起始时间、性质、程度和发展情况
2. 病程演变：按时间顺序描述症状变化，使用"X天前"、"随后"等自然过渡
3. 关联症状：详述伴随症状及其特点，并注明重要的阴性症状
4. 缓解加重：记录症状的缓解或加重因素，如休息、药物、体位变化等
5. 就诊经过：简要描述患者此前就诊情况和治疗效果（如有）
6. 时间表述：使用阿拉伯数字或占位符表示时间（根据输入确定），避免模糊表达
7. 语言风格：使用医学专业术语，保持客观、简洁、完整

请直接输出高质量现病史文本，无需额外解释或分析过程。针对不同科室，适当调整内容重点和专业术语。

示例输入1：咳嗽3天
示例输出1：患者于3天前无明显诱因出现咳嗽，呈阵发性，以干咳为主，偶有少量白色粘痰，不易咳出。咳嗽多在夜间或清晨加重，伴有轻度咽部不适，无明显胸痛、咯血或呼吸困难。患者自述无发热、寒战，无盗汗、乏力，无恶心、呕吐，无腹痛、腹泻等其他伴随症状。

示例输入2：发热d天
示例输出2：患者于d天前出现发热，体温最高达t℃，伴有畏寒、乏力等症状。发热呈持续性，无明显规律性波动。无咳嗽、咳痰、流涕、鼻塞等上呼吸道感染症状，无咽痛、吞咽困难，无腹痛、腹泻、呕吐等消化道症状。患儿精神状态尚可，食欲略有下降，睡眠及大小便正常。
"""

@dataclass
class GenerationConfig:
    """生成相关配置"""
    max_new_tokens: int = 258
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

# 创建默认配置实例
model_config = ModelConfig()
training_config = TrainingConfig()
lora_config = LoRAConfig()
system_prompt = SystemPrompt()
generation_config = GenerationConfig() 