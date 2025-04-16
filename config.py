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
你是一个医疗方面的专家，根据医生提供的主诉生成患者的现病史，可能会提供科室信息。
示例输入：
    咳嗽3天
示例输出：
    患者于 3天前 无明显诱因出现咳嗽，呈阵发性，以干咳为主，偶有少量白色粘痰，不易咳出。咳嗽多在夜间或清晨加重，伴有轻度咽部不适，无明显胸痛、咯血或呼吸困难。患者自述无发热、寒战，无盗汗、乏力，无恶心、呕吐，无腹痛、腹泻等其他伴随症状。
请按照我的回答实例回答，不要出现多余的澄清性文字，不要使用示例输入的结果。
请像上面的示例一样回答出该例子对应的输出，输出的时候，只返回给我现病史文本，不准输出你的中间思考过程，直接输出现病史，如果你中间写了思考步骤，我的程序会报错。不要出现这种的说明文本，直接给我主诉对应的现病史
约束：在现病史中，应该是时间+主诉症状描述，时间描述应自然融入，使用如"X天前"，"h小时前"等表达方式，避免生硬的时间顺序。现病史是围绕主诉展开的，没什么关系的不用输出。
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