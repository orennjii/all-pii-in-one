import os
import logging
import json
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import tempfile
import shutil

from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification,
    pipeline
)
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from .base_client import BaseLLMClient

logger = logging.getLogger("llm_recognizer.local_llm_client")


class LocalLLMClient(BaseLLMClient):
    """
    本地LLM客户端实现，支持命名实体识别任务。
    
    可以加载本地的Hugging Face模型进行推理，并支持对模型进行微调。
    支持两种模式：
    1. Token分类模式：使用如BERT等模型进行传统的NER标注
    2. 指令跟随模式：使用如Llama等本地大模型进行生成式的NER
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        mode: str = "token_classification",  # 'token_classification' 或 'instruction'
        max_length: int = 512,
        device: str = None,
        quantization: Optional[str] = None,  # 'int8', 'int4', None
        entity_labels: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        初始化本地LLM客户端。
        
        Args:
            model_name: 模型名称或本地路径
            mode: 运行模式 - token_classification: 传统NER，instruction: 生成式NER
            max_length: 最大序列长度
            device: 运行设备（'cpu', 'cuda:0'等），None表示自动选择
            quantization: 是否使用量化（'int8', 'int4' 或 None）
            entity_labels: NER标签列表（仅token_classification模式需要）
            cache_dir: 模型缓存目录
            **kwargs: 额外参数
        """
        self.mode = mode
        self.max_length = max_length
        self.entity_labels = entity_labels or []
        self.cache_dir = cache_dir
        self.quantization = quantization
        
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        super().__init__(
            model_name=model_name,
            api_key=None,  # 本地模型不需要API密钥
            **kwargs
        )
        
    def _setup_client(self) -> None:
        """设置模型和分词器"""
        logger.info(f"正在加载本地模型: {self.model_name}, 模式: {self.mode}, 设备: {self.device}")
        
        try:
            load_kwargs = {"device_map": self.device}
            
            # 处理量化选项
            if self.quantization:
                if self.quantization == "int8":
                    load_kwargs["load_in_8bit"] = True
                elif self.quantization == "int4":
                    load_kwargs["load_in_4bit"] = True
                logger.info(f"使用{self.quantization}量化加载模型")
            
            # 根据不同模式加载不同类型的模型
            if self.mode == "token_classification":
                # 加载用于标记分类的模型 (如BERT/RoBERTa for NER)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir
                )
                
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir,
                    **load_kwargs
                )
                
                # 创建NER pipeline
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple"
                )
                
                if hasattr(self.model.config, "id2label"):
                    self.id2label = self.model.config.id2label
                    self.label2id = self.model.config.label2id
                    logger.info(f"已加载模型标签映射: {list(self.id2label.values())}")
                else:
                    logger.warning("模型没有标签映射。如果这是预训练模型，请在微调时提供entity_labels")
                
            else:  # instruction模式
                # 加载用于生成任务的大模型 (如Llama, Baichuan等)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True, 
                    **load_kwargs
                )
            
            logger.info(f"模型 {self.model_name} 加载完成")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        使用本地LLM生成响应。
        
        Args:
            prompt: 输入提示词
            temperature: 生成温度（针对instruction模式）
            max_tokens: 最大生成token数
            **kwargs: 额外参数
            
        Returns:
            str: 生成的响应
        """
        if self.mode == "token_classification":
            # 使用NER pipeline进行实体识别
            entities = self.ner_pipeline(prompt)
            
            # 将NER结果转换为所需的JSON格式
            results = []
            for entity in entities:
                results.append({
                    "entity_type": entity["entity_group"],
                    "text": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": float(entity["score"])
                })
            
            # 返回JSON字符串
            return json.dumps(results, ensure_ascii=False, indent=2)
            
        else:  # instruction模式
            # 使用生成模型进行推理
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            max_new_tokens = max_tokens or 256
            
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            }
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **tokens,
                    **generation_config
                )
            
            # 解码文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除prompt部分，只保留生成的内容
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            return generated_text
    
    def generate_with_json_response(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON格式的结构化响应。
        
        Args:
            prompt: 输入提示词
            schema: 预期的JSON格式
            temperature: 生成温度
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 生成的JSON响应
        """
        # 在instruction模式下，我们可以将schema添加到prompt中
        if self.mode == "instruction":
            json_prompt = (
                f"{prompt}\n\n"
                f"请生成符合以下结构的JSON格式结果:\n"
                f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
                f"只需返回符合格式的JSON数据，不要有额外的解释。"
            )
            
            response_text = self.generate(json_prompt, temperature, **kwargs)
            
            # 尝试从回复中提取JSON
            try:
                json_str = self._extract_json(response_text)
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"无法解析生成的JSON: {response_text}")
                raise ValueError("模型未生成有效的JSON响应")
        
        else:  # token_classification模式
            # 直接使用NER进行处理，然后格式化为所需的JSON格式
            entities = self.ner_pipeline(prompt)
            
            # 构建符合schema结构的响应
            result = {}
            if isinstance(schema, dict) and "entities" in schema:
                result["entities"] = []
                for entity in entities:
                    result["entities"].append({
                        "entity_type": entity["entity_group"],
                        "text": entity["word"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "score": float(entity["score"])
                    })
            else:
                # 如果schema是一个列表或其他结构，直接返回实体列表
                result = []
                for entity in entities:
                    result.append({
                        "entity_type": entity["entity_group"],
                        "text": entity["word"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "score": float(entity["score"])
                    })
            
            return result
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON部分"""
        import re
        
        # 尝试查找代码块中的JSON
        json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_block_match:
            return json_block_match.group(1)
        
        # 尝试查找括号匹配的JSON对象或数组
        first_brace = text.find('{')
        if first_brace >= 0:
            stack = 0
            for i, char in enumerate(text[first_brace:]):
                if char == '{':
                    stack += 1
                elif char == '}':
                    stack -= 1
                    if stack == 0:
                        return text[first_brace:first_brace+i+1]
        
        # 尝试找方括号
        first_bracket = text.find('[')
        if first_bracket >= 0:
            stack = 0
            for i, char in enumerate(text[first_bracket:]):
                if char == '[':
                    stack += 1
                elif char == ']':
                    stack -= 1
                    if stack == 0:
                        return text[first_bracket:first_bracket+i+1]
        
        # 返回清理过的文本
        return text.strip()
    
    def fine_tune(
        self,
        training_data: Union[str, List[Dict], Dataset],
        output_dir: str,
        entity_labels: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        save_steps: int = 500,
        evaluation_data: Optional[Union[str, List[Dict], Dataset]] = None,
        **kwargs
    ) -> str:
        """
        使用自定义数据微调模型。
        
        Args:
            training_data: 训练数据 (JSON文件路径、数据列表或Huggingface Dataset)
            output_dir: 输出目录
            entity_labels: 实体标签列表 (仅token_classification模式)
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            save_steps: 保存步数
            evaluation_data: 验证数据
            **kwargs: 额外的训练参数
            
        Returns:
            str: 微调后模型的路径
        """
        if os.path.exists(output_dir) and os.listdir(output_dir):
            logger.warning(f"输出目录 {output_dir} 已存在且不为空，训练可能会覆盖现有文件")
        
        # 准备训练数据
        train_dataset = self._prepare_dataset(training_data, entity_labels)
        
        # 准备验证数据
        eval_dataset = None
        if evaluation_data:
            eval_dataset = self._prepare_dataset(evaluation_data, entity_labels)
        
        if self.mode == "token_classification":
            # 准备微调NER模型
            return self._fine_tune_token_classification(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                entity_labels=entity_labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_steps=save_steps,
                **kwargs
            )
        else:
            # 准备微调指令型模型
            return self._fine_tune_instruction(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_steps=save_steps,
                **kwargs
            )
    
    def _prepare_dataset(
        self, 
        data_source: Union[str, List[Dict], Dataset],
        entity_labels: Optional[List[str]] = None
    ) -> Dataset:
        """准备训练数据集"""
        # 如果已经是Dataset对象，直接返回
        if isinstance(data_source, Dataset):
            return data_source
        
        # 如果是文件路径，加载数据
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, 'r', encoding='utf-8') as f:
                if data_source.endswith('.json'):
                    data = json.load(f)
                else:
                    raise ValueError(f"不支持的文件格式: {data_source}")
        else:
            # 否则假设它是数据列表
            data = data_source
        
        # 根据不同模式处理数据格式
        if self.mode == "token_classification":
            # 转换为NER标签格式
            formatted_data = self._format_ner_data(data, entity_labels)
        else:
            # 转换为指令格式
            formatted_data = self._format_instruction_data(data)
        
        # 创建Dataset对象
        return Dataset.from_dict(formatted_data)
    
    def _format_ner_data(
        self, 
        data: List[Dict],
        entity_labels: Optional[List[str]] = None
    ) -> Dict[str, List]:
        """
        将数据格式化为NER训练格式。
        
        期望的输入格式:
        [{
            "text": "文本内容",
            "entities": [
                {"entity_type": "PERSON", "start": 3, "end": 5, "text": "张三"},
                ...
            ]
        }, ...]
        """
        formatted_data = {"tokens": [], "tags": []}
        
        # 更新实体标签列表
        if entity_labels:
            all_entity_types = entity_labels
            self.entity_labels = entity_labels
        elif self.entity_labels:
            all_entity_types = self.entity_labels
        else:
            # 从数据中收集所有实体类型
            all_entity_types = set()
            for example in data:
                for entity in example.get("entities", []):
                    all_entity_types.add(entity["entity_type"])
            all_entity_types = sorted(list(all_entity_types))
            self.entity_labels = all_entity_types
        
        # 更新标签到ID的映射
        self.label2id = {"O": 0}
        for i, label in enumerate(all_entity_types, 1):
            self.label2id[f"B-{label}"] = 2*i - 1
            self.label2id[f"I-{label}"] = 2*i
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 处理每个训练样本
        for example in data:
            text = example["text"]
            entities = sorted(example.get("entities", []), key=lambda x: x["start"])
            
            # 构建BIO标签
            tokens = list(text)  # 字符级分割
            tags = ["O"] * len(tokens)
            
            # 标注实体
            for entity in entities:
                start, end = entity["start"], entity["end"]
                entity_type = entity["entity_type"]
                
                if 0 <= start < end <= len(tokens):
                    tags[start] = f"B-{entity_type}"
                    for i in range(start+1, end):
                        tags[i] = f"I-{entity_type}"
            
            formatted_data["tokens"].append(tokens)
            formatted_data["tags"].append(tags)
        
        return formatted_data
    
    def _format_instruction_data(self, data: List[Dict]) -> Dict[str, List]:
        """
        将数据格式化为指令微调格式。
        """
        formatted_data = {"instruction": [], "output": []}
        
        for example in data:
            text = example["text"]
            entities = example.get("entities", [])
            
            # 构建指令
            instruction = f"请识别以下文本中的命名实体:\n\n{text}"
            
            # 构建输出 (JSON格式)
            output = json.dumps(entities, ensure_ascii=False)
            
            formatted_data["instruction"].append(instruction)
            formatted_data["output"].append(output)
        
        return formatted_data
    
    def _fine_tune_token_classification(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        output_dir: str,
        entity_labels: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        save_steps: int = 500,
        **kwargs
    ) -> str:
        """
        微调用于命名实体识别的标记分类模型。
        """
        logger.info("开始微调NER模型")
        
        # 数据预处理函数
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"], 
                is_split_into_words=True, 
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 对齐标签
            labels = []
            for i, (tokens, tags) in enumerate(zip(examples["tokens"], examples["tags"])):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                
                for word_id in word_ids:
                    if word_id is None:
                        # 特殊token设为-100
                        label_ids.append(-100)
                    else:
                        # 本来应该进一步处理子词对齐，但简化处理
                        try:
                            tag = tags[word_id]
                            label_ids.append(self.label2id.get(tag, 0))  # 未知标签映射为O
                        except IndexError:
                            label_ids.append(-100)
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        # 预处理数据集
        train_tokenized = train_dataset.map(
            tokenize_and_align_labels, 
            batched=True,
            desc="处理训练数据"
        )
        
        eval_tokenized = None
        if eval_dataset:
            eval_tokenized = eval_dataset.map(
                tokenize_and_align_labels, 
                batched=True,
                desc="处理验证数据"
            )
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            evaluation_strategy="steps" if eval_tokenized else "no",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            **kwargs
        )
        
        # 更新模型配置中的标签映射
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        
        # 设置标签数量
        if self.model.config.num_labels != len(self.id2label):
            # 需要调整分类头大小，先保存并重新加载模型
            temp_dir = tempfile.mkdtemp()
            try:
                # 保存当前模型
                self.model.save_pretrained(temp_dir)
                
                # 使用新的标签数量重新加载
                config = self.model.config
                config.num_labels = len(self.id2label)
                config.id2label = self.id2label
                config.label2id = self.label2id
                
                self.model = AutoModelForTokenClassification.from_pretrained(
                    temp_dir, 
                    config=config,
                    ignore_mismatched_sizes=True
                ).to(self.device)
            finally:
                shutil.rmtree(temp_dir)
        
        # 数据校准器
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, 
            padding=True,
            return_tensors="pt"
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final-model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"模型训练完成并保存到 {final_model_path}")
        
        # 更新当前模型为微调后的模型
        self.model_name = final_model_path
        self._setup_client()
        
        return final_model_path
    
    def _fine_tune_instruction(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        save_steps: int = 500,
        **kwargs
    ) -> str:
        """
        微调指令跟随模型用于NER任务。
        """
        logger.info("开始微调指令型模型")
        
        # 使用LoRA进行参数高效微调
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self._get_target_modules_for_model()  # 获取适合当前模型的LoRA目标模块
            )
            
            # 准备模型
            if hasattr(self.model, "is_loaded_in_8bit") or hasattr(self.model, "is_loaded_in_4bit"):
                self.model = prepare_model_for_kbit_training(self.model)
            
            # 应用LoRA
            self.model = get_peft_model(self.model, lora_config)
            logger.info("已应用LoRA适配器进行参数高效微调")
        except ImportError:
            logger.warning("未安装PEFT库，将进行全参数微调（不推荐）")
        except Exception as e:
            logger.warning(f"设置LoRA失败: {str(e)}，将进行全参数微调")
        
        # 处理指令跟随格式的数据
        def format_instruction_data(examples):
            inputs = []
            for instruction, output in zip(examples["instruction"], examples["output"]):
                # 构建提示模板
                full_prompt = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
                inputs.append(full_prompt)
                
            # 标记化
            tokenized = self.tokenizer(
                inputs,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 准备因果语言建模的标签（与输入相同）
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # 预处理数据集
        train_tokenized = train_dataset.map(
            format_instruction_data, 
            batched=True,
            desc="处理训练数据"
        )
        
        eval_tokenized = None
        if eval_dataset:
            eval_tokenized = eval_dataset.map(
                format_instruction_data, 
                batched=True,
                desc="处理验证数据"
            )
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # 积累梯度，减少内存使用
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            evaluation_strategy="steps" if eval_tokenized else "no",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            fp16=torch.cuda.is_available(),  # 如果可用则使用半精度训练
            **kwargs
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final-model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"模型训练完成并保存到 {final_model_path}")
        
        # 更新当前模型为微调后的模型
        self.model_name = final_model_path
        self._setup_client()
        
        return final_model_path
    
    def _get_target_modules_for_model(self) -> List[str]:
        """根据模型类型获取适合的LoRA目标模块"""
        # 检查模型名称中的关键词来判断模型类型
        model_name_lower = self.model_name.lower()
        
        if any(name in model_name_lower for name in ["llama", "vicuna"]):
            return ["q_proj", "v_proj"]
        elif "chatglm" in model_name_lower:
            return ["query_key_value"]
        elif any(name in model_name_lower for name in ["baichuan", "aquila"]):
            return ["W_pack"]
        elif "qwen" in model_name_lower:
            return ["c_attn"]
        elif "gemma" in model_name_lower:
            return ["q_proj", "v_proj"]
        else:
            # 默认配置，适用于许多模型
            return ["query", "value"]