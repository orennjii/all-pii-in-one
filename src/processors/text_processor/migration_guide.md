# text_processor 目录重构迁移指南

## 1. 项目结构变更

我们按照Python项目结构指南和Google Python Style Guide规范，对`text_processor`目录进行了重构。主要变更包括：

### 原有结构

```
text_processor/
├── __init__.py
├── pattern_recognizer/
│   ├── __init__.py
│   ├── pattern_recognizer.py
│   ├── prc_cc_recognizer.py
│   ├── prc_cpn_recognizer.py
│   ├── prc_idc_recognizer.py
│   ├── prc_pn_recognizer.py
│   └── prc_utl_recognizer.py
└── llm_recognizer/
    ├── __init__.py
    ├── llm_recognizer.py
    ├── config/
    ├── llm_clients/
    ├── parsers/
    └── prompts/
```

### 新结构

```
text_processor_new/
├── __init__.py
├── registry.py
├── utils.py
├── main_example.py
├── test_structure.py
└── recognizers/
    ├── __init__.py
    ├── base.py
    ├── pattern/
    │   ├── __init__.py
    │   ├── bank_card.py
    │   ├── car_plate.py
    │   ├── id_card.py
    │   ├── phone_number.py
    │   └── url.py
    └── llm/
        ├── __init__.py
        ├── recognizer.py
        ├── clients/
        ├── config/
        ├── parsers/
        └── prompts/
```

## 2. 文件名与类名变更

为了遵循Python命名规范，我们对文件名和类名进行了调整：

| 原文件名 | 新文件名 | 原类名 | 新类名 |
|---------|---------|-------|-------|
| prc_cc_recognizer.py | bank_card.py | ChineseBankCardRecognizer | BankCardRecognizer |
| prc_cpn_recognizer.py | car_plate.py | ChineseCarPlateNumberRecognizer | CarPlateNumberRecognizer |
| prc_idc_recognizer.py | id_card.py | ChineseIDCardRecognizer | IDCardRecognizer |
| prc_pn_recognizer.py | phone_number.py | ChineseMobileNumberRecognizer | PhoneNumberRecognizer |
| prc_utl_recognizer.py | url.py | WebsiteURLRecognizer | URLRecognizer |

## 3. 功能变更

* `pattern_recognizer/__init__.py`中的函数移到了`registry.py`
* 删除了空的`pattern_recognizer.py`文件
* 在`recognizers/base.py`中添加了通用功能
* 在`utils.py`中添加了工具函数

## 4. 导入路径变更

原导入:
```python
from src.processors.text_processor.pattern_recognizer import register_chinese_pattern_recognizers, get_all_chinese_pattern_recognizers
```

新导入:
```python
from src.processors.text_processor_new.registry import register_chinese_pattern_recognizers, get_all_chinese_pattern_recognizers
```

模式识别器导入:
```python
from src.processors.text_processor_new.recognizers.pattern import (
    BankCardRecognizer, 
    CarPlateNumberRecognizer, 
    IDCardRecognizer,
    PhoneNumberRecognizer,
    URLRecognizer
)
```

LLM识别器导入:
```python
from src.processors.text_processor_new.recognizers.llm import LLMRecognizer
```

## 5. 迁移步骤

1. 将`text_processor_new`目录复制到项目中
2. 修改导入路径，替换为新路径
3. 运行`test_structure.py`确保一切正常
4. 在新结构稳定后，可以删除旧的`text_processor`目录

## 6. 兼容性注意事项

* 类名变化可能导致需要修改调用代码
* 所有功能保持不变，只是组织方式发生了变化
* 添加了更完善的类型注解和文档字符串，有助于IDE自动补全和代码提示

## 7. 改进与优化

* 遵循了Google Python Style Guide，提高代码可读性
* 改进了模块组织和层次结构，更易维护
* 添加了更详细的文档字符串和类型注解
* 优化了命名约定，使用更具描述性的名称
* 增加了工具函数在`utils.py`中，便于复用
