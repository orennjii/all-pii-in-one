#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理 Gradio UI 模块

提供与 test_text_processor.py 相同功能的 Web 界面
"""

import gradio as gr
from gradio.themes import Soft
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from src.configs import AppConfig
from src.processors.text_processor import TextProcessor
from src.commons import get_module_logger, find_project_root

logger = get_module_logger(__name__)


class TextProcessorTab:
    """文本处理器 Gradio Tab"""
    
    def __init__(self):
        """初始化文本处理器 Tab"""
        self.text_processor = None
        self.project_root = find_project_root(Path(__file__))
        self._load_config()
        
    def _load_config(self):
        """加载配置并初始化文本处理器"""
        try:
            config_path = self.project_root / "config" / "app_config.yaml"
            app_config = AppConfig.load_from_yaml(str(config_path))
            text_processor_config = app_config.processor.text_processor
            self.text_processor = TextProcessor(config=text_processor_config)
            logger.info("文本处理器初始化成功")
        except Exception as e:
            logger.error(f"文本处理器初始化失败: {e}")
            
    def process_text(
        self,
        text: str,
        enable_segmentation: bool,
        enable_analysis: bool,
        enable_anonymization: bool,
        language: str,
        entities_filter: str,
        score_threshold: float
    ) -> Tuple[str, str, str, str]:
        """
        处理文本并返回结果
        
        Args:
            text: 输入文本
            enable_segmentation: 是否启用分割
            enable_analysis: 是否启用分析
            enable_anonymization: 是否启用匿名化
            language: 语言代码
            entities_filter: 实体过滤器（逗号分隔）
            score_threshold: 置信度阈值
            
        Returns:
            Tuple[原始文本, 匿名化文本, 分析结果, 处理统计]
        """
        if not self.text_processor:
            return text, "错误：文本处理器未初始化", "错误：文本处理器未初始化", "错误：文本处理器未初始化"
            
        if not text.strip():
            return "", "", "请输入要处理的文本", ""
            
        try:
            # 处理实体过滤器
            entities = None
            if entities_filter.strip():
                entities = [e.strip() for e in entities_filter.split(',') if e.strip()]
            
            # 处理语言参数
            lang = language if language != "auto" else None
            
            # 执行处理
            processing_result = self.text_processor.process(
                text=text,
                enable_segmentation=enable_segmentation,
                enable_analysis=enable_analysis,
                enable_anonymization=enable_anonymization,
                language=lang,
                entities=entities,
                score_threshold=score_threshold
            )
            
            # 格式化结果
            original_text = processing_result.original_text
            anonymized_text = processing_result.anonymized_text
            
            # 构建分析结果
            analysis_results = self._format_analysis_results(processing_result)
            
            # 构建统计信息
            statistics = self._format_statistics(processing_result)
            
            return original_text, anonymized_text, analysis_results, statistics
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logger.error(error_msg)
            return text, error_msg, error_msg, error_msg
    
    def _format_analysis_results(self, processing_result) -> str:
        """格式化分析结果"""
        if not processing_result.analysis_results:
            return "未检测到任何 PII 实体"
        
        result_lines = ["=== PII 检测结果 ===\n"]
        
        total_entities = 0
        for i, segment_results in enumerate(processing_result.analysis_results):
            if segment_results:
                total_entities += len(segment_results)
                segment_text = (processing_result.segments[i].text 
                              if processing_result.segments 
                              else processing_result.original_text)
                
                result_lines.append(f"段落 {i+1} 中检测到 {len(segment_results)} 个PII实体:")
                
                for result in segment_results:
                    entity_value = segment_text[result.start:result.end]
                    result_lines.append(
                        f"  • 类型: {result.entity_type}"
                        f"  | 值: {entity_value}"
                        f"  | 位置: {result.start}-{result.end}"
                        f"  | 置信度: {result.score:.2f}"
                    )
                result_lines.append("")
        
        if total_entities == 0:
            return "未检测到任何 PII 实体"
        
        return "\n".join(result_lines)
    
    def _format_statistics(self, processing_result) -> str:
        """格式化统计信息"""
        total_entities = sum(len(results) for results in processing_result.analysis_results)
        segments_count = len(processing_result.segments) if processing_result.segments else 1
        
        # 按类型统计
        entity_types = {}
        for segment_results in processing_result.analysis_results:
            for result in segment_results:
                entity_types[result.entity_type] = entity_types.get(result.entity_type, 0) + 1
        
        stats_lines = [
            "=== 处理统计 ===",
            f"原始文本长度: {len(processing_result.original_text)} 字符",
            f"匿名化后长度: {len(processing_result.anonymized_text)} 字符",
            f"文本段落数: {segments_count}",
            f"检测到的PII实体总数: {total_entities}",
            ""
        ]
        
        if entity_types:
            stats_lines.append("按类型统计:")
            for entity_type, count in sorted(entity_types.items()):
                stats_lines.append(f"  • {entity_type}: {count} 个")
        
        return "\n".join(stats_lines)
    
    def get_example_text(self) -> str:
        """获取示例文本"""
        return """尊敬的客服团队：

你们好，我叫李明轩，想咨询一下我的订单 #TS20250527A88G 的最新发货状态。这个订单是我上周二下午通过你们的手机应用下的，当时预估3天内发货，但现在还没有收到任何更新。

我的注册手机号是 138-1088-6688，注册邮箱是 mingxuan.li.test@emailservice.cn。如果需要核实身份，我的会员卡号是 VIP9876543210。

麻烦你们帮忙查一下，看看包裹具体到哪里了。如果已经发出了，希望能提供一下快递单号。我的收货地址是：北京市海淀区中关村南大街28号院3号楼B座1101室，邮编100081。

另外，我记得当时购买这款"智能空气净化器Pro"时，客服代表王小姐（工号大概是A073）提到，如果我是1990年6月15日之前出生的，可以享受一个额外的老客户折扣。我的出生日期是1988年10月26日，不知道这个折扣是否已经应用到订单里了？如果方便的话，也请一并核实。

非常感谢！期待你们的回复。

祝好"""
    
    def clear_all(self) -> Tuple[str, str, str, str, str]:
        """清空所有内容"""
        return "", "", "", "", ""
    
    def create_interface(self) -> gr.Blocks:
        """创建 Gradio 界面"""
        with gr.Blocks(title="文本PII处理", theme=Soft()) as interface:
            gr.Markdown("# 📝 文本隐私信息检测与匿名化工具")
            gr.Markdown("基于 TextProcessor 的文本隐私信息处理工具，支持多种 PII 实体的检测和匿名化。")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    gr.Markdown("## 📥 输入文本")
                    input_text = gr.Textbox(
                        label="请输入要处理的文本",
                        placeholder="在此输入包含个人隐私信息的文本...",
                        lines=10,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        example_btn = gr.Button("📋 使用示例文本", variant="secondary")
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                    
                with gr.Column(scale=1):
                    # 配置区域
                    gr.Markdown("## ⚙️ 处理配置")
                    
                    enable_segmentation = gr.Checkbox(
                        label="启用文本分割",
                        value=False,
                        info="将长文本分割为更小的处理单元"
                    )
                    
                    enable_analysis = gr.Checkbox(
                        label="启用PII分析",
                        value=True,
                        info="检测文本中的个人隐私信息"
                    )
                    
                    enable_anonymization = gr.Checkbox(
                        label="启用匿名化",
                        value=True,
                        info="对检测到的PII进行匿名化处理"
                    )
                    
                    language = gr.Dropdown(
                        label="处理语言",
                        choices=["auto", "zh", "en"],
                        value="zh",
                        info="指定文本语言，auto表示自动检测"
                    )
                    
                    entities_filter = gr.Textbox(
                        label="实体类型过滤",
                        placeholder="PERSON,PHONE_NUMBER,EMAIL_ADDRESS",
                        info="指定要检测的实体类型（逗号分隔），留空表示检测所有支持的类型"
                    )
                    
                    score_threshold = gr.Slider(
                        label="置信度阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        info="只显示置信度高于此阈值的检测结果"
                    )
                    
                    process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
            
            # 输出区域
            gr.Markdown("## 📤 处理结果")
            
            with gr.Tabs():
                with gr.TabItem("匿名化结果"):
                    original_output = gr.Textbox(
                        label="原始文本",
                        lines=8,
                        max_lines=15,
                        interactive=False
                    )
                    
                    anonymized_output = gr.Textbox(
                        label="匿名化后文本",
                        lines=8,
                        max_lines=15,
                        interactive=False
                    )
                
                with gr.TabItem("分析详情"):
                    analysis_output = gr.Textbox(
                        label="PII检测详情",
                        lines=12,
                        max_lines=20,
                        interactive=False
                    )
                
                with gr.TabItem("统计信息"):
                    statistics_output = gr.Textbox(
                        label="处理统计",
                        lines=8,
                        max_lines=15,
                        interactive=False
                    )
            
            # 事件绑定
            process_btn.click(
                fn=self.process_text,
                inputs=[
                    input_text,
                    enable_segmentation,
                    enable_analysis,
                    enable_anonymization,
                    language,
                    entities_filter,
                    score_threshold
                ],
                outputs=[
                    original_output,
                    anonymized_output,
                    analysis_output,
                    statistics_output
                ]
            )
            
            example_btn.click(
                fn=lambda: self.get_example_text(),
                outputs=input_text
            )
            
            clear_btn.click(
                fn=self.clear_all,
                outputs=[
                    input_text,
                    original_output,
                    anonymized_output,
                    analysis_output,
                    statistics_output
                ]
            )
            
            # 添加说明信息
            with gr.Accordion("📖 使用说明", open=False):
                gr.Markdown("""
                ### 支持的PII实体类型
                - **PERSON**: 人名
                - **PHONE_NUMBER**: 电话号码
                - **EMAIL_ADDRESS**: 电子邮箱
                - **ID_CARD**: 身份证号码
                - **BANK_ACCOUNT**: 银行账号
                - **CREDIT_CARD**: 信用卡号
                - **LOCATION**: 地理位置
                - **ORGANIZATION**: 组织机构
                - **DATE_TIME**: 日期时间
                - **IP_ADDRESS**: IP地址
                - **URL**: 网址链接
                - **AGE**: 年龄
                - **CURRENCY**: 货币金额
                
                ### 使用步骤
                1. 在输入框中输入要处理的文本（可点击"使用示例文本"加载示例）
                2. 根据需要调整处理配置参数
                3. 点击"开始处理"按钮执行处理
                4. 在结果标签页中查看匿名化结果、分析详情和统计信息
                
                ### 配置说明
                - **文本分割**: 对于长文本，启用分割可以提高处理效率
                - **PII分析**: 检测文本中的个人隐私信息
                - **匿名化**: 对检测到的PII进行星号替换等匿名化处理
                - **实体过滤**: 可以指定只检测特定类型的PII实体
                - **置信度阈值**: 调整检测的敏感度，值越高越严格
                """)
        
        return interface


def create_text_tab() -> gr.Blocks:
    """创建文本处理标签页"""
    tab = TextProcessorTab()
    return tab.create_interface()


if __name__ == "__main__":
    # 独立运行测试
    demo = create_text_tab()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )
