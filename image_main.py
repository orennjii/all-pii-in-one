# from src.core.llm_recognizer.llm_clients.gemini_client import GeminiClient
# import os
# # os.environ["HTTP_PROXY"] = "http://127.0.0.1:17072"
# # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:17072"


# gemini_client = GeminiClient(
#     api_key="",
#     model_name='models/gemini-2.0-flash-thinking-exp-01-21',
# )

# response = gemini_client.generate(
#     """
#     请帮我找出以下文本中的所有命名实体,生成一个JSON对象,包含实体的类型和文本和打分.
#     例如: [{"entity_type": 'DATE_TIME', "text": "2023年10月1日", "score": 0.95},
#     {"entity_type": 'PERSON', "text": "张三", "score": 0.92}]

#     文本: "
#     这是一份包含敏感信息的示例文本：

#     我的姓名是张三，来自北京。
#     我的车牌号是京A12345，手机号码是13812345678。
#     我的身份证号是210102199203183096，银行卡号是4929717705917895。

#     2023年10月15日，我访问了www.example.com并发送邮件到test@example.com，IP地址是192.168.1.1。
#     "
#     """
# )
# print(response)

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 指定模型本地路径
# model_path = "d:/code/LLMs/Qwen2.5-VL-7B-Instruct"

# # 加载tokenizer和模型
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",  # 自动选择GPU或CPU
#     trust_remote_code=True
# ).eval()

# query = "给我解释一下量子计算的基本原理"
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

from PIL import Image
from presidio_image_redactor import ImageRedactorEngine

# Get the image to redact using PIL lib (pillow)
print("正在加载图片...")
image = Image.open("./image.png")

# Initialize the engine
engine = ImageRedactorEngine()

print("正在进行图片处理...")
# Redact the image with pink color
redacted_image = engine.redact(image, (255, 192, 203), ocr_kwargs={"lang": "chi_sim"})

# save the redacted image 
redacted_image.save("redacted_image.png")
print("图片已保存为 redacted_image.png")