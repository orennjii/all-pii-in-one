# 部署指南

本文档详细介绍了如何在不同环境中部署 ALL PII IN ONE 系统。

## 📋 目录

- [环境要求](#环境要求)
- [本地部署](#本地部署)
- [Docker部署](#docker部署)
- [云端部署](#云端部署)
- [生产环境配置](#生产环境配置)
- [监控和日志](#监控和日志)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

## 🔧 环境要求

### 最低配置

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4核心 | 8核心+ |
| 内存 | 8GB | 16GB+ |
| 存储 | 50GB | 100GB+ SSD |
| Python | 3.8+ | 3.10+ |
| CUDA | 11.8+ (可选) | 12.0+ |

### 操作系统支持

- ✅ Ubuntu 20.04+
- ✅ CentOS 8+
- ✅ macOS 12.0+
- ✅ Windows 10+
- ✅ Docker容器

## 🏠 本地部署

### 1. 快速启动

```bash
# 克隆项目
git clone https://github.com/your-username/all-pii-in-one.git
cd all-pii-in-one

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动应用
python src/app/main.py
```

### 2. 开发环境配置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit钩子
pre-commit install

# 运行测试
pytest test/

# 代码格式化
black src/
isort src/
```

### 3. 环境变量配置

创建 `.env` 文件：

```bash
# API密钥
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# 应用配置
APP_ENV=development
LOG_LEVEL=DEBUG
TEMP_DIR=/tmp/pii_app

# 设备配置
DEVICE=cuda  # 或 cpu
GPU_ID=0

# UI配置
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false
```

## 🐳 Docker部署

### 1. 基础Docker部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "src/app/main.py"]
```

构建和运行：

```bash
# 构建镜像
docker build -t all-pii-in-one .

# 运行容器
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e GEMINI_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  all-pii-in-one
```

### 2. Docker Compose部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DEVICE=cpu
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

启动服务：

```bash
docker-compose up -d
```

### 3. GPU支持的Docker部署

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 安装Python和系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置Python别名
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# 复制依赖文件
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "src/app/main.py"]
```

运行GPU容器：

```bash
docker run --gpus all -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e GEMINI_API_KEY=your_key \
  -e DEVICE=cuda \
  all-pii-in-one:gpu
```

## ☁️ 云端部署

### 1. AWS部署

#### EC2部署

```bash
# 创建EC2实例
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --user-data file://user-data.sh

# user-data.sh 脚本
#!/bin/bash
yum update -y
yum install -y docker git
systemctl start docker
systemctl enable docker

# 安装Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 克隆项目并启动
git clone https://github.com/your-username/all-pii-in-one.git
cd all-pii-in-one
docker-compose up -d
```

#### ECS部署

```json
{
  "family": "all-pii-in-one",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/all-pii-in-one:latest",
      "portMappings": [
        {
          "containerPort": 7860,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "HF_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:hf-token"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/all-pii-in-one",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### 2. Google Cloud Platform

```yaml
# app.yaml for App Engine
runtime: python310

env_variables:
  HF_TOKEN: "your_token"
  GEMINI_API_KEY: "your_key"

resources:
  cpu: 2
  memory_gb: 4
  disk_size_gb: 20

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.7
```

### 3. Azure部署

```yaml
# azure-container-instances.yml
apiVersion: 2019-12-01
location: eastus
name: all-pii-in-one
properties:
  containers:
  - name: app
    properties:
      image: your-registry.azurecr.io/all-pii-in-one:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 7860
      environmentVariables:
      - name: HF_TOKEN
        secureValue: your_token
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 7860
  restartPolicy: Always
```

## 🔧 生产环境配置

### 1. 反向代理配置

#### Nginx配置

```nginx
# nginx.conf
upstream app {
    server 127.0.0.1:7860;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;

    # SSL配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # 上传文件大小限制
    client_max_body_size 100M;

    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # 静态文件缓存
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### 2. 系统服务配置

```ini
# /etc/systemd/system/all-pii-in-one.service
[Unit]
Description=ALL PII IN ONE Service
After=network.target

[Service]
Type=simple
User=app
Group=app
WorkingDirectory=/opt/all-pii-in-one
Environment=PATH=/opt/all-pii-in-one/venv/bin
ExecStart=/opt/all-pii-in-one/venv/bin/python src/app/main.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

# 环境变量
Environment=HF_TOKEN=your_token
Environment=GEMINI_API_KEY=your_key
Environment=APP_ENV=production

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable all-pii-in-one
sudo systemctl start all-pii-in-one
sudo systemctl status all-pii-in-one
```

## 📊 监控和日志

### 1. 日志配置

```yaml
# logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/error.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file, error_file]
```

### 2. 健康检查

```python
# health_check.py
from flask import Flask, jsonify
import psutil
import torch

app = Flask(__name__)

@app.route('/health')
def health_check():
    """系统健康检查"""
    try:
        # 检查CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 检查内存使用
        memory = psutil.virtual_memory()
        
        # 检查GPU状态（如果有）
        gpu_available = torch.cuda.is_available()
        gpu_memory = None
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
        status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': psutil.disk_usage('/').percent
            },
            'gpu': {
                'available': gpu_available,
                'memory_total': gpu_memory
            }
        }
        
        # 检查关键阈值
        if cpu_percent > 90 or memory.percent > 90:
            status['status'] = 'warning'
            
        return jsonify(status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 3. Prometheus监控

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'all-pii-in-one'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## ⚡ 性能优化

### 1. 应用层优化

```python
# 配置优化
production_config = {
    'general': {
        'device': 'cuda',
        'gpu_id': 0,
        'log_level': 'WARNING'
    },
    'processor': {
        'audio_processor': {
            'transcription': {
                'batch_size': 8,
                'compute_type': 'float16'
            }
        },
        'text_processor': {
            'analyzer': {
                'parallel_processing': True,
                'max_workers': 8,
                'enable_cache': True,
                'cache_size': 10000
            }
        }
    }
}
```

### 2. 缓存策略

```python
# Redis缓存配置
CACHES = {
    'default': {
        'BACKEND': 'redis_cache.RedisCache',
        'LOCATION': 'redis://localhost:6379/1',
        'OPTIONS': {
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,
                'retry_on_timeout': True,
            }
        },
        'TIMEOUT': 3600,  # 1小时
        'KEY_PREFIX': 'pii_cache',
        'VERSION': 1,
    }
}
```

### 3. 数据库优化

```python
# PostgreSQL配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'pii_db',
    'user': 'pii_user',
    'password': 'secure_password',
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

## 🔍 故障排除

### 常见问题

#### 1. 内存不足

```bash
# 检查内存使用
free -h
ps aux --sort=-%mem | head -10

# 解决方案
# 1. 减少batch_size
# 2. 使用CPU而非GPU
# 3. 增加swap空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 2. GPU内存不足

```python
# 清理GPU内存
import torch
torch.cuda.empty_cache()

# 监控GPU使用
nvidia-smi -l 1
```

#### 3. 模型下载失败

```bash
# 手动下载模型
export HF_HUB_CACHE=/path/to/cache
huggingface-cli download model_name --cache-dir /path/to/cache

# 设置代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=https://proxy:port
```

#### 4. 端口冲突

```bash
# 检查端口占用
netstat -tulpn | grep :7860
lsof -i :7860

# 修改端口
export GRADIO_SERVER_PORT=8080
```

### 日志分析

```bash
# 查看错误日志
tail -f logs/error.log

# 搜索特定错误
grep -n "ERROR" logs/app.log | tail -20

# 分析性能
grep "processing time" logs/app.log | awk '{print $NF}' | sort -n
```

### 性能监控

```bash
# 系统资源监控
htop
iostat -x 1
iotop

# 应用性能分析
python -m cProfile -o profile.stats src/app/main.py
python -m pstats profile.stats
```

## 📋 部署清单

### 部署前检查

- [ ] 系统资源充足
- [ ] 依赖包已安装
- [ ] API密钥已配置
- [ ] 防火墙规则正确
- [ ] SSL证书有效
- [ ] 数据库连接正常
- [ ] 缓存服务可用

### 部署步骤

1. **环境准备**
   - [ ] 创建用户账户
   - [ ] 设置目录权限
   - [ ] 配置环境变量

2. **应用部署**
   - [ ] 代码部署
   - [ ] 依赖安装
   - [ ] 配置文件更新
   - [ ] 数据库迁移

3. **服务配置**
   - [ ] 系统服务配置
   - [ ] 反向代理配置
   - [ ] 监控配置
   - [ ] 日志轮转配置

4. **测试验证**
   - [ ] 功能测试
   - [ ] 性能测试
   - [ ] 安全测试
   - [ ] 监控验证

5. **上线部署**
   - [ ] 启动服务
   - [ ] 健康检查
   - [ ] 流量切换
   - [ ] 监控告警

---

如有部署问题，请参考 [故障排除文档](TROUBLESHOOTING.md) 或提交 [Issue](https://github.com/your-username/all-pii-in-one/issues)。
