# holoheart
Where memories have feelings, and companionship becomes real.

# Project Structure
```
    ai_hologram_character/
    ├── api/                      # API层
    │   ├── core/                 # 核心功能模块
    │   │   ├── __init__.py
    │   │   ├── dialogue_processor.py  # 对话处理主流程
    │   │   └── memory_manager.py      # 记忆管理
    │   ├── services/             # 服务层
    │   │   ├── __init__.py
    │   │   ├── audio_service.py  # 语音处理服务
    │   │   ├── emotion_service.py # 情感处理服务
    │   │   └── video_service.py  # 视频处理服务
    │   ├── data_access/          # 数据访问层
    │   │   ├── __init__.py
    │   │   ├── database_manager.py  # 数据库管理
    │   │   ├── memory_repository.py # 记忆仓库
    │   │   ├── user_repository.py   # 用户仓库 
    │   │   └── vector_repository.py # 向量仓库
    │   └── models/               # 数据模型
    │       ├── __init__.py
    │       ├── memory_model.py  
    │       ├── user_model.py
    │       └── dialogue_model.py
    ├── data/                     # 数据定义和配置
    │   ├── __init__.py
    │   ├── schema/               # 数据架构
    │   │   ├── memory_schema.py
    │   │   └── dialogue_schema.py
    │   ├── prompt_templates/     # 提示词模板
    │   │   ├── analysis_template.py
    │   │   └── response_template.py
    │   └── config/               # 配置文件
    │       ├── db_config.py
    │       └── model_config.py
    ├── db/                       # 数据库脚本和工具
    │   ├── migrations/           # 数据库迁移脚本
    │   │   └── initial_schema.sql
    │   └── utils/                # 数据库工具
    │       └── db_initializer.py
    ├── utils/                    # 通用工具类
    │   ├── __init__.py
    │   ├── error_handler.py
    │   └── logger.py
    └── main.py                   # 应用入口
```