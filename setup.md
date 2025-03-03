# AI全息角色系统安装文档

## 环境要求

- Python 3.10（TTS库对Python版本有特定要求）
- SQLite 3.x
- 足够的存储空间用于存储视频文件和数据库
- 兼容的音频输入/输出设备

## 依赖包列表

以下是项目所需的主要依赖包：

```
# 基础依赖
numpy>=1.22.0
pyaudio>=0.2.11
pyttsx3>=2.90
librosa>=0.9.2
sqlite3>=2.6.0

# 音频处理
sounddevice>=0.4.4
webrtcvad>=2.0.10
soundfile>=0.10.3

# 向量存储与检索（可选，用于高级功能）
pymilvus>=2.1.0
faiss-cpu>=1.7.2

# 其他工具
scipy>=1.8.0
scikit-learn>=1.0.2
matplotlib>=3.5.1
```

## 安装步骤

### 1. 创建虚拟环境

```bash
# 创建Python 3.10虚拟环境
python3.10 -m venv ai_hologram_env

# 激活虚拟环境
# Linux/Mac
source ai_hologram_env/bin/activate
# Windows
ai_hologram_env\Scripts\activate
```

### 2. 克隆仓库

```bash
git clone https://github.com/yourusername/ai_hologram_character.git
cd ai_hologram_character
```

### 3. 安装依赖

创建`requirements.txt`文件，包含以下内容：

```
# 基础依赖
numpy>=1.22.0
pyaudio>=0.2.11
pyttsx3>=2.90
librosa>=0.9.2

# 音频处理
sounddevice>=0.4.4
webrtcvad>=2.0.10
soundfile>=0.10.3

# 可选依赖
scikit-learn>=1.0.2
scipy>=1.8.0
matplotlib>=3.5.1

# 系统特定依赖
# 请根据需要取消注释
# pymilvus>=2.1.0  # 向量数据库支持
# faiss-cpu>=1.7.2  # 向量相似度搜索
```

然后安装依赖：

```bash
# 安装PyAudio的系统依赖（Linux系统）
sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio

# 安装requirements.txt中的依赖
pip install -r requirements.txt
```

### 4. 系统特定安装注意事项

#### PyAudio安装

不同系统上PyAudio的安装可能需要额外步骤：

- **Windows**: 
  ```
  pip install pipwin
  pipwin install pyaudio
  ```

- **MacOS**: 
  ```
  brew install portaudio
  pip install pyaudio
  ```

- **Linux (Ubuntu/Debian)**: 
  ```
  sudo apt-get install python3-pyaudio
  ```

#### Librosa依赖

Librosa依赖于一些系统库：

```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1

# MacOS
brew install libsndfile
```

### 5. 初始化数据库

```bash
# 创建必要的目录
mkdir -p db logs videos data

# 初始化数据库
python -m db.utils.db_initializer --db_path db/i_memory.db
```

## 配置

在根目录创建`config.json`文件，包含以下内容：

```json
{
  "app": {
    "debug": false,
    "log_dir": "logs"
  },
  "database": {
    "path": "db/i_memory.db"
  },
  "audio": {
    "rate": 16000,
    "chunk_size": 1024,
    "threshold": 0.01,
    "min_silence_duration": 1.0,
    "min_speech_duration": 0.5
  },
  "video": {
    "dir": "videos"
  },
  "model": {
    "type": "local",
    "path": "models/ai_character_300M.bin"
  }
}
```

## 运行系统

```bash
# 基本运行模式
python main.py

# 使用增强对话模式（支持多用户识别）
python main.py --enhanced

# 调试模式
python main.py --debug
```

## 验证安装

以下是验证系统各组件是否正常工作的简单脚本。创建文件`verify_installation.py`：

```python
"""
验证AI全息角色系统安装
"""
import os
import sys
import importlib
import sqlite3

def verify_imports():
    """验证必要的模块导入"""
    required_modules = [
        "numpy", "pyaudio", "pyttsx3", "sqlite3", "threading", 
        "logging", "argparse", "datetime", "uuid", "re", "json"
    ]
    
    optional_modules = ["librosa", "sklearn"]
    
    print("检查必要模块...")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} 未安装或无法导入")
            return False
    
    print("\n检查可选模块...")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"? {module} 未安装（可选）")
    
    return True

def verify_audio():
    """验证音频系统"""
    print("\n检查音频系统...")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        # 获取默认输入/输出设备
        input_device_info = p.get_default_input_device_info()
        output_device_info = p.get_default_output_device_info()
        print(f"✓ 默认输入设备: {input_device_info['name']}")
        print(f"✓ 默认输出设备: {output_device_info['name']}")
        p.terminate()
        return True
    except Exception as e:
        print(f"✗ 音频系统验证失败: {e}")
        return False

def verify_database():
    """验证数据库"""
    print("\n检查数据库...")
    db_path = "db/i_memory.db"
    
    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        print(f"✗ 数据库文件不存在: {db_path}")
        return False
    
    # 检查数据库连接和表结构
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        tables = [table[0] for table in tables]
        
        required_tables = [
            "Memory", "UserProfile", "Interaction", 
            "EmotionalMark", "SentenceEmbedding"
        ]
        
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"✗ 缺少必要的表: {', '.join(missing_tables)}")
            return False
        
        print(f"✓ 数据库包含 {len(tables)} 个表")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ 数据库验证失败: {e}")
        return False

def verify_directories():
    """验证必要的目录结构"""
    print("\n检查目录结构...")
    required_dirs = ["db", "logs", "videos", "data"]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ 不存在")
            return False
    
    return True

def main():
    """主函数"""
    print("AI全息角色系统安装验证")
    print("=" * 30)
    
    # 验证各部分
    imports_ok = verify_imports()
    dirs_ok = verify_directories()
    db_ok = verify_database()
    audio_ok = verify_audio()
    
    # 总结
    print("\n验证结果摘要:")
    print(f"- 模块导入: {'通过' if imports_ok else '失败'}")
    print(f"- 目录结构: {'通过' if dirs_ok else '失败'}")
    print(f"- 数据库: {'通过' if db_ok else '失败'}")
    print(f"- 音频系统: {'通过' if audio_ok else '失败'}")
    
    if imports_ok and dirs_ok and db_ok and audio_ok:
        print("\n✓ 全部验证通过！系统准备就绪。")
        return 0
    else:
        print("\n✗ 验证失败，请解决上述问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

运行验证脚本：

```bash
python verify_installation.py
```

## 可能的问题及解决方案

### 1. PyAudio安装失败

**症状**: `pip install pyaudio` 命令出错

**解决方案**:
- 先安装PortAudio开发包，再安装PyAudio
- 对于Windows用户，尝试使用预编译的wheel: `pipwin install pyaudio`

### 2. 音频设备未检测到

**症状**: 启动时显示"找不到音频设备"或相关错误

**解决方案**:
- 检查系统音频设置
- 确保麦克风已连接并被系统识别
- 检查PyAudio版本与Python版本是否兼容

### 3. TTS引擎未正常工作

**症状**: 无法播放语音或播放错误

**解决方案**:
- 确保pyttsx3正确安装
- 对于Windows系统，可能需要安装Microsoft Speech API
- 对于Linux，检查espeak是否已安装: `sudo apt-get install espeak`

### 4. 数据库错误

**症状**: 程序启动时出现SQLite相关错误

**解决方案**:
- 确保数据库目录存在并有写权限
- 重新运行数据库初始化脚本
- 检查SQLite版本，确保与Python SQLite模块兼容

## 一键安装脚本

创建一个脚本`setup.sh`(Linux/Mac)或`setup.bat`(Windows)用于一键安装和验证系统。

### Linux/Mac (`setup.sh`):

```bash
#!/bin/bash

echo "AI全息角色系统 - 一键安装脚本"
echo "=============================="

# 检查Python 3.10
if command -v python3.10 &>/dev/null; then
    echo "✓ 已找到Python 3.10"
    PYTHON_CMD=python3.10
else
    echo "✗ 未找到Python 3.10，尝试使用python3..."
    PYTHON_CMD=python3
fi

# 创建虚拟环境
echo "正在创建虚拟环境..."
$PYTHON_CMD -m venv ai_hologram_env
source ai_hologram_env/bin/activate

# 创建目录
echo "创建必要目录..."
mkdir -p db logs videos data

# 创建requirements.txt
echo "创建依赖文件..."
cat > requirements.txt << EOL
numpy>=1.22.0
pyaudio>=0.2.11
TTS>=0.17.0
librosa>=0.9.2
sounddevice>=0.4.4
webrtcvad>=2.0.10
soundfile>=0.10.3
scikit-learn>=1.0.2
scipy>=1.8.0
matplotlib>=3.5.1
EOL

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 初始化数据库
echo "初始化数据库..."
python -m db.utils.db_initializer --db_path db/i_memory.db

# 创建验证脚本
echo "创建验证脚本..."
cat > verify_installation.py << 'EOL'
"""
验证AI全息角色系统安装
"""
import os
import sys
import importlib
import sqlite3

def verify_imports():
    """验证必要的模块导入"""
    required_modules = [
        "numpy", "pyaudio", "pyttsx3", "sqlite3", "threading", 
        "logging", "argparse", "datetime", "uuid", "re", "json"
    ]
    
    optional_modules = ["librosa", "sklearn"]
    
    print("检查必要模块...")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} 未安装或无法导入")
            return False
    
    print("\n检查可选模块...")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"? {module} 未安装（可选）")
    
    return True

def verify_audio():
    """验证音频系统"""
    print("\n检查音频系统...")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        # 获取默认输入/输出设备
        input_device_info = p.get_default_input_device_info()
        output_device_info = p.get_default_output_device_info()
        print(f"✓ 默认输入设备: {input_device_info['name']}")
        print(f"✓ 默认输出设备: {output_device_info['name']}")
        p.terminate()
        return True
    except Exception as e:
        print(f"✗ 音频系统验证失败: {e}")
        return False

def verify_database():
    """验证数据库"""
    print("\n检查数据库...")
    db_path = "db/i_memory.db"
    
    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        print(f"✗ 数据库文件不存在: {db_path}")
        return False
    
    # 检查数据库连接和表结构
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        tables = [table[0] for table in tables]
        
        required_tables = [
            "Memory", "UserProfile", "Interaction", 
            "EmotionalMark", "SentenceEmbedding"
        ]
        
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"✗ 缺少必要的表: {', '.join(missing_tables)}")
            return False
        
        print(f"✓ 数据库包含 {len(tables)} 个表")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ 数据库验证失败: {e}")
        return False

def verify_directories():
    """验证必要的目录结构"""
    print("\n检查目录结构...")
    required_dirs = ["db", "logs", "videos", "data"]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ 不存在")
            return False
    
    return True

def main():
    """主函数"""
    print("AI全息角色系统安装验证")
    print("=" * 30)
    
    # 验证各部分
    imports_ok = verify_imports()
    dirs_ok = verify_directories()
    db_ok = verify_database()
    audio_ok = verify_audio()
    
    # 总结
    print("\n验证结果摘要:")
    print(f"- 模块导入: {'通过' if imports_ok else '失败'}")
    print(f"- 目录结构: {'通过' if dirs_ok else '失败'}")
    print(f"- 数据库: {'通过' if db_ok else '失败'}")
    print(f"- 音频系统: {'通过' if audio_ok else '失败'}")
    
    if imports_ok and dirs_ok and db_ok and audio_ok:
        print("\n✓ 全部验证通过！系统准备就绪。")
        return 0
    else:
        print("\n✗ 验证失败，请解决上述问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOL

# 创建简单配置文件
echo "创建配置文件..."
cat > config.json << EOL
{
  "app": {
    "debug": false,
    "log_dir": "logs"
  },
  "database": {
    "path": "db/i_memory.db"
  },
  "audio": {
    "rate": 16000,
    "chunk_size": 1024,
    "threshold": 0.01,
    "min_silence_duration": 1.0,
    "min_speech_duration": 0.5
  },
  "video": {
    "dir": "videos"
  },
  "model": {
    "type": "local",
    "path": "models/ai_character_300M.bin"
  }
}
EOL

# 验证安装
echo "验证安装..."
python verify_installation.py

echo "安装完成！"
echo "使用以下命令运行系统:"
echo "  python main.py"
```

### Windows (`setup.bat`):

```batch
@echo off
echo AI全息角色系统 - 一键安装脚本
echo ==============================

:: 检查Python 3.10
python --version 2>&1 | findstr /r "3\.10\." > nul
if %errorlevel% equ 0 (
    echo ✓ 已找到Python 3.10
) else (
    echo ✗ 未找到Python 3.10，请安装Python 3.10后重试
    exit /b 1
)

:: 创建虚拟环境
echo 正在创建虚拟环境...
python -m venv ai_hologram_env
call ai_hologram_env\Scripts\activate

:: 创建目录
echo 创建必要目录...
mkdir db logs videos data 2>nul

:: 创建requirements.txt
echo 创建依赖文件...
(
echo numpy>=1.22.0
echo pyaudio>=0.2.11
echo pyttsx3>=2.90
echo librosa>=0.9.2
echo sounddevice>=0.4.4
echo webrtcvad>=2.0.10
echo soundfile>=0.10.3
echo scikit-learn>=1.0.2
echo scipy>=1.8.0
echo matplotlib>=3.5.1
) > requirements.txt

:: 安装依赖
echo 安装依赖...
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt

:: 初始化数据库
echo 初始化数据库...
python -m db.utils.db_initializer --db_path db/i_memory.db

:: 创建验证脚本 (略，内容与Linux版相同)

:: 创建简单配置文件
echo 创建配置文件...
(
echo {
echo   "app": {
echo     "debug": false,
echo     "log_dir": "logs"
echo   },
echo   "database": {
echo     "path": "db/i_memory.db"
echo   },
echo   "audio": {
echo     "rate": 16000,
echo     "chunk_size": 1024,
echo     "threshold": 0.01,
echo     "min_silence_duration": 1.0,
echo     "min_speech_duration": 0.5
echo   },
echo   "video": {
echo     "dir": "videos"
echo   },
echo   "model": {
echo     "type": "local",
echo     "path": "models/ai_character_300M.bin"
echo   }
echo }
) > config.json

:: 验证安装
echo 验证安装...
python verify_installation.py

echo 安装完成！
echo 使用以下命令运行系统:
echo   python main.py
```

## 使用说明

安装完成后，系统的基本使用方法如下：

1. **启动系统**:
   ```bash
   python main.py
   ```

2. **系统将自动**:
   - 初始化音频服务
   - 等待用户语音输入
   - 处理语音并产生回应
   - 选择合适的视频在全息三棱锥上显示

3. **结束对话**:
   按下 `Ctrl+C` 可以安全结束程序。

## 高级配置

对于需要更多控制的用户，可以通过编辑`config.json`文件进行高级配置：

- 调整音频参数
- 配置视频目录
- 设置日志级别
- 配置数据库路径

## 故障排除

如果系统运行过程中遇到问题，请尝试：

1. 查看日志文件（在`logs`目录下）
2. 运行验证脚本确认依赖是否完整
3. 使用`--debug`参数启动以获取更详细的输出

如果问题仍然存在，请提交问题报告，包括错误信息和日志文件。