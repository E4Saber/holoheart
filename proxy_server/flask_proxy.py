from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的路径
parent_dir = os.path.dirname(current_dir)
# 构建 .env 文件的完整路径
dotenv_path = os.path.join(parent_dir, '.env')

# 加载环境变量
load_dotenv(dotenv_path, override=True)

app = Flask(__name__)
# 启用CORS，允许所有来源
# 使用flask_cors的更高级配置允许所有请求头
# CORS(app, resources={r"/v1/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*", "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"]}})
CORS(app)

# Kimi API 配置
KIMI_API_URL = os.getenv('KIMI_API_URL', 'https://api.moonshot.cn/v1')
KIMI_API_KEY = os.getenv('KIMI_API_KEY', '')

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'OPTIONS', 'PUT', 'DELETE'])
def proxy(path):

    # 对于 OPTIONS 请求，直接返回 CORS 头
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
        return response

    # 构建目标URL
    url = f"{KIMI_API_URL}/{path}"

    print(f"Proxying request to {url}")
    
    # 获取原始请求的所有内容
    headers = {
        key: value for key, value in request.headers.items()
        if key.lower() not in ['host', 'content-length']
    }
    
    # 添加API密钥到请求头
    headers['Authorization'] = f"Bearer {KIMI_API_KEY}"
    
    # 转发请求
    try:
        if request.method == 'GET':
            resp = requests.get(url, headers=headers, params=request.args)
        elif request.method == 'POST':
            resp = requests.post(url, headers=headers, json=request.get_json())
        elif request.method == 'PUT':
            resp = requests.put(url, headers=headers, json=request.get_json())
        elif request.method == 'DELETE':
            resp = requests.delete(url, headers=headers)
        else:
            return jsonify({"error": "Method not allowed"}), 405
        
        # 构建响应
        response = Response(
            resp.content,
            resp.status_code,
            content_type=resp.headers.get('Content-Type')
        )
        
        
        return response
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)