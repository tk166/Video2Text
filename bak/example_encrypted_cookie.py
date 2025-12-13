#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
加密Cookie数据示例脚本
演示如何加密cookie数据并发送到远程API
"""

import requests
import json
from crypto_utils import encrypt_data

# 远程API地址
BASE_URL = "http://localhost:5001"

def create_encrypted_cookie_data(cookie_content):
    """
    创建加密的cookie数据
    
    Args:
        cookie_content (str): 原始cookie内容
        
    Returns:
        str: 加密后的cookie数据
    """
    return encrypt_data(cookie_content)

def send_process_request(video_url, encrypted_cookie_data):
    """
    发送处理请求到远程API
    
    Args:
        video_url (str): 视频URL
        encrypted_cookie_data (str): 加密的cookie数据
        
    Returns:
        str: 任务ID
    """
    payload = {
        "url": video_url,
        "encrypted_cookie_data": encrypted_cookie_data
    }
    
    response = requests.post(
        f"{BASE_URL}/api/process",
        json=payload
    )
    
    if response.status_code == 202:
        return response.json()["task_id"]
    else:
        raise Exception(f"请求失败: {response.text}")

def main():
    """主函数"""
    print("🔐 加密Cookie数据示例")
    
    # 示例cookie内容（Netscape格式）
    sample_cookie = """# Netscape HTTP Cookie File
# 示例cookie文件

.youtube.com	TRUE	/	TRUE	1768000000	SID	XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
.youtube.com	TRUE	/	TRUE	1768000000	SSID	XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
    
    # 1. 加密cookie数据
    print("🔒 正在加密cookie数据...")
    encrypted_cookie = create_encrypted_cookie_data(sample_cookie)
    print(f"✅ Cookie数据已加密，长度: {len(encrypted_cookie)} 字符")
    
    # 2. 发送处理请求
    print("\n📤 正在发送处理请求...")
    try:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 示例URL
        task_id = send_process_request(video_url, encrypted_cookie)
        print(f"✅ 请求已发送，任务ID: {task_id}")
        print(f"🔄 您可以使用以下命令查询任务状态:")
        print(f"   curl {BASE_URL}/api/status/{task_id}")
    except Exception as e:
        print(f"❌ 发送请求失败: {e}")

if __name__ == "__main__":
    main()