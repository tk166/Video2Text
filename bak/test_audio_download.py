#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试音频文件功能的脚本
"""

import requests
import time
import os

# 测试服务器地址
BASE_URL = "http://localhost:5001"

def test_audio_feature():
    """测试音频文件功能"""
    print("🎵 测试音频文件功能...")
    
    try:
        # 发送处理请求，要求保留音频
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 示例链接
        payload = {
            "url": test_url,
            "keep_audio": True  # 要求保留音频文件
        }
        
        response = requests.post(f"{BASE_URL}/api/process", json=payload)
        
        if response.status_code != 202:
            print(f"❌ 请求失败: {response.text}")
            return None
            
        task_id = response.json()["task_id"]
        print(f"✅ 任务已启动，ID: {task_id}，要求保留音频文件")
        
        # 等待任务完成
        print("⏳ 等待任务完成...")
        while True:
            status_response = requests.get(f"{BASE_URL}/api/status/{task_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                print("✅ 任务已完成!")
                
                # 检查是否有音频URL
                result = status_data.get("result", {})
                audio_url = result.get("audio_url")
                
                if audio_url:
                    print(f"✅ 成功获取音频下载URL: {audio_url}")
                    
                    # 尝试下载音频文件
                    print("⬇️  正在下载音频文件...")
                    audio_response = requests.get(f"{BASE_URL}{audio_url}")
                    
                    if audio_response.status_code == 200:
                        # 保存音频文件
                        filename = f"downloaded_audio_{task_id}.mp3"
                        with open(filename, 'wb') as f:
                            f.write(audio_response.content)
                        print(f"✅ 音频文件下载成功: {filename} ({len(audio_response.content)} 字节)")
                        
                        # 检查文件是否存在
                        if os.path.exists(filename):
                            file_size = os.path.getsize(filename)
                            print(f"✅ 文件验证通过，大小: {file_size} 字节")
                            
                            # 可选：删除测试文件
                            # os.remove(filename)
                            # print("🗑️  测试文件已清理")
                        else:
                            print("❌ 文件不存在")
                    else:
                        print(f"❌ 音频文件下载失败: {audio_response.status_code}")
                else:
                    print("⚠️  未返回音频URL，可能未启用音频保留功能")
                
                return task_id
            elif status_data["status"] == "failed":
                print(f"❌ 任务失败: {status_data.get('error', '未知错误')}")
                return None
                
            time.sleep(3)
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def main():
    """主函数"""
    print("🚀 开始测试音频文件功能")
    
    # 测试健康检查
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code != 200:
            print("❌ 服务不可用，请确保远程服务已启动")
            return
        print("✅ 服务正常运行")
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return
    
    # 测试音频文件功能
    test_audio_feature()
    
    print("\n🏁 测试完成!")

if __name__ == "__main__":
    main()