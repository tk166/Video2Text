#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 converter_app_remote.py 的简单脚本
"""

import requests
import time
import json
from crypto_utils import encrypt_data

# 测试服务器地址
BASE_URL = "http://localhost:5001"

def test_health():
    """测试健康检查接口"""
    print("🧪 测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_process_video(video_url, cookie_file=None, encrypted_cookie_data=None):
    """测试视频处理接口"""
    print(f"\n🎬 测试视频处理接口: {video_url}")
    try:
        # 发送处理请求
        payload = {"url": video_url}
        if cookie_file:
            payload["cookie_file"] = cookie_file
        if encrypted_cookie_data:
            payload["encrypted_cookie_data"] = encrypted_cookie_data

        response = requests.post(
            f"{BASE_URL}/api/process",
            json=payload
        )

        print(f"状态码: {response.status_code}")
        if response.status_code != 202:
            print(f"❌ 请求失败: {response.text}")
            return None

        result = response.json()
        task_id = result.get("task_id")
        print(f"任务ID: {task_id}")
        print(f"消息: {result.get('message')}")
        return task_id
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def test_get_status(task_id):
    """测试获取任务状态接口"""
    print(f"\n🔍 查询任务状态: {task_id}")
    try:
        response = requests.get(f"{BASE_URL}/api/status/{task_id}")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"状态: {result.get('status')}")
        print(f"进度: {result.get('progress')}")
        return result
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def test_list_tasks():
    """测试列出所有任务接口"""
    print(f"\n📋 列出所有任务...")
    try:
        response = requests.get(f"{BASE_URL}/api/tasks")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"任务数量: {len(result.get('tasks', []))}")
        for task in result.get('tasks', [])[:3]:  # 只显示前3个任务
            print(f"  - {task['task_id']}: {task['status']}")
        return result
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def test_encrypted_cookie_feature():
    """测试加密cookie功能"""
    print(f"\n🔐 测试加密Cookie功能...")

    # 创建示例cookie内容
    sample_cookie = """# Netscape HTTP Cookie File
.youtube.com	TRUE	/	TRUE	1768000000	SID	XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""

    # 加密cookie数据
    encrypted_cookie = encrypt_data(sample_cookie)
    print(f"✅ Cookie数据已加密，长度: {len(encrypted_cookie)} 字符")

    # 使用加密cookie发送请求
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    task_id = test_process_video(test_url, encrypted_cookie_data=encrypted_cookie)

    if task_id:
        print(f"✅ 使用加密cookie的任务已启动，ID: {task_id}")
        return task_id
    else:
        print("❌ 使用加密cookie的任务启动失败")
        return None

def wait_for_completion(task_id, timeout=300):
    """等待任务完成"""
    print(f"\n⏳ 等待任务完成 (超时: {timeout}秒)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = test_get_status(task_id)
        if result:
            status = result.get('status')
            if status == 'completed':
                print("✅ 任务已完成!")
                print(f"📝 识别结果预览: {result.get('result', {}).get('transcription', '')[:100]}...")
                return True
            elif status == 'failed':
                print("❌ 任务失败!")
                print(f"错误信息: {result.get('error')}")
                return False
        time.sleep(5)

    print("⏰ 等待超时!")
    return False

def main():
    """主函数"""
    print("🚀 开始测试 converter_app_remote.py")

    # 1. 测试健康检查
    if not test_health():
        print("❌ 健康检查失败，请确保服务已启动")
        return

    # 2. 测试普通处理（不使用cookie）
    print("\n=== 测试普通处理 ===")
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 示例链接
    task_id = test_process_video(test_url, None)  # 不使用cookie文件

    if not task_id:
        print("❌ 无法启动任务")
        return

    # 3. 测试列出任务
    test_list_tasks()

    # 4. 等待普通任务完成
    if wait_for_completion(task_id):
        # 5. 最后再检查一次状态
        test_get_status(task_id)
        test_list_tasks()

    # 6. 测试加密cookie功能
    print("\n=== 测试加密Cookie功能 ===")
    encrypted_task_id = test_encrypted_cookie_feature()
    if encrypted_task_id:
        # 等待加密cookie任务完成
        if wait_for_completion(encrypted_task_id):
            test_get_status(encrypted_task_id)
            test_list_tasks()

    print("\n🏁 测试完成!")

if __name__ == "__main__":
    main()