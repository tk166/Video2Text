#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试音频文件清理功能的脚本
"""

import requests
import time
import os

# 测试服务器地址
BASE_URL = "http://localhost:5001"

def test_delete_audio_file(task_id):
    """测试删除音频文件功能"""
    print(f"🗑️  测试删除音频文件功能，任务ID: {task_id}")
    
    try:
        # 删除音频文件
        response = requests.delete(f"{BASE_URL}/api/audio/{task_id}")
        
        if response.status_code == 200:
            print("✅ 音频文件删除成功")
            result = response.json()
            print(f"消息: {result.get('message', '')}")
            return True
        else:
            print(f"❌ 删除音频文件失败: {response.status_code}")
            result = response.json()
            print(f"错误: {result.get('error', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 删除音频文件时出错: {e}")
        return False

def test_cleanup_expired_files():
    """测试清理过期文件功能"""
    print("🧹 测试清理过期文件功能")
    
    try:
        # 清理过期文件（设置为1小时以测试）
        response = requests.post(f"{BASE_URL}/api/cleanup", 
                                json={"max_age_hours": 1})
        
        if response.status_code == 200:
            print("✅ 清理过期文件请求成功")
            result = response.json()
            deleted_count = result.get('deleted_count', 0)
            print(f"消息: {result.get('message', '')}")
            print(f"删除文件数: {deleted_count}")
            return True
        else:
            print(f"❌ 清理过期文件失败: {response.status_code}")
            result = response.json()
            print(f"错误: {result.get('error', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 清理过期文件时出错: {e}")
        return False

def test_list_tasks():
    """测试列出任务功能"""
    print("📋 测试列出任务功能")
    
    try:
        response = requests.get(f"{BASE_URL}/api/tasks")
        
        if response.status_code == 200:
            result = response.json()
            tasks = result.get('tasks', [])
            print(f"✅ 成功获取任务列表，共 {len(tasks)} 个任务")
            
            # 显示前3个任务的信息
            for task in tasks[:3]:
                has_audio = "✓" if task.get('has_audio', False) else "✗"
                print(f"  - {task['task_id']}: {task['status']} (音频: {has_audio})")
            return True
        else:
            print(f"❌ 获取任务列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取任务列表时出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试音频文件清理功能")
    
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
    
    # 测试列出任务
    test_list_tasks()
    
    # 测试清理过期文件
    test_cleanup_expired_files()
    
    print("\n🏁 清理功能测试完成!")

if __name__ == "__main__":
    main()