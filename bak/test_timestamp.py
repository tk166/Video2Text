#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试时间戳功能的简单脚本
"""

import requests
import time
import json

# 测试服务器地址
BASE_URL = "http://localhost:5001"

def test_timestamp_feature():
    """测试时间戳功能"""
    print("⏱️  测试时间戳功能...")
    
    try:
        # 发送处理请求
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 示例链接
        payload = {"url": test_url}
        
        response = requests.post(f"{BASE_URL}/api/process", json=payload)
        
        if response.status_code != 202:
            print(f"❌ 请求失败: {response.text}")
            return None
            
        task_id = response.json()["task_id"]
        print(f"✅ 任务已启动，ID: {task_id}")
        
        # 等待任务完成
        print("⏳ 等待任务完成...")
        while True:
            status_response = requests.get(f"{BASE_URL}/api/status/{task_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                print("✅ 任务已完成!")
                
                # 检查是否有时间戳数据
                result = status_data.get("result", {})
                timestamp = result.get("timestamp", [])
                
                if timestamp:
                    print(f"✅ 成功获取时间戳数据，共 {len(timestamp)} 个时间点")
                    print(f"📝 前5个时间戳: {timestamp[:5]}")

                    # 验证时间戳格式
                    if len(timestamp) > 0:
                        first_stamp = timestamp[0]
                        if isinstance(first_stamp, list) and len(first_stamp) == 2:
                            print(f"✅ 时间戳格式正确: [start_ms, end_ms]")
                            print(f"   第一个字符时间: {first_stamp[0]}ms -> {first_stamp[1]}ms")

                            # 显示几个示例
                            print(f"\n📊 时间戳数据示例:")
                            for i, (start, end) in enumerate(timestamp[:10]):
                                print(f"   字符 {i}: {start}ms -> {end}ms ({end-start}ms)")
                        else:
                            print(f"⚠️  时间戳格式可能不正确: {first_stamp}")
                else:
                    print("⚠️  未找到时间戳数据")
                
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
    print("🚀 开始测试时间戳功能")
    
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
    
    # 测试时间戳功能
    test_timestamp_feature()
    
    print("\n🏁 测试完成!")

if __name__ == "__main__":
    main()