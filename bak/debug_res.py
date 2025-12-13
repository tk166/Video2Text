#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试脚本，用于检查res的实际结构
"""

import json

# 模拟res数据结构
sample_res = [
    {
        'text': '这是一个示例文本',
        'timestamp': [[100, 500], [550, 800], [850, 1200], [1250, 1500], [1550, 1800], [1850, 2200]]
    }
]

def debug_res_structure(res):
    """调试res结构"""
    print("=== 调试res结构 ===")
    print(f"res类型: {type(res)}")
    print(f"res长度: {len(res)}")
    
    if res and len(res) > 0:
        first_item = res[0]
        print(f"第一个元素类型: {type(first_item)}")
        print(f"第一个元素键: {list(first_item.keys())}")
        
        if 'text' in first_item:
            print(f"text字段: {first_item['text'][:50]}...")
            
        if 'timestamp' in first_item:
            timestamp = first_item['timestamp']
            print(f"timestamp字段类型: {type(timestamp)}")
            print(f"timestamp字段长度: {len(timestamp)}")
            if len(timestamp) > 0:
                print(f"前3个时间戳: {timestamp[:3]}")
        else:
            print("没有timestamp字段")
    else:
        print("res为空或长度为0")

if __name__ == "__main__":
    debug_res_structure(sample_res)