# Cookie 文件格式说明

## Netscape Cookie 文件格式

yt-dlp 支持 Netscape 格式的 cookie 文件。这种格式的文件通常具有以下特征：

### 文件格式
```
# Netscape HTTP Cookie File
# http://www.netscape.com/newsref/std/cookie_spec.html
# This is a generated file!  Do not edit.

.example.com	TRUE	/	FALSE	2147483647	NAME	VALUE
```

### 字段说明
1. **域名** (Domain) - cookie 适用的域名
2. **标志** (Flag) - TRUE/FALSE，表示是否为域名标志
3. **路径** (Path) - cookie 适用的路径
4. **安全标志** (Secure) - TRUE/FALSE，表示是否仅通过 HTTPS 传输
5. **过期时间** (Expires) - Unix 时间戳，表示 cookie 过期时间
6. **名称** (Name) - cookie 名称
7. **值** (Value) - cookie 值

### 示例
```
# Netscape HTTP Cookie File
.youtube.com	TRUE	/	TRUE	1768000000	SID	XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
.youtube.com	TRUE	/	TRUE	1768000000	SSID	XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
.youtube.com	TRUE	/	TRUE	1768000000	APISID	XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## 如何获取 Cookie 文件

### 方法 1: 使用浏览器扩展
1. 安装 "Get cookies.txt" 或类似扩展
2. 登录目标网站
3. 使用扩展导出 cookie 文件

### 方法 2: 使用命令行工具
```bash
# 使用 curl 和 jq (需要安装 jq)
curl -s 'http://www.example.com' -b cookies.txt
```

### 方法 3: 手动创建
创建一个文本文件，按照上述格式手动添加 cookie 信息。

## 注意事项

1. **文件权限**: 确保 cookie 文件只有您自己可以读取，不要包含敏感信息
2. **时效性**: cookie 可能会过期，需要定期更新
3. **安全性**: 不要在公共计算机或不安全的环境中存储 cookie 文件
4. **格式验证**: yt-dlp 要求 cookie 文件必须是有效的 Netscape 格式