#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 19: 正则表达式数据提取实战
从日志/文本中提取结构化信息
"""

import re
import random
import time
from collections import Counter
import os

# ============================================================
# 1. 生成模拟 Nginx 日志（200行）
# ============================================================
print("=" * 60)
print("  1. 生成模拟 Nginx 日志")
print("=" * 60)

IPS = [f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(20)]
PATHS = ['/index.html', '/api/users', '/api/orders', '/products',
         '/login', '/logout', '/static/style.css', '/static/app.js',
         '/admin/dashboard', '/api/products/search?q=phone',
         '/contact', '/about', '/favicon.ico', '/robots.txt']
STATUS_CODES = [200, 200, 200, 200, 200, 200, 301, 302, 304, 400, 403, 404, 500, 502]
METHODS = ['GET', 'GET', 'GET', 'GET', 'POST', 'PUT', 'DELETE']
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0) Chrome/120.0',
    'Mozilla/5.0 (Macintosh) Safari/17.0',
    'curl/8.0',
    'python-requests/2.31',
    'Mozilla/5.0 (Linux) Firefox/121.0',
]
TIMES = [f'2024-{m:02d}-{d:02d}T{h:02d}:{min:02d}:{s:02d}Z'
         for m in range(1,4) for d in range(1,29,3) for h in range(0,24,6)
         for min in random.sample(range(60), 1) for s in random.sample(range(60), 1)][:200]

log_lines = []
for i in range(200):
    ip = random.choice(IPS)
    time_str = random.choice(TIMES)
    method = random.choice(METHODS)
    path = random.choice(PATHS)
    status = random.choice(STATUS_CODES)
    size = random.randint(100, 50000)
    ua = random.choice(USER_AGENTS)
    # 标准 Nginx 格式
    line = f'{ip} - - [{time_str}] "{method} {path} HTTP/1.1" {status} {size} "-" "{ua}"'
    log_lines.append(line)

log_text = '\n'.join(log_lines)
log_path = os.path.join(os.path.dirname(__file__), 'mock_nginx.log')
with open(log_path, 'w') as f:
    f.write(log_text)

print(f"  生成: {log_path} ({len(log_lines)} 行)")
print(f"  前3行示例:")
for line in log_lines[:3]:
    print(f"    {line[:100]}...")


# ============================================================
# 2. 用正则提取日志信息（命名分组）
# ============================================================
print("\n" + "=" * 60)
print("  2. 命名分组提取日志字段")
print("=" * 60)

# 编译正则（使用命名分组 (?P<name>pattern)）
LOG_PATTERN = re.compile(
    r'(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s'      # IP地址
    r'-\s-\s'
    r'\[(?P<timestamp>[^\]]+)\]\s'                          # 时间戳 [dd/MMM/yyyy:HH:mm:ss +ZZZZ]
    r'"(?P<method>\w+)\s'                                    # HTTP方法
    r'(?P<path>[^\s]+)\s'                                    # 请求路径
    r'(?P<protocol>[^"]+)"\s'                                # 协议版本
    r'(?P<status>\d{3})\s'                                   # 状态码
    r'(?P<size>\d+)\s'                                       # 响应大小
    r'"(?P<referer>[^"]*)"\s'                                # Referer
    r'"(?P<user_agent>[^"]*)"'                              # User-Agent
)

# 提取
records = []
for line in log_lines[:5]:  # 先看前5行
    m = LOG_PATTERN.match(line)
    if m:
        records.append(m.groupdict())

print(f"  前5条提取结果:")
for i, rec in enumerate(records):
    print(f"  [{i}] IP={rec['ip']:18s} | {rec['timestamp']:25s} | "
          f"{rec['method']:6s} {rec['path']:30s} | {rec['status']}")


# ============================================================
# 3. 从混杂文本中提取：邮箱、手机号、URL、金额
# ============================================================
print("\n" + "=" * 60)
print("  3. 从混杂文本提取信息")
print("=" * 60)

mixed_text = """
联系我们: support@mycompany.com 或 sales-team@partner.co.uk
客服热线: 138-1234-5678 或 (010) 8234-5678
紧急联系: +86 13912345678
访问官网: https://www.example.com/products?id=123
特价商品: ¥199.99  原价: $299.50  折扣: 50%
发票金额: CNY 1,234.56  欧元 €89,90
用户ID: USR_20240001  订单号: ORD-2024-ABCD-001
IP: 192.168.1.100  端口: 8080
日期: 2024-01-15  时间: 14:30:00
"""

# 各种提取正则
email_pattern    = re.compile(r'[\w\.-]+@[\w\.-]+\.\w{2,}')
phone_pattern    = re.compile(r'(?:\+?86\s*)?1[3-9]\d{1}[-\s]?\d{4}[-\s]?\d{4}')
url_pattern      = re.compile(r'https?://[^\s]+')
money_pattern    = re.compile(r'(?:¥|USD|CNY|EUR|\$)\s*[\d,]+\.?\d*')
ip_pattern       = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
date_pattern     = re.compile(r'\d{4}-\d{2}-\d{2}')
orderid_pattern  = re.compile(r'[A-Z]{2,4}[-_]\d{4,8}[-_]?[A-Z\d]*[-_]?\d*')

extractions = {
    '邮箱': email_pattern.findall(mixed_text),
    '手机号': phone_pattern.findall(mixed_text),
    'URL': url_pattern.findall(mixed_text),
    '金额': money_pattern.findall(mixed_text),
    'IP地址': ip_pattern.findall(mixed_text),
    '日期': date_pattern.findall(mixed_text),
    '订单号': orderid_pattern.findall(mixed_text),
}

for category, items in extractions.items():
    print(f"  {category}: {items}")


# ============================================================
# 4. 用正则做数据清洗
# ============================================================
print("\n" + "=" * 60)
print("  4. 正则数据清洗")
print("=" * 60)

dirty_html = """
<div class="product">
  <h1>  Apple iPhone 15 Pro  </h1>
  <p class="price">   <span>¥7,999</span>   </p>
  <p>Available in: <b>Black</b>, <b>White</b>, <b>Blue</b></p>
  <script>alert('ad')</script>
  <br/>
</div>
"""

# 4a. 去除 HTML 标签
clean_no_html = re.sub(r'<[^>]+>', ' ', dirty_html)
print(f"  去HTML标签: '{clean_no_html.strip()[:80]}...'")

# 4b. 去除多余空白（多个空格→1个）
clean_spaces = re.sub(r'\s+', ' ', clean_no_html).strip()
print(f"  统一空白:   '{clean_spaces[:80]}...'")

# 4c. 去除脚本内容
clean_no_script = re.sub(r'<script[^>]*>.*?</script>', '', dirty_html, flags=re.DOTALL)
print(f"  去脚本标签: '{clean_no_script.strip()[:80]}...'")

# 4d. 提取纯文本内容
text_only = re.sub(r'<[^>]+>', '', dirty_html)
text_only = re.sub(r'\s+', ' ', text_only).strip()
print(f"  纯文本:     '{text_only}'")

# 4e. 统一货币符号
prices = "¥199.99  $299.50  USD 150.00  EUR 89.90  CNY 1,234.56"
normalized = re.sub(r'(¥|\$|USD|EUR|CNY)\s*', '¥', prices)
print(f"  统一货币:   '{normalized}'")


# ============================================================
# 5. re.compile vs 直接使用 — 性能对比
# ============================================================
print("\n" + "=" * 60)
print("  5. re.compile vs 直接使用 — 性能对比")
print("=" * 60)

test_text = log_text  # 200行日志
N_ITERATIONS = 1000

# 方法1: 直接使用 re.match
start = time.perf_counter()
for _ in range(N_ITERATIONS):
    for line in log_lines:
        m = re.match(
            r'(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s'
            r'-\s-\s\[(?P<timestamp>[^\]]+)\]\s'
            r'"(?P<method>\w+)\s(?P<path>[^\s]+)\s(?P<protocol>[^"]+)"\s'
            r'(?P<status>\d{3})\s(?P<size>\d+)\s'
            r'"(?P<referer>[^"]*)"\s"(?P<user_agent>[^"]*)"',
            line
        )
t_direct = time.perf_counter() - start
print(f"  直接 re.match: {t_direct*1000:.1f} ms ({N_ITERATIONS} iterations)")

# 方法2: 使用编译后的正则
start = time.perf_counter()
for _ in range(N_ITERATIONS):
    for line in log_lines:
        m = LOG_PATTERN.match(line)
t_compiled = time.perf_counter() - start
print(f"  re.compile:    {t_compiled*1000:.1f} ms ({N_ITERATIONS} iterations)")
print(f"  编译版提速: {t_direct/t_compiled:.1f}x")


# ============================================================
# 6. 命名分组让代码更可读
# ============================================================
print("\n" + "=" * 60)
print("  6. 命名分组 (?P<name>...) 对比")
print("=" * 60)

sample_line = log_lines[0]
print(f"  原始日志: {sample_line[:90]}...")

# --- 方式A: 无命名分组（用索引） ---
m_unnamed = re.match(
    r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s-\s-\s\[([^\]]+)\]\s"(\w+)\s([^\s]+)',
    sample_line
)
print(f"\n  无命名分组: m.group(1)={m_unnamed.group(1)},  m.group(2)={m_unnamed.group(2)}")
print(f"  问题: 一个月后你能记住 group(2) 是什么吗？")

# --- 方式B: 命名分组 ---
m_named = LOG_PATTERN.match(sample_line)
print(f"\n  命名分组: m['ip']={m_named['ip']},  m['timestamp']={m_named['timestamp']}")
print(f"  优势: m['ip'] 永远比 m.group(1) 更可读 ✓")


# ============================================================
# 7. 实际应用：日志分析
# ============================================================
print("\n" + "=" * 60)
print("  7. 实际应用 — 日志分析")
print("=" * 60)

# 解析全部200行日志
all_records = []
for line in log_lines:
    m = LOG_PATTERN.match(line)
    if m:
        all_records.append(m.groupdict())

print(f"  成功解析: {len(all_records)}/{len(log_lines)} 行")

# 7a. 每个IP的访问次数
ip_counter = Counter(r['ip'] for r in all_records)
print(f"\n  📊 访问量 Top 5 IP:")
for ip, count in ip_counter.most_common(5):
    bar = '█' * (count // 2)
    print(f"    {ip:18s} {count:3d} 次 {bar}")

# 7b. HTTP状态码分布
status_counter = Counter(r['status'] for r in all_records)
print(f"\n  📊 HTTP状态码分布:")
for status, count in status_counter.most_common():
    label = {'200':'OK','301':'Moved','302':'Found','304':'Not Modified',
             '400':'Bad Request','403':'Forbidden','404':'Not Found',
             '500':'Internal Error','502':'Bad Gateway'}.get(status, '?')
    print(f"    {status} ({label:15s}): {count:3d} 次")

# 7c. 404错误的URL分布
error_404_urls = Counter(r['path'] for r in all_records if r['status'] == '404')
if error_404_urls:
    print(f"\n  📊 404 错误 URL 分布:")
    for url, count in error_404_urls.most_common():
        print(f"    {url:40s} {count} 次")

# 7d. 请求方法分布
method_counter = Counter(r['method'] for r in all_records)
print(f"\n  📊 HTTP方法分布:")
for method, count in method_counter.most_common():
    print(f"    {method:8s}: {count:3d} 次")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  Day 19 总结：正则表达式核心")
print("=" * 60)
print("""
  常用模式速查:
  \\d+        → 数字
  \\w+        → 字母数字下划线
  [\\w\\.-]+@ → 邮箱前缀
  \\d{1,3}\\. → IP段
  https?://  → URL
  <[^>]+>    → HTML标签（非贪婪）
  \\s+        → 空白字符

  关键技巧:
  1. 用 re.compile() 提升重复使用的性能
  2. 用 (?P<name>...) 命名分组提升可读性
  3. re.sub() 做文本清洗，re.findall() 做批量提取
  4. 在 regex101.com 上先调试，再写入代码
  5. 注意贪婪匹配(.*) vs 非贪婪匹配(.*?)
""")
