with open('swim_video_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the pattern
idx = content.find('# 绘制所有候选区域')
if idx >= 0:
    snippet = content[idx:idx+400]
    print("Found at index:", idx)
    print("Snippet (40 chars per line):")
    for i in range(0, len(snippet), 40):
        print(repr(snippet[i:i+40]))
else:
    print('Pattern not found')
