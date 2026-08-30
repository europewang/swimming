import sys
import re

file_path = r"d:\myflie\ALL_CODE\swimming\swim_video_v3.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False

for line in lines:
    if line.startswith("# ============================================================"):
        # We need to find the block that says Step 1: 颠倒检测
        pass
        
    new_lines.append(line)
