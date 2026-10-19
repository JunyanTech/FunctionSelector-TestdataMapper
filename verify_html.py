"""Quick verification of the generated Bokeh HTML document."""
from __future__ import annotations

import json
import re
import sys

html_path = sys.argv[1]
html = open(html_path, encoding="utf-8").read()

print("file size:", len(html))
for marker in [
    "Training data and the four chosen ideal functions",
    "Test data mapped to the chosen ideal functions",
]:
    print(repr(marker), "->", marker in html)

match = re.search(
    r'<script type="application/json" id="[^"]+">(.*?)</script>',
    html,
    re.S,
)
print("embedded bokeh json found:", bool(match))
if match:
    data = json.loads(match.group(1))

    def find_keys(obj, key, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    out.append(v)
                find_keys(v, key, out)
        elif isinstance(obj, list):
            for item in obj:
                find_keys(item, key, out)

    legend = []
    find_keys(data, "legend_label", legend)
    print("legend labels found:", len(legend))
    for label in sorted(set(legend))[:12]:
        print("   ", label)

    text = json.dumps(data, ensure_ascii=False)
    print("--- content probes ---")
    for probe in [
        "Training data 1",
        "Ideal function 12",
        "Test -> Ideal 2",
        "Unmapped",
        "delta_y",
    ]:
        print(repr(probe), "->", probe in text)
