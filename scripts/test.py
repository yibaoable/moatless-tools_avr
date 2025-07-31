from unidiff import PatchSet
from moatless.benchmark.utils import get_moatless_instance


import re


def fix_patch_format(patch_str):

    
    # 尝试用 unidiff 自动修复 hunk header
    try:
        print(patch_str)
        fixed_patch = PatchSet(patch_str)
        
        return str(fixed_patch)
    except Exception as e:
        print("❌ 无法解析 patch，可能格式严重错误:", e)
        return patch_str
    
instance = get_moatless_instance("django__django-custom1")
test_patch = instance.get("test_patch", "")

fixed_test_patch = fix_patch_format(test_patch)
instance["test_patch"] = fixed_test_patch

print(repr(instance["test_patch"]))

# 再次验证是否修复成功
try:
    patch = PatchSet(instance["test_patch"])
    print("✅ Patch 格式已修复")
except Exception as e:
    print("❌ 修复失败:", e)


