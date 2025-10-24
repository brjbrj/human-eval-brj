# 1. 拼接完整函数（prompt + 模型completion）
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:  # 模型的<=
                return True
    return False

# 2. 复制测试用例
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True  # 3.9-4.0=0.1 ≤0.3 → 对
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False # 3.9-4.0=0.1 >0.05 → 对
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True       #5.9-5.0=0.9 ≤0.95 → 对
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False      #5.9-5.0=0.9 >0.8 → 对
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True  #2.0-2.0=0 ≤0.1 → 对
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True       #2.2-3.1=0.9 ≤1.0 → 对
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False      #所有差>0.5 → 对

# 3. 手动执行测试
check(has_close_elements)