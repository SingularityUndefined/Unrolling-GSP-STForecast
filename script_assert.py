for i in range(3):
    try:
        assert i >= 0, f"外层循环: i {i} 应该是非负数"
        for j in range(3):
            try:
                assert j >= 0, f"内层循环: j {j} 应该是非负数"
                # 你的代码
                # 模拟代码执行
                if i == 1 and j == 2:  # 假设在某种条件下发生错误
                    raise ValueError("示例错误")
            except AssertionError as ae:
                print(f"断言失败: {ae}")
            except Exception as e:
                print(f"错误发生在: 外层循环 i={i}, 内层循环 j={j} - 错误信息: {e}")
    except AssertionError as ae:
        print(f"断言失败: {ae}")
    except Exception as e:
        print(f"错误发生在外层循环 i={i} - 错误信息: {e}")
