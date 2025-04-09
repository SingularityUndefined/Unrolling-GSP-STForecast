
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import matplotlib.pyplot as plt


def dataframe_from_tensorboard(log_dir, selected_tag=None):
    """
    处理 TensorBoard 日志数据
    :param log_dir: TensorBoard 日志目录
    :return: 返回处理后的数据
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 获取所有的 tags
    tags = ea.Tags()
    print("Available tags:", tags)

    # 提取某个 tag 的数据（例如 'train/loss'）
    scalar_data = ea.Scalars(selected_tag)

    # 将数据转换为 DataFrame

    df = pd.DataFrame({
        "step": [e.step for e in scalar_data],
        "value": [e.value for e in scalar_data]
    })

    return df

def plot_dataframe(df, title="Training Loss Curve"):
    """
    绘制 DataFrame 数据
    :param df: DataFrame 数据
    :param title: 图表标题
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["value"])
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # log_dir = "dense_logs/TB_log/diffM_1channel_4_hop_4_int_lr_5e-04_seed3407_true_loss/PEMS04_Huber_5b25_4h_6f"  # 替换为你的 TensorBoard 日志目录
    df = dataframe_from_tensorboard("dense_logs/TB_log/diffM_1channel_4_hop_4_int_lr_5e-04_seed3407_true_loss/PEMS04_Huber_5b25_4h_6f", "Loss_batch")
    print(df)  # 打印前几行数据
    # plot_dataframe(df)

