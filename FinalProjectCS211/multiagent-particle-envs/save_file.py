import os
import numpy as np
def save_with_unique_name(file_name, data):
    counter = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reward_dir = os.path.join(current_dir, "reward_records")
    dir = os.path.join(reward_dir, file_name+".npy")
    while os.path.exists(dir):
        dir = os.path.join(reward_dir, f"{file_name}_{counter}.npy")
        counter += 1

    # Lưu tệp với tên mới hoặc tên gốc nếu chưa tồn tại
    np.save(dir, data)