import torch as T

a = T.tensor([[ 0.1474,  0.0657,  0.0593, -0.1459, -0.0573],
              [ 0.2907,  0.0831,  0.1645, -0.1456,  0.0097],
              [ 0.2064,  0.0564,  0.0524, -0.1494, -0.0301],
              [ 0.2522,  0.0038,  0.0903, -0.1335, -0.0755],
              [ 0.3102, -0.0563,  0.1275, -0.1660, -0.0023],
              [ 0.3414,  0.0400,  0.1595, -0.2264,  0.1377]])

action = T.tensor([[1, 2, 4, 2, 3, 2]])  # Kích thước hiện tại [1, 6]

# Chuyển `action` thành kích thước [6, 1]
action = action.T  # Chuyển đổi từ [1, 6] thành [6, 1]

# Lấy giá trị từ `a` dựa trên các chỉ số trong `action`
values = a.gather(1, action)

print(values)
