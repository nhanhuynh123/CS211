# import numpy as np
# buffer = []
# for i in range (3):
#     buffer.append(np.zeros((10, 3, 84, 84)))

# state = np.random.randint(0, 10, (3, 84, 84))
# print(state.shape)
# buffer[0][0] = state

# # a = np.random.randint(0, 10, size=(457, 120))
# # print(a.shape)
# # buffer[0][0] = a
# print(np.array(buffer)[0][0])

import torch as T
import numpy as np
test = np.random.randint(0,10,(2,2,3))
print(test)
test_transpose = np.transpose(test, (2, 0, 1))
print(test_transpose)

# for i in range(5):
#     test.append(T.rand(1))

# print(test)

# temp = []
# for i in range(5):
#     if i == 0:
#         temp.append(T.cat([act for act in test[i:i+2]], dim=-1))
#     elif i == 4:
#         temp.append(T.cat([act for act in test[i-1:i+1]], dim=-1))

#     else:
#         temp.append(T.cat([act for act in test[i-1:i+2]], dim=-1))

# for i in range(5):
#     print(temp[i])