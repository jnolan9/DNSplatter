import numpy as np

a=np.load("/home/jnolan9/DNSplatter/phomo_init_cornelia/normals_from_pretrain/FC21B0008195_11275021648F1G.npy")

b = a[:,0,1]
print(b)
a = a.reshape(3,-1)
print(a.shape)
print(a[:,1])


norm = np.linalg.norm(a, axis=0, keepdims=True)  # Shape: (1, 1048576)
print(norm)
print(norm.shape)

