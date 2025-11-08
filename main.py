import math
from torch_mini import Tensor

def main():
    print("Hello from lstm-from-scratch!")
    A = Tensor.randn((2,4), 0, 1)
    B = Tensor.randn((2,4), 100, 1)
    
    print("A and B")
    print(A)
    print()
    print(B)
    print("---")
    
    print("A + B")
    print(repr(A + B))
    # d = Tensor.zeros((1, 2, 3))
    print("---")
    idx = [0, 3]
    print(f"at idx{idx}: {A[idx]}")
    # yikes i am nearly out of lines...

class FCC:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Xavier initialization
        sigma = math.sqrt(1 / (in_dim * out_dim)) # should actually be of layer l-1 (prev layer)
        self.weights = Tensor.randn([out_dim, in_dim], 0, sigma)
        self.bias = Tensor.zeros([out_dim])
    
    def forward(self, x):
        return Tensor.matmul(self.weights, x) + self.bias
        

if __name__ == "__main__":
    main()
