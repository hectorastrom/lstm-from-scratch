def main():
    print("Hello from lstm-from-scratch!")
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    print(a.shape)
    b = a.transpose
    print(matmul(a, b))
    # print(a.shape)
    # print(a.dim)
    # print(a)
    # print(a[0][2])
    # a.transpose
    # print(a)
    
    print(dot_product([1,2, 3], [1, 2, 3]))
    # yikes i am nearly out of lines...
    
class Tensor:
    """2D, immutable tensors (you can tranpose it tho!). THESE ARE JUST 2D RIGHT NOW"""
    def __init__(self, data):
        if not isinstance(data, list): 
            data = [data]
        self.data = data
        self.shape= self._get_shape(data)
        self.dim = self._get_dim()
        
    def _get_shape(self, data):
        if isinstance(data, list):
            return [len(data)] + self._get_shape(data[0])
        else: 
            return []
    
    def _get_dim(self):
        return len(self.shape)
    
    @property
    def transpose(self):
        """Returns a copy of the transposed tensor"""
        if self.dim == 1: 
            self.unsqueeze()
        data_T = [[0 for _ in range(self.shape[-2])] for _ in range(self.shape[-1])] # inverted dims
        for i in range(self.shape[-2]):
            for j in range(self.shape[-1]):
                data_T[j][i] = self.data[i][j]

        return Tensor(data_T)
    
    def unsqueeze(self):
        """Prepends leading dim of 1 in place"""
        self.data = [self.data]
        self.shape = [1] + self.shape
        self.dim = self._get_dim()
    
    def __getitem__(self, key):
        return Tensor(self.data.__getitem__(key))
    
    def copy(self):
        return Tensor(self.data)

    def __str__(self):
        return f"tensor{self.data}"

def dot_product(A : list, B: list) -> int | float:
    assert len(A) == len(B), "can't compute product of two diff size lists"
    return sum([a * b for a, b in zip(A, B)])

def matmul(A : Tensor, B: Tensor):
    """matmul for 2D tensors"""
    # output size of (n x m) x (m x p) = (n x p)
    assert A.shape[1] == B.shape[0], "Shape mismatch of tensors"
    out_data = [[0 for _ in range(A.shape[0])] for _ in range(B.shape[1])]
    for i, row in enumerate(A):
        for j, col in enumerate(B.transpose):
            product = dot_product(row.data, col.data)
            out_data[i][j] = product
    
    return Tensor(out_data)


if __name__ == "__main__":
    main()
