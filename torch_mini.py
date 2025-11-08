import random

class Tensor:
    """2D, immutable tensors (you can tranpose it tho!). THESE ARE JUST 2D RIGHT NOW"""
    def __init__(self, data):
        if not isinstance(data, list): 
            data = [data]
        self.data = data
        self.shape = self._get_shape(data)
        self.dim = self._get_dim()
        # needed to manually redefine fillwith const to avoid infinite recursion
        self.grad = Tensor._fill_with(self.shape, lambda x:1, x=0) # all ones

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

    @staticmethod
    def _fill_with(shape, fn, **kwargs):
        if len(shape) > 1:
            out = []
            for _ in range(shape[0]):
                out.append(Tensor._fill_with(shape[1:], fn, **kwargs))
            return out
        else:
            return [fn(**kwargs) for _ in range(shape[0])]

    @staticmethod
    def fill(shape, c):
        """Constructor to fill a tensor with with a constant c"""
        return Tensor(Tensor._fill_with(shape, lambda x:c, x=0))

    @staticmethod
    def randn(shape, mu=0, sigma=1):
        return Tensor(Tensor._fill_with(shape, random.gauss, mu=mu, sigma=sigma))

    @staticmethod
    def zeros(shape):
        return Tensor.fill(shape, 0)

    def __getitem__(self, key):
        cur_list = self.data
        if isinstance(key, (list, tuple)):
            for idx in key[:-1]:
                cur_list = cur_list[idx] # go down a level
        else:
            key = [key]
        return cur_list[key[-1]]

    def __setitem__(self, key : int | list | tuple, data):
        """key can be a single idx or list of values"""
        cur_list = self.data
        if isinstance(key, (list, tuple)):
            for idx in key[:-1]: # exclude last one
                cur_list = cur_list[idx] # go down a level
        else:
            key = [key]
        cur_list[key[-1]] = data # final assignment

    def copy(self):
        copy_t = Tensor(self.data)
        copy_t.grad = self.grad
        return copy_t

    def __str__(self):
        return f"tensor{self.data}"

    def __repr__(self):
        return f"tensor({self.data}, shape={tuple(self.shape)})"

    @staticmethod
    def _gen_idx(shape):
        if len(shape) == 1:
            return [[i] for i in range(shape[0])] # inside list for consistency
        else:
            temp = []
            prev = Tensor._gen_idx(shape[1:])
            for i in range(shape[0]):
                for partner in prev:
                    temp.append([i] + partner)
            return temp

    def __add__(self, other):
        assert self.shape == other.shape, "add: shape mismatch of tensors"
        self_copy = self.copy()
        if self_copy.shape == [1]: # constant
            return Tensor(self_copy.data + other.data)
        for idx in Tensor._gen_idx(self_copy.shape):
            data = self_copy.__getitem__(idx) + other.__getitem__(idx)
            self_copy.__setitem__(idx, data)
        return self_copy

    @staticmethod
    def dot_product(A : list, B: list) -> int | float:
        assert len(A) == len(B), "can't compute product of two diff size lists"
        return sum([a * b for a, b in zip(A, B)])

    @staticmethod
    def matmul(A, B):
        """matmul for 2D tensors"""
        # output size of (n x m) x (m x p) = (n x p)
        assert A.shape[1] == B.shape[0], "matmul: shape mismatch of tensors"
        out_data = [[0 for _ in range(A.shape[0])] for _ in range(B.shape[1])]
        for i, row in enumerate(A):
            for j, col in enumerate(B.transpose):
                product = Tensor.dot_product(row.data, col.data)
                out_data[i][j] = product

        return Tensor(out_data)
