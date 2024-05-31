from scipy.sparse import coo_matrix
data = {
    "x": [1, 3], # pixel
    "y": [1, 3], # pixel
    "e": [5, 8], # GeV
}
shape =(5, 5)
coo = coo_matrix(
    (data["e"], (data["x"], data["y"])), shape=shape,
)
print(coo.toarray())
