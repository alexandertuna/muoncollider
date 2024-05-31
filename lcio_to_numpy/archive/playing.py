"""
Playing with converting lists into image matrices.

e.g., if we have a calorimeter hit at cell (row=1, col=2), with E=255, 
and a 3x3 array of cells, how can we convert that into an image like:

((0, 0, 0),
 (0, 0, 255),
 (0, 0, 0))

The internet offers two options: scipy.sparse.coo_matrix and df.pivot.
Conclusion: pivot sucks
"""

from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

DATA = pd.DataFrame({
    "row": [1],
    "col": [2],
    "val": [255],
})
ROWS = COLS = 3

def main():

    coo = coo_matrix((DATA.val, (DATA.row, DATA.col)), shape=(ROWS, COLS))
    arr = coo.toarray()
    print("coo\n", arr)

    # df_pivot = DATA.pivot('Y', 'X', 'I').values
    # df_pivot = DATA.pivot('row', 'col', 'val', fill_value = 0).values
    # print(df_pivot)
    # print(pd.crosstab(index=DATA.row, columns=DATA.col, values=DATA.val, aggfunc='sum').values)

if __name__ == "__main__":
    main()
