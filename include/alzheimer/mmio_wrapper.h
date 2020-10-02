
template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
        int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
        int **aColInd, int extendSymMatrix);

