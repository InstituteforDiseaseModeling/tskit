#include "idmextensions.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TSKIT_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdlib.h>


#define IDM_ALIGNMENT   32

static void
_traverse_u16(tsk_tree_t *tree, tsk_id_t node, tsk_id_t root, void* genome, size_t stride, size_t current_interval /*, int depth */)
{
    // Visit node - store root at current interval for this node
    uint16_t* pibd = (uint16_t*)(genome + node * stride) + current_interval;
    *pibd = root;

    // Visit node's children
    for (tsk_id_t child = tree->left_child[node]; child != TSK_NULL; child = tree->right_sib[child]) {
        _traverse_u16(tree, child, root, genome, stride, current_interval /*, depth + 1 */);
    }
}

static void
_traverse_u32(tsk_tree_t *tree, tsk_id_t node, tsk_id_t root, void* genome, size_t stride, size_t current_interval /*, int depth */)
{
    // Visit node - store root at current interval for this node
    uint32_t* pibd = (uint32_t*)(genome + node * stride) + current_interval;
    *pibd = root;

    // Visit node's children
    for (tsk_id_t child = tree->left_child[node]; child != TSK_NULL; child = tree->right_sib[child]) {
        _traverse_u32(tree, child, root, genome, stride, current_interval /*, depth + 1 */);
    }
}

static void
traverse_recursive_u16(tsk_tree_t *tree, void* genome, size_t stride, size_t current_interval)
{
    // For each unique root, visit root (and it's children)
    for (tsk_id_t root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        _traverse_u16(tree, root, root, genome, stride, current_interval /*, 0 */);
    }
}

static void
traverse_recursive_u32(tsk_tree_t *tree, void* genome, size_t stride, size_t current_interval)
{
    // For each unique root, visit root (and it's children)
    for (tsk_id_t root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        _traverse_u32(tree, root, root, genome, stride, current_interval /*, 0 */);
    }
}

/*
void progress(size_t& percent, size_t current, size_t total, clock_t start)
{
    if ( 100 * current / total > percent ) {
        percent = 100 * current / total;
        clock_t now = clock();
        double elapsed = now - start;
        double remaining = ((double(total) / current - 1) * elapsed) / CLOCKS_PER_SEC;
        std::cout << percent << "% complete, ETA " << remaining << " seconds." << std::endl;
    }
}
*/

PyArrayObject *allocate_array(tsk_treeseq_t *tree_sequence);

PyObject *idm_get_genomes(tsk_treeseq_t *tree_sequence)
{
    tsk_tree_t tree;
    tsk_tree_init(&tree, tree_sequence, 0);
    PyArrayObject *array = allocate_array(tree_sequence);

    int iter = 0;
    size_t current_interval = 0;
    size_t elementsize = PyArray_ITEMSIZE(array);
    void *pdata = PyArray_DATA(array);
    size_t stride = PyArray_STRIDES(array)[0];
    // size_t percent = 0;
    // clock_t start = clock();
    for (iter = tsk_tree_first(&tree), current_interval = 0; iter == 1; iter = tsk_tree_next(&tree), ++current_interval) {
        // progress(percent, current_interval, num_trees, start);
        if ( elementsize == sizeof(uint16_t) ) {
            traverse_recursive_u16(&tree, (uint16_t*)pdata, stride, current_interval);
        } else {
            traverse_recursive_u32(&tree, (uint32_t*)pdata, stride, current_interval);
        }
    }

    // Py_XDECREF(array);

    return array;
}

size_t get_elementsize(tsk_tree_t *tree);

PyArrayObject *allocate_array(tsk_treeseq_t *tree_sequence)
{
    tsk_tree_t tree;
    tsk_tree_init(&tree, tree_sequence, 0);
    tsk_tree_first(&tree);

    tsk_size_t num_nodes = tree.num_nodes;              // rows/genomes
    tsk_size_t num_trees = tree_sequence->num_trees;    // columns/intervals
    printf("# nodes = .. %d\n", num_nodes);
    printf("# trees =    %d\n", num_trees);

    int ndims = 2;
    npy_intp dims[2] = { num_nodes, num_trees };    // { #rows, #columns }

    size_t elementsize = get_elementsize(&tree);
    int typenum = elementsize == 2 ? NPY_UINT16 : NPY_UINT32;

    size_t stride = (num_trees * elementsize + (IDM_ALIGNMENT - 1)) & ~(IDM_ALIGNMENT - 1);
    printf("stride = ... %ld\n", stride);
    // Note that in C order, the first (index 0) dimension is rows, so it is the full stride
    // The second (index 1) dimension is columns, so it is just the element size
    npy_intp strides[2] = { stride, elementsize };

    size_t total_bytes = num_nodes * stride;
    void *pdata;
    // https://stackoverflow.com/a/6563142 - posix_memalign() rather than allocate_aligned()
    posix_memalign(&pdata, IDM_ALIGNMENT, total_bytes);
    printf("bytes =      %ld\n", total_bytes);
    printf("pdata = .... %p\n", pdata);

    // https://stackoverflow.com/a/58001107 - NPY_ARRAY_OWNDATA should tell Numpy to free(pdata) when appropriate
    int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA;

    // PyObject *PyArray_New(PyTypeObject *subtype, int nd, npy_intp const *dims, int typenum, npy_intp* strides, void *data, int itemsize, int flags, PyObject *obj)
    PyArrayObject *array = (PyArrayObject *)PyArray_New(&PyArray_Type, ndims, dims, typenum, strides, pdata, elementsize, flags, NULL);

    printf("NDIM =       %d\n", PyArray_NDIM(array));
    printf("FLAGS = .... %d\n", PyArray_FLAGS(array));
    printf("TYPE =       %d\n", PyArray_TYPE(array));
    printf("DATA = ..... %p\n", PyArray_DATA(array));
    printf("BYTES =      %p\n", PyArray_BYTES(array));
    printf("ITEMSIZE = . %ld\n", PyArray_ITEMSIZE(array));
    printf("SIZE =       %ld\n", PyArray_SIZE(array));
    printf("NBYTES = ... %ld\n", PyArray_NBYTES(array));

    return array;
}

size_t get_elementsize(tsk_tree_t *tree)
{
    size_t size = 2;
    for (tsk_id_t root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        if ( root >= (1 << 16) ) {
            size = 4;
            break;
        }
    }

    printf("sizeof(id) = %ld\n", size);
    return size;
}

PyObject *idm_calculate_ibx(PyObject *args)
{
    PyArrayObject *ibx = NULL;
    PyArrayObject *ndarraygenomes = NULL;
    PyArrayObject *ndvectorids = NULL;
    PyArrayObject *ndvectorintervals = NULL;
    if ( !PyArg_ParseTuple(args, "O!|OO", &PyArray_Type, (PyObject *)&ndarraygenomes,
                                                         (PyObject *)&ndvectorids,
                                                         (PyObject *)&ndvectorintervals) )
        goto out;

    printf("In calculate_ibx...(%p, %p, %p)\n", ndarraygenomes, ndvectorids, ndvectorintervals);

    // Validate arguments
    {
        // ndarraygenomes has two dimensions
        if ( !(PyArray_NDIM(ndarraygenomes) == 2) ) {
            PyErr_SetString(PyExc_RuntimeError, "Genome array must be two dimensional.\n");
            goto out;
        }

        // ndarraygenomes is uint8_t, uint16_t or uint32_t
        switch (PyArray_TYPE(ndarraygenomes)) {
            case NPY_INT8:
            case NPY_UINT8:
            case NPY_INT16:
            case NPY_UINT16:
            case NPY_INT32:
            case NPY_UINT32:
                // all is okay
                break;

            default:
                PyErr_SetString(PyExc_RuntimeError, "Genome array must be integral type of 1, 2, or 4 bytes.\n");
                goto out;
        }

        // ndarraygenomes is appropriately aligned
        if ( !(((size_t)PyArray_DATA(ndarraygenomes) & (IDM_ALIGNMENT -1)) == 0) ) {
            // TODO - ¡check stride also!
            PyErr_Format(PyExc_RuntimeError, "Genome array must be aligned to %d byte boundary.\n", IDM_ALIGNMENT);
            goto out;
        }

        // ndvectorids is None or one dimension (vector) of uint32_t, <= #rows of ndarraygenomes
        if ( (PyObject *)ndvectorids == Py_None ) ndvectorids = NULL;
        if ( ndvectorids ) {
            if ( !(PyArray_Check(ndvectorids) &&
                   (PyArray_NDIM(ndvectorids) == 1) &&
                   (PyArray_TYPE(ndvectorids) == NPY_UINT32) &&
                   (PyArray_DIMS(ndvectorids)[0] <= PyArray_DIMS(ndarraygenomes)[0])) ) {
                       PyErr_SetString(PyExc_RuntimeError, "Genome IDs must be a one dimensional vector of uint32 <= #rows in genome array.\n");
                       goto out;
                   }
        }

        // ndvectorintervals is None or one dimension (vector) of uint32_t, == #columns of ndarraygenomes
        if ( (PyObject *)ndvectorintervals == Py_None ) ndvectorintervals = NULL;
        if ( ndvectorintervals ) {
            if ( !(PyArray_Check(ndvectorintervals) &&
                   (PyArray_NDIM(ndvectorintervals) == 1) &&
                   (PyArray_TYPE(ndvectorintervals) == NPY_UINT32) &&
                   (PyArray_DIMS(ndvectorintervals)[0] == PyArray_DIMS(ndarraygenomes)[1])) ) {
                       PyErr_SetString(PyExc_RuntimeError, "Interval lengths must be a one dimensional vector of uint32 == #columns in genome array.\n");
                       goto out;
                   }
        }
    }

    npy_int size = ndvectorids ? PyArray_DIMS(ndvectorids)[0] : PyArray_DIMS(ndarraygenomes)[0];
    npy_intp dimensions[2] = { size, size };
    ibx = (PyArrayObject *)PyArray_ZEROS(2, dimensions, NPY_UINT32, false);

    // Calculations here...

out:
    // NxN calculated values, unique hash values, entry->hash index map
    return Py_BuildValue("OOO", ibx, Py_None, Py_None);
}