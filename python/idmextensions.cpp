#include "idmextensions.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL IDM_ARRAY_API
#include <numpy/arrayobject.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "sha256.h"

#if __AVX2__
#include <emmintrin.h>
#include <immintrin.h>
#endif

#if 0
#define DPRINT(_s_)        printf(_s_)
#define DPRINTFMT(_f_, args...)  printf(_f_, args)
#else
#define DPRINT(_s_)
#define DPRINTFMT(_f, args...)
#endif

#define IDM_ALIGNMENT   32

void progress(size_t& percent, size_t current, size_t total, clock_t start);
PyArrayObject* allocate_aligned_1d(size_t num_elements, size_t itemsize, void* indata, int type);
PyArrayObject* allocate_aligned_2d(size_t num_rows, size_t num_columns, size_t itemsize, void* indata, size_t instride, int type);

/********* Align NumPy Array *********/

PyArrayObject* validate_arg(PyObject *arg);

PyObject *idm_align_data(PyObject *arg)
{
    PyArrayObject* array = validate_arg(arg);
    PyArrayObject* aligned = nullptr;
    
    if ( array ) {
        aligned = allocate_aligned_2d(PyArray_DIMS(array)[0],
                                      PyArray_DIMS(array)[1],
                                      PyArray_ITEMSIZE(array),
                                      PyArray_DATA(array),
                                      PyArray_STRIDES(array)[0],
                                      PyArray_TYPE(array));
    }
    else {
        PyErr_SetString(PyExc_ValueError, "argument must be a two dimensional NumPy array");
    }

    Py_XINCREF(aligned);
    return (PyObject*)aligned;
}

PyArrayObject* validate_arg(PyObject *arg)
{
    PyArrayObject* retval = nullptr;

    if ( PyArray_Check(arg) ) {
        retval = (PyArrayObject*)arg;
        if ( PyArray_NDIM(retval) != 2) {
            retval = nullptr;
        }
    }

    return retval;
}

/********* Genomes from tree sequence *********/

template<typename T>
static void
_traverse(tsk_tree_t *tree, tsk_id_t node, tsk_id_t root, uint8_t* genome, size_t stride, size_t interval, uint32_t* pids /*, int depth*/)
{
    // Visit node - store root at current interval for this node
    // If we haven't (sub)sampled the genomes, store the value
    if (!pids) {
        T* pibd = (T*)(genome + node * stride) + interval;
        *pibd = root;
    }
    // If we've (sub)sampled the genomes, see if this node is selected.
    else if (pids[node] != uint32_t(-1)) {
        // If so, use the index in the pids mapping rather than the node value itself.
        T* pibd = (T*)(genome + pids[node] * stride) + interval;
        *pibd = root;
    }
    // pids != nullptr but pids[node] == -1 ... don't store the interval root information

    // Visit node's children
    for (tsk_id_t child = tree->left_child[node]; child != TSK_NULL; child = tree->right_sib[child]) {
        _traverse<T>(tree, child, root, genome, stride, interval, pids /*, depth + 1 */);
    }
}

template<typename T>
static void
traverse_recursive(tsk_tree_t *tree, uint8_t* genome, size_t stride, size_t interval, uint32_t* pids)
{
    // For each unique root, visit root (and it's children)
    for (tsk_id_t root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        _traverse<T>(tree, root, root, genome, stride, interval, pids /*, 0*/);
    }
}

size_t get_elementsize(tsk_tree_t *tree);
PyArrayObject *allocate_array(size_t num_rows, size_t num_cols, size_t element_size);

typedef struct {
    PyObject_HEAD
    tsk_treeseq_t *tree_sequence;
} TreeSequence;

PyObject *idm_get_genomes(PyObject *args)
{
    DPRINT("In idm_get_genomes()...\n");

    tsk_treeseq_t *tree_sequence;
    tsk_tree_t tree;

    size_t element_size = 0;
    size_t num_rows = 0;
    size_t num_cols = 0;

    PyObject *ts_arg = nullptr;
    PyArrayObject* nd_sample_ids = nullptr;
    size_t num_samples = 0;

    PyArrayObject *array = nullptr;
    PyArrayObject *intervals = nullptr;
    uint8_t* pdata = nullptr;
    size_t stride = 0;
    uint32_t* pids = nullptr;
    int iter = 0;
    size_t current_interval = 0;
    size_t percent = 0;
    std::vector<uint32_t> interval_lengths;
    // DEBUG ONLY clock_t start, end;
    // DEBUG ONLY float elapsed;

    DPRINT("idm_get_genomes: calling PyArg_ParseTuple()...\n");
    if ( !PyArg_ParseTuple(args, "O|O", (PyObject *)&ts_arg, (PyObject *)&nd_sample_ids) )
        goto out;

    DPRINT("idm_get_genomes: checking ts_arg name...\n");
    if ( std::string(ts_arg->ob_type->tp_name) != "_tskit.TreeSequence" ) {
        PyErr_Format(PyExc_RuntimeError, "Argument to get_genomes() must be a _tskit.TreeSequence - got '%s'.\n", ts_arg->ob_type->tp_name);
        goto out;
    }

    tree_sequence = ((TreeSequence *)ts_arg)->tree_sequence;

    DPRINTFMT("tree_sequence->num_trees   = %d\n", uint32_t(tree_sequence->num_trees));
    DPRINTFMT("tree_sequence->num_samples = %d\n", uint32_t(tree_sequence->num_samples));
    for ( uint32_t i = 0; i < uint32_t(std::min(16, int(tree_sequence->num_samples))); ++i ) {
        DPRINTFMT("seq %02d, id = %d\n", i, uint32_t(tree_sequence->samples[i]));
    }

    if ( !tree_sequence ) {
        PyErr_SetString(PyExc_RuntimeError, "TreeSequence is not initialized.\n");
        goto out;
    }

    DPRINT("idm_get_genomes: checking for nd_sample_ids == Py_None\n");
    if ( (PyObject *)nd_sample_ids == Py_None ) {
        DPRINT("nd_sample_ids == Py_None, setting to NULL\n");
        nd_sample_ids = nullptr;
    }

    DPRINT("idm_get_genomes: validating nd_sample_ids (if != nullptr)...\n");
    if ( nd_sample_ids ) {
        if ( !(PyArray_Check(nd_sample_ids) &&
               (PyArray_NDIM(nd_sample_ids) == 1) &&
               (PyArray_TYPE(nd_sample_ids) == NPY_INT32) &&
               (PyArray_DIMS(nd_sample_ids)[0] == tree_sequence->num_samples)) ) {

                   DPRINTFMT("PyArray_NDIM(nd_sample_ids)    = %d\n", uint32_t(PyArray_NDIM(nd_sample_ids)));
                   DPRINTFMT("PyArray_Type(nd_sample_ids)    = %d (expecting %d)\n", uint32_t(PyArray_TYPE(nd_sample_ids)), uint32_t(NPY_INT32));
                   DPRINTFMT("PyArray_DIMS(nd_sample_ids)[0] = %d (expecting %d)\n", uint32_t(PyArray_DIMS(nd_sample_ids)[0]), uint32_t(tree_sequence->num_samples));

                   PyErr_SetString(PyExc_RuntimeError, "Sample IDs must be a one dimensional vector of int32 == # genomes in the tree.\n");
                   goto out;
               }

        for ( uint32_t i = 0; i < uint32_t(std::min(16, int(PyArray_DIMS(nd_sample_ids)[0]))); ++i ) {
            DPRINTFMT("nd_sample_ids[%02d] = %d\n", i, uint32_t(((uint32_t *)PyArray_DATA(nd_sample_ids))[i]));
        }

        int32_t* pids = (int32_t*)PyArray_DATA(nd_sample_ids);
        for ( uint32_t i = 0; i < uint32_t(PyArray_DIMS(nd_sample_ids)[0]); ++i ) {
            if ( pids[i] != -1 ) {
                ++num_samples;
            }
        }
        DPRINTFMT("%d sample ids in nd_sample_ids\n", num_samples);

        DPRINT("nd_sample_ids passed validation, incrementing refcount");
        Py_XINCREF(nd_sample_ids);
    }

    tsk_tree_init(&tree, tree_sequence, 0);

    element_size = get_elementsize(&tree);
    num_rows = num_samples ? num_samples : size_t(tree.num_nodes);
    num_cols = size_t(tree_sequence->num_trees);
    array = allocate_array(num_rows, num_cols, element_size);
    pdata = (uint8_t*)PyArray_DATA(array);
    stride = PyArray_STRIDES(array)[0];
    pids = nd_sample_ids ? (uint32_t*)PyArray_DATA(nd_sample_ids) : nullptr;

    // DEBUG ONLY start = clock();
    for (iter = tsk_tree_first(&tree), current_interval = 0; iter == 1; iter = tsk_tree_next(&tree), ++current_interval) {
        interval_lengths.push_back(uint32_t(tree.right-tree.left));
        progress(percent, current_interval, tree_sequence->num_trees, start);
        if ( PyArray_ITEMSIZE(array) == sizeof(uint16_t) ) {
            traverse_recursive<uint16_t>(&tree, pdata, stride, current_interval, pids);
        } else {
            traverse_recursive<uint32_t>(&tree, pdata, stride, current_interval, pids);
        }
    }
    // DEBUG ONLY end = clock();
    // DEBUG ONLY elapsed = float(end - start) / CLOCKS_PER_SEC;
    // DEBUG ONLY DPRINTFMT("Recursive traversal of all trees took %f seconds\n", elapsed);

    intervals = allocate_aligned_1d(interval_lengths.size(), sizeof(uint32_t), interval_lengths.data(), NPY_UINT32);

out:
    DPRINT("Returning from idm_get_genomes()...\n");

    // return (PyObject *)array;
    // Tuple of (genomes, interval lengths)
    return Py_BuildValue("OO", array, intervals);
}

PyArrayObject *allocate_array(size_t num_rows, size_t num_columns, size_t element_size)
{
    PyArrayObject* array = nullptr;

    DPRINTFMT("# nodes = .. %d\n", num_rows);
    DPRINTFMT("# trees =    %d\n", num_columns);
    DPRINTFMT("element size = %d\n", element_size);

    int typenum = element_size == 2 ? NPY_UINT16 : NPY_UINT32;

    array = allocate_aligned_2d(num_rows, num_columns, element_size, nullptr, 0, typenum);

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

    DPRINTFMT("sizeof(id) = %ld\n", size);
    return size;
}

/********* IBx calculation *********/

#if __AVX2__

uint32_t hsum_epi32_sse3(__m128i v) {                                       // v     = DCBA
    __m128i hi64  = _mm_unpackhi_epi64(v, v);                               // hi64  = DCDC
    __m128i sum64 = _mm_add_epi32(hi64, v);                                 // sum64 = D+D, C+C, D+B, C+A
    __m128i hi32  = _mm_shufflelo_epi16(sum64, _MM_SHUFFLE(1, 0, 3, 2));    // hi32  = D+D, C+C, C+A, D+B
    __m128i sum32 = _mm_add_epi32(sum64, hi32);                             // sum32 = 4D, 4C, A+B+C+D, A+B+C+D
    return _mm_cvtsi128_si32(sum32);                                        // return bottom 32 bits of sum32 == A+B+C+D
}

uint32_t hsum256_epi32_avx(__m256i v) {             // v   = HGFEDCBA
    __m128i vlo = _mm256_castsi256_si128(v);        // vlo = 0000DCBA
    __m128i vhi = _mm256_extracti128_si256(v, 1);   // vhi = 0000HGFE
            vlo = _mm_add_epi32(vlo, vhi);          // vlo = 0000, H+D, G+C, F+B, E+A
    return hsum_epi32_sse3(vlo);                    // return sum of bottom 128 bits of vlo = H+G+F+E+D+C+B+A
}

#endif

typedef void (*simd_comp_t)(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibd, size_t xibd, uint32_t* pintervals, size_t num_unique, uint32_t* ibx);

static void _calculate_ibx(PyArrayObject *ndgenomes, PyArrayObject *ndids, PyArrayObject *ndintervals, PyArrayObject *&ndibx, PyObject *&digests, PyObject *&mapping, simd_comp_t inner_loop_fn);
void inner_loop32(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibx, size_t xibx, uint32_t* pintervals, size_t num_unique, uint32_t* ibx);
void inner_loop16(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibx, size_t xibx, uint32_t* pintervals, size_t num_unique, uint32_t* ibx);
void inner_loop08(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibx, size_t xibx, uint32_t* pintervals, size_t num_unique, uint32_t* ibx);

PyObject *idm_calculate_ibx(PyObject *args)
{
    PyArrayObject *ibx               = nullptr;
    PyArrayObject *ndarraygenomes    = nullptr;
    PyArrayObject *ndvectorids       = nullptr;
    PyArrayObject *ndvectorintervals = nullptr;
    PyObject *hashes                 = nullptr;
    PyObject *mapping                = nullptr;

    if ( !PyArg_ParseTuple(args, "O!|OO", &PyArray_Type, (PyObject *)&ndarraygenomes,
                                                         (PyObject *)&ndvectorids,
                                                         (PyObject *)&ndvectorintervals) )
        goto out;

    DPRINTFMT("In calculate_ibx...(%p, %p, %p)\n", ndarraygenomes, ndvectorids, ndvectorintervals);

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
        if ( (PyObject *)ndvectorids == Py_None ) {
            DPRINT("ndvectorids == Py_None, setting to NULL\n");
            ndvectorids = nullptr;
        }
        if ( ndvectorids ) {
            if ( !(PyArray_Check(ndvectorids) &&
                   (PyArray_NDIM(ndvectorids) == 1) &&
                   (PyArray_TYPE(ndvectorids) == NPY_UINT32) &&
                   (PyArray_DIMS(ndvectorids)[0] <= PyArray_DIMS(ndarraygenomes)[0])) ) {
                       PyErr_SetString(PyExc_RuntimeError, "Genome IDs must be a one dimensional vector of uint32 <= #rows in genome array.\n");
                       goto out;
                   }
            DPRINTFMT("ndvectorids passed validation, incrementing refcount (ndvectorids[1] == %d)\n", ((uint32_t*)PyArray_DATA(ndvectorids))[1]);
            Py_XINCREF(ndvectorids);
        } else {
            npy_intp length = PyArray_DIMS(ndarraygenomes)[0];
            ndvectorids = (PyArrayObject *)PyArray_ZEROS(1, &length, NPY_UINT32, false);
            uint32_t *pid = (uint32_t *)PyArray_DATA(ndvectorids);
            for (uint32_t id = 0; id < length; ++id) {
                *pid++ = id;
            }
            DPRINTFMT("ndvectorids was NULL, allocated 'identity' vector (%d)\n", ((uint32_t *)PyArray_DATA(ndvectorids))[8]);
        }

        // ndvectorintervals is None or one dimension (vector) of uint32_t, == #columns of ndarraygenomes
        if ( (PyObject *)ndvectorintervals == Py_None ) {
            DPRINT("ndvectorintervals == Py_None, setting to NULL\n");
            ndvectorintervals = nullptr;
        }
        if ( ndvectorintervals ) {
            if ( !(PyArray_Check(ndvectorintervals) &&
                   (PyArray_NDIM(ndvectorintervals) == 1) &&
                   (PyArray_TYPE(ndvectorintervals) == NPY_UINT32) &&
                   (PyArray_DIMS(ndvectorintervals)[0] == PyArray_DIMS(ndarraygenomes)[1])) ) {
                       PyErr_SetString(PyExc_RuntimeError, "Interval lengths must be a one dimensional vector of uint32 == #columns in genome array.\n");
                       goto out;
                   }
            if ( (((size_t)PyArray_DATA(ndvectorintervals)) & (IDM_ALIGNMENT-1)) == 0 ) { // ndvectorintervals data is appropriately aligned
                DPRINT("Incoming ndvectorintervals is appropriately aligned, incrementing refcount.\n");
                Py_XINCREF(ndvectorintervals);  // Offset Py_XDECREF below.
            }
            else {  // ndvectorintervals data is _not_ appropriately aligned
                DPRINT("Incoming ndvectorintervals is NOT appropriately aligned, copy to temporary, aligned array.\n");
                ndvectorintervals = allocate_aligned_1d(PyArray_DIMS(ndvectorintervals)[0], sizeof(uint32_t), PyArray_DATA(ndvectorintervals), NPY_UINT32);
                // Py_XDECREF below will release this temporary object.
            }
        } else {
            DPRINT("Incoming ndvectorintervals is None, creating temporary, aligned array of ones.\n");
            size_t count = PyArray_DIMS(ndarraygenomes)[1];
            ndvectorintervals = allocate_aligned_1d(count, sizeof(uint32_t), nullptr, NPY_UINT32);
            uint32_t* pdata = (uint32_t*)PyArray_DATA(ndvectorintervals);
            for ( size_t i = 0; i < count; ++i ) {
                pdata[i] = 1;
            }
            // Py_XDECREF below will release this temporary object.
        }
    }

    switch (PyArray_ITEMSIZE(ndarraygenomes)) {
        case 1:
            _calculate_ibx(ndarraygenomes, ndvectorids, ndvectorintervals, ibx, hashes, mapping, inner_loop08);
            break;

        case 2:
            _calculate_ibx(ndarraygenomes, ndvectorids, ndvectorintervals, ibx, hashes, mapping, inner_loop16);
            break;

        case 4:
            _calculate_ibx(ndarraygenomes, ndvectorids, ndvectorintervals, ibx, hashes, mapping, inner_loop32);
            break;

        default:
            break;
    }

    Py_XDECREF(ndvectorintervals);

out:
    // Tuple of (NxN calculated values, sorted list of unique hash values, hash value->index map)
    return Py_BuildValue("OOO", ibx, hashes, mapping);
}

void get_stats(PyArrayObject* ndgenomes, PyArrayObject* ndids,
               size_t &num_rows, size_t& num_intervals, uint32_t*& pids, uint8_t*& pdata, size_t& stride, size_t& bytes);
void calculate_hashes(size_t num_rows, uint32_t* pids, uint8_t* pdata, size_t stride, size_t bytes, std::vector<std::string>& hashes);
void make_list_of_hashes(std::vector<std::string>& hashes, PyObject*& digests);
void get_unique_hash_values(std::vector<std::string>& hashes, size_t& num_unique, std::map<std::string, size_t>& index_for_hash, PyObject*& mapping);
void allocate_output_array(size_t num_unique, PyArrayObject*& ndibx, uint32_t*& ibx);
void calculate_block_size(size_t bytes, size_t num_rows, size_t& block_size, size_t& num_blocks);
void process_blocks(size_t num_blocks,
                    size_t block_size,
                    size_t num_rows,
                    std::map<std::string, size_t>& index_for_hash,
                    std::vector<std::string>& hashes,
                    uint32_t* ibx,
                    size_t num_unique,
                    uint8_t* pdata,
                    uint32_t* pids,
                    size_t stride,
                    size_t num_intervals,
                    uint32_t* pintervals,
                    simd_comp_t inner_loop);

static void
_calculate_ibx(PyArrayObject *ndgenomes, PyArrayObject *ndids, PyArrayObject *ndintervals,
               PyArrayObject *&ndibx, PyObject *&digests, PyObject *&mapping,
               simd_comp_t inner_loop_fn)
{
    DPRINT("In calculate_ibx()...\n");
    size_t num_rows      = 0;
    size_t num_intervals = 0;

    uint32_t* pids = nullptr;
    uint8_t* pdata = nullptr;
    size_t stride  = 0;
    size_t bytes   = 0;

    get_stats(ndgenomes, ndids,
              num_rows, num_intervals, pids, pdata, stride, bytes);

    std::vector<std::string> hashes;
    calculate_hashes(num_rows, pids, pdata, stride, bytes, hashes);
    make_list_of_hashes(hashes, digests);

    std::map<std::string, size_t> index_for_hash;
    size_t num_unique = 0;
    get_unique_hash_values(hashes, num_unique, index_for_hash, mapping);

    uint32_t* ibx = nullptr;
    allocate_output_array(num_unique, ndibx, ibx);

    size_t block_size = 0;
    size_t num_blocks = 0;
    calculate_block_size(bytes, num_rows, block_size, num_blocks);

    uint32_t* pintervals = (uint32_t*)PyArray_DATA(ndintervals);

    process_blocks(num_blocks,
                   block_size,
                   num_rows,
                   index_for_hash,
                   hashes,
                   ibx,
                   num_unique,
                   pdata,
                   pids,
                   stride,
                   num_intervals,
                   pintervals,
                   inner_loop_fn);

    return;
}

void get_stats(PyArrayObject* ndgenomes, PyArrayObject* ndids,
               size_t &num_rows, size_t& num_intervals, uint32_t*& pids, uint8_t*& pdata, size_t& stride, size_t& bytes)
{
    num_rows      = PyArray_DIMS(ndids ? ndids : ndgenomes)[0];          DPRINTFMT("num_rows      = %ld\n", num_rows);
    num_intervals = PyArray_DIMS(ndgenomes)[1];                          DPRINTFMT("num_intervals = %ld\n", num_intervals);

    pids   = (uint32_t*)PyArray_DATA(ndids);
    pdata  = (uint8_t*)PyArray_DATA(ndgenomes);
    stride = PyArray_STRIDES(ndgenomes)[0];                              DPRINTFMT("stride/genome = %ld\n", stride);
    bytes  = PyArray_DIMS(ndgenomes)[1] * PyArray_ITEMSIZE(ndgenomes);   DPRINTFMT("bytes/genome  = %ld\n", bytes);

    return;
}

void calculate_hashes(size_t num_rows, uint32_t* pids, uint8_t* pdata, size_t stride, size_t bytes, std::vector<std::string>& hashes)
{
    // Calculate SHA-256 hash for each genome by id in ndids
    clock_t start = clock();
    for ( size_t idindex = 0; idindex < num_rows; ++idindex ) {     // There are num_rows genomes for which to calculate the hash
        size_t row = pids[idindex];                                 // Actual ids of interest are in pids, so retrieve actual id/index
        hashes.push_back(sha256(pdata + row * stride, bytes));      // Index into memory based on stride/genome, use "bytes" of memory
    }
    clock_t finish = clock();

    std::cout << double(finish - start) / CLOCKS_PER_SEC << " seconds to calculate SHA-256 hashes for " << num_rows << " genomes." << std::endl;

    return;
}

void make_list_of_hashes(std::vector<std::string>& hashes, PyObject*& digests)
{
    // Create a Python list of hashes. Implicitly, list[N] == hash_value for ndids[N]
    digests = nullptr;
    digests = PyList_New(hashes.size());
    for ( size_t index = 0; index < hashes.size(); ++index) {
        PyObject *string = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, hashes[index].c_str(), hashes[index].size());
        /* int ret = */ PyList_SetItem(digests, index, string);
    }

    return;
}

void get_unique_hash_values(std::vector<std::string>& hashes, size_t& num_unique, std::map<std::string, size_t>& index_for_hash, PyObject*& mapping)
{
    // Determine unique hash values and sort them.
    std::set<std::string> unique(hashes.begin(), hashes.end());
    std::vector<std::string> sorted(unique.begin(), unique.end());
    std::sort(sorted.begin(), sorted.end());

    num_unique = unique.size();
    DPRINTFMT("%ld unique hash values\n", num_unique);

    // Map each hash value to its position in the sorted list of unique hash values.
    for ( size_t i = 0; i < sorted.size(); ++i ) {
        index_for_hash[sorted[i]] = i;
    }

    // Create a Python map of each hash value to its position in the sorted list.
    mapping = nullptr;
    mapping = PyDict_New();
    for ( auto& entry : index_for_hash ) {
        PyObject* key = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, entry.first.c_str(), entry.first.size());
        PyObject* value =  PyLong_FromUnsignedLong( entry.second );
        /* int ret = */ PyDict_SetItem(mapping, key, value);
    }

    return;
}

void allocate_output_array(size_t num_unique, PyArrayObject*& ndibx, uint32_t*& ibx)
{
    npy_intp dimensions[2] = { npy_intp(num_unique), npy_intp(num_unique) };
    ndibx = (PyArrayObject *)PyArray_ZEROS(2, dimensions, NPY_UINT32, false);

    ibx = (uint32_t*)PyArray_DATA(ndibx);
    memset((void*)ibx, 0xFF, size_t(PyArray_NBYTES(ndibx)));

    return;
}

void calculate_block_size(size_t bytes, size_t num_rows, size_t& block_size, size_t& num_blocks)
{
    // Figure out how many genomes fit in the CPU cache.
    size_t cache_size = 19.25 * 1024 * 1024;    // 19.25MB L3 cache on Intel(R) Xeon(R) Gold 6126 CPU
    size_t num_genomes = cache_size / bytes;
    DPRINTFMT("num_genomes in 19.25MB cache is %ld\n", num_genomes);
    block_size = size_t(sqrt(num_genomes));
    DPRINTFMT("block_size is %ld x %ld genomes.\n", block_size, block_size);
    num_blocks = (num_rows + block_size - 1) / block_size;
    DPRINTFMT("num_blocks is %ld\n", num_blocks);

    return;
}

void process_blocks(size_t num_blocks,
                    size_t block_size,
                    size_t num_rows,
                    std::map<std::string, size_t>& index_for_hash,
                    std::vector<std::string>& hashes,
                    uint32_t* ibx,
                    size_t num_unique,
                    uint8_t* pdata,
                    uint32_t* pids,
                    size_t stride,
                    size_t num_intervals,
                    uint32_t* pintervals,
                    simd_comp_t inner_loop)
{
    clock_t start = clock();
    size_t percent = 0;

    for ( size_t yblock = 0; yblock < num_blocks; ++yblock ) {
        progress(percent, yblock, num_blocks, start);

        for ( size_t xblock = yblock; xblock < num_blocks; ++xblock ) {

            size_t ystart = yblock * block_size;
            size_t ystop = std::min(ystart + block_size, num_rows);

            for ( size_t ygenome = ystart; ygenome < ystop; ++ygenome ) {

                size_t yibd = index_for_hash[ hashes[ ygenome ] ];
                size_t xstart = xblock * block_size;
                size_t xstop = std::min(xstart + block_size, num_rows);

                for ( size_t xgenome = xstart; xgenome < xstop; ++xgenome ) {

                    size_t xibd = index_for_hash[ hashes[ xgenome ] ];

                    if ( ibx[ yibd * num_unique + xibd ] == 0xFFFFFFFF ) {
                        inner_loop(pdata, pids, ygenome, stride, xgenome, num_intervals, yibd, xibd, pintervals, num_unique, ibx);
                    }
                }
            }
        }
    }
    clock_t finish = clock();
    std::cout << double(finish - start) / CLOCKS_PER_SEC << " seconds to calculate IBD for " << num_unique * (num_unique + 1) / 2 << " hash pairs." << std::endl;

    return;
}

void inner_loop32(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibx, size_t xibx, uint32_t* pintervals, size_t num_unique, uint32_t* ibx)
{
    uint32_t ibx_for_pair = 0;                                                  // IBD is uint32_t
    uint32_t* genomea = (uint32_t*)(pdata + pids[ygenome] * stride);            // GENOME is uint32_t
    uint32_t* genomeb = (uint32_t*)(pdata + pids[xgenome] * stride);            // GENOME is uint32_t

    size_t num_simd = num_intervals & ~7;
#if __AVX2__
    for ( size_t simd = 0; simd < num_simd; simd += 8 ) {
        __m256i ga    = _mm256_load_si256((const __m256i*)(genomea + simd));    // load 8x uint32_t
        __m256i gb    = _mm256_load_si256((const __m256i*)(genomeb + simd));    // load 8x uint32_t
        __m256i span  = _mm256_load_si256((const __m256i*)(pintervals + simd)); // load matching interval spans
        __m256i eq    = _mm256_cmpeq_epi32(ga, gb);                             // compare each entry equal -> 0xFFFFFFFF, not equal -> 0x00000000
        __m256i tmp   = _mm256_and_si256(eq, span);                             // bitwise and interval span with comparison result
        ibx_for_pair += hsum256_epi32_avx(tmp);                                 // sum resulting values and add to running total
    }
#endif
#if __ARM_NEON__
    for ( size_t simd = 0; simd < num_simd; simd += 4 ) {
        uint32x4_t ga    = vld1q_u32(genomea + simd);                           // load 4x uint32_t
        uint32x4_t gb    = vld1q_u32(genomeb + simd);                           // load 4x uint32_t
        uint32x4_t span  = vld1q_u32(pintervals + simd);                        // load matching interval spans
        uint32x4_t eq    = vceqq_u32(ga, gb);                                   // compare each entry equal -> 0xFFFFFFFF, not equal -> 0x00000000
        uint32x4_t tmp   = vandq_u32(eq, span);                                 // bitwise and interval span with comparison result
        // ibx_for_pair += hsum_neon(tmp);                                         // sum resulting values and add to running total
        ibx_for_pair += vaddvq_u32(tmp);
    }
#endif

    for ( size_t iinterval = num_simd; iinterval < num_intervals; ++iinterval ) {
        if ( genomea[iinterval] == genomeb[iinterval] ) {
            ibx_for_pair += pintervals[iinterval];
        }
    }

    ibx[ yibx * num_unique + xibx ] = ibx[ xibx * num_unique + yibx ] = ibx_for_pair;
}

void inner_loop16(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibx, size_t xibx, uint32_t* pintervals, size_t num_unique, uint32_t* ibx)
{
    uint32_t ibx_for_pair = 0;                                          // IBD is _always_ uint32_t
    uint16_t* genomea = (uint16_t*)(pdata + pids[ygenome] * stride);    // GENOME in _this function_ is uint16_t
    uint16_t* genomeb = (uint16_t*)(pdata + pids[xgenome] * stride);    // GENOME in _this function_ is uint16_t

    size_t num_simd = num_intervals & ~7;                               // We calculate 8 at a time, round num_intervals down to largest multiple of 8
#if __AVX2__
    for ( size_t simd = 0; simd < num_simd; simd += 8 ) {
        __m256i ga    = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)(genomea + simd)));    // load 8x uint16_t, convert to 8x uint32_t
        __m256i gb    = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)(genomeb + simd)));    // load 8x uint16_t, convert to 8x uint32_t
        __m256i span  = _mm256_load_si256((const __m256i*)(pintervals + simd));                     // load matching interval spans
        __m256i eq    = _mm256_cmpeq_epi32(ga, gb);                                                 // compare each entry equal -> 0xFFFFFFFF, not equal -> 0x00000000
        __m256i tmp   = _mm256_and_si256(eq, span);                                                 // bitwise and interval span with comparison result
        ibx_for_pair += hsum256_epi32_avx(tmp);                                                     // sum resulting values and add to running total
    }
#endif
#if __ARM_NEON__
    for ( size_t simd = 0; simd < num_simd; simd += 4 ) {
        uint32x4_t ga    = vmovl_u16(vld1_u16(genomea + simd)); // load 4x uint16_t, convert to 4x uint32_t
        uint32x4_t gb    = vmovl_u16(vld1_u16(genomeb + simd)); // load 4x uint16_t, convert to 4x uint32_t
        uint32x4_t span  = vld1q_u32(pintervals + simd);        // load matching interval spans
        uint32x4_t eq    = vceqq_u32(ga, gb);                   // compare each entry equal -> 0xFFFFFFFF, not equal -> 0x00000000
        uint32x4_t tmp   = vandq_u32(eq, span);                 // bitwise and interval span with comparison result
        // ibx_for_pair += hsum_neon(tmp);                         // sum resulting values and add to running total
        ibx_for_pair += vaddvq_u32(tmp);
    }
#endif

    for ( size_t iinterval = num_simd; iinterval < num_intervals; ++iinterval ) {
        if ( genomea[iinterval] == genomeb[iinterval] ) {
            ibx_for_pair += pintervals[iinterval];
        }
    }

    ibx[ yibx * num_unique + xibx ] = ibx[ xibx * num_unique + yibx ] = ibx_for_pair;

    return;
}

void inner_loop08(uint8_t* pdata, uint32_t* pids, size_t ygenome, size_t stride, size_t xgenome, size_t num_intervals, size_t yibx, size_t xibx, uint32_t* pintervals, size_t num_unique, uint32_t* ibx)
{
#if __AVX2__
    __m256i eightFifteen = _mm256_set_epi8( 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,         // zero all these
                                            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,         // zero all these
                                            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,         // zero all these
                                            0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08);        // shuffle 8..15 -> 0..7

    __m256i twentyFourthirtyOne = _mm256_set_epi8( 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // zero all these
                                                   0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08,  // shuffle 24..31 -> 16..23
                                                   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // zero all these
                                                   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80); // zero all these
#endif

    uint32_t ibx_for_pair = 0;                                                          // IBD is _always_ uint32_t
    uint8_t* genomea = (uint8_t*)(pdata + pids[ygenome] * stride);                      // GENOME in _this function_ is uint8_t
    uint8_t* genomeb = (uint8_t*)(pdata + pids[xgenome] * stride);                      // GENOME in _this function_ is uint8_t

    DPRINTFMT("inner_loop08: pdata = %p, pids = %p, ygenome = %ld, stride = %ld, xgenome = %ld, num_intervals = %ld, yibx = %ld, xibx = %ld\n", pdata, pids, ygenome, stride, xgenome, num_intervals, yibx, xibx);
    DPRINTFMT("inner_loop08: pintervals = %p, num_unique = %ld, ibx = %p\n", pintervals, num_unique, ibx);
    DPRINTFMT("inner_loop08: genomea = %p, genomeb = %p\n", genomea, genomeb);

    size_t num_simd = num_intervals & ~31;
#if __AVX2__
    for ( size_t simd = 0; simd < num_simd; simd += 32 ) {
        __m256i ga    = _mm256_load_si256((const __m256i*)(genomea + simd));            // load 32x uint8_t
        __m256i gb    = _mm256_load_si256((const __m256i*)(genomeb + simd));            // load 32x uint8_t
        __m256i eq    = _mm256_cmpeq_epi8(ga, gb);                                      // compare each entry, equal -> 0xFF, not equal -> 0x00
        __m256i span0 = _mm256_load_si256((const __m256i*)(pintervals + simd));         // load 8x spans
        __m256i span1 = _mm256_load_si256((const __m256i*)(pintervals + simd +  8));    // load 8x spans
        __m256i span2 = _mm256_load_si256((const __m256i*)(pintervals + simd + 16));    // load 8x spans
        __m256i span3 = _mm256_load_si256((const __m256i*)(pintervals + simd + 24));    // load 8x spans

        // values 0..7 to int32_t
        __m256i temp0 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(eq));
                temp0 = _mm256_and_si256(temp0, span0); // bitwise and masks with intervals
        ibx_for_pair += hsum256_epi32_avx(temp0);

        // values 8..15 to int32_t
        __m256i temp1 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(_mm256_shuffle_epi8(eq, eightFifteen)));
                temp1 = _mm256_and_si256(temp1, span1); // bitwise and masks with intervals
        ibx_for_pair += hsum256_epi32_avx(temp1);

        // values 16..23 to int32_t
        __m256i temp2 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(eq, 1));
                temp2 = _mm256_and_si256(temp2, span2); // bitwise and masks with intervals
        ibx_for_pair += hsum256_epi32_avx(temp2);

        // values 24..31 to int32_t
        __m256i temp3 = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(_mm256_shuffle_epi8(eq, twentyFourthirtyOne), 1));
                temp3 = _mm256_and_si256(temp3, span3); // bitwise and masks with intervals
        ibx_for_pair += hsum256_epi32_avx(temp3);
    }
#endif
#if __ARM_NEON__
    for ( size_t simd = 0; simd < num_simd; simd += 16 ) {
        uint8x16_t ga0 = vld1q_u8(genomea + simd);              // load 16x uint8_t
        uint8x16_t gb0 = vld1q_u8(genomeb + simd);              // load 16x uint8_t
        uint32x4_t span0 = vld1q_u32(pintervals + simd);        // load 4x spans
        uint32x4_t span1 = vld1q_u32(pintervals + simd + 4);    // load 4x spans
        uint32x4_t span2 = vld1q_u32(pintervals + simd + 8);    // load 4x spans
        uint32x4_t span3 = vld1q_u32(pintervals + simd + 12);   // load 4x spans
        uint8x16_t eq0 = vceqq_u8(ga0, gb0);                    // compare each entry, equal -> 0xFF, not equal -> 0x00
        // convert first four values of eq0 to uint32_t
        uint32x4_t temp0 = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(eq0))));
                   temp0 = vandq_u32(temp0, span0); // bitwise and masks with intervals
        // ibx_for_pair += hsum_neon(temp0);
        ibx_for_pair += vaddvq_u32(temp0);
        // convert second four values of eq0 to uint32_t
        uint32x4_t temp1 = vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(eq0))));
                   temp1 = vandq_u32(temp1, span1); // bitwise and masks with intervals 
        // ibx_for_pair += hsum_neon(temp1);
        ibx_for_pair += vaddvq_u32(temp1);
        // convert third four values of eq0 to uint32_t
        uint32x4_t temp2 = vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(eq0))));
                   temp2 = vandq_u32(temp2, span2); // bitwise and masks with intervals 
        // ibx_for_pair += hsum_neon(temp2);
        ibx_for_pair += vaddvq_u32(temp2);
        // convert fourth four values of eq0 to uint32_t
        uint32x4_t temp3 = vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(eq0))));
                   temp3 = vandq_u32(temp3, span3); // bitwise and masks with intervals 
        // ibx_for_pair += hsum_neon(temp3);
        ibx_for_pair += vaddvq_u32(temp3);
    }
#endif
    for ( size_t iinterval = num_simd; iinterval < num_intervals; ++iinterval ) {
        if ( genomea[iinterval] == genomeb[iinterval] ) {
            ibx_for_pair += pintervals[iinterval];
        }
    }
    ibx[ yibx * num_unique + xibx ] = ibx[ xibx * num_unique + yibx ] = ibx_for_pair;

    return;
}

/********* Utility functions ********/

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

PyArrayObject* allocate_aligned_1d(size_t num_elements, size_t itemsize, void* indata, int type)
{
    PyArrayObject* aligned = nullptr;
    uint8_t* pdata = nullptr;

    DPRINTFMT("incoming array shape is %ld elements\n", num_elements);

    size_t num_bytes = (size_t)num_elements * itemsize;
    posix_memalign((void**)&pdata, IDM_ALIGNMENT, num_bytes);

    DPRINTFMT("allocated %ld bytes aligned on %d byte boundary: %p\n", num_bytes, IDM_ALIGNMENT, pdata);

    if ( indata ) {
        memcpy(pdata, indata, num_bytes);
    }
    else {
        memset((void*)pdata, 0, num_bytes);
    }

    npy_intp count = num_elements;
    int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA;
    aligned = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, &count, type, /*strides*/nullptr, pdata, static_cast<npy_intp>(itemsize), flags, /*obj*/nullptr);

    DPRINTFMT("DIM[0] =     %d\n", PyArray_DIMS(aligned)[0]);
    DPRINTFMT("FLAGS = .... 0x%08X\n", PyArray_FLAGS(aligned));
    DPRINTFMT("TYPE =       %d\n", PyArray_TYPE(aligned));
    DPRINTFMT("DATA = ..... %p\n", PyArray_DATA(aligned));
    DPRINTFMT("ITEMSIZE =   %ld\n", PyArray_ITEMSIZE(aligned));
    DPRINTFMT("SIZE = ..... %ld\n", PyArray_SIZE(aligned));
    DPRINTFMT("NBYTES =     %ld\n", PyArray_NBYTES(aligned));

    return aligned;
}

PyArrayObject* allocate_aligned_2d(size_t num_rows, size_t num_columns, size_t itemsize, void* indata, size_t instride, int type)
{
    PyArrayObject* aligned = nullptr;
    uint8_t* pdata = nullptr;

    DPRINTFMT("incoming array shape is (%ld, %ld)\n", num_rows, num_columns);

    size_t outstride = (num_columns * itemsize + IDM_ALIGNMENT - 1) & ~(IDM_ALIGNMENT - 1);
    size_t total_bytes = num_rows * outstride;
    posix_memalign((void**)&pdata, IDM_ALIGNMENT, total_bytes);

    DPRINTFMT("allocated %ld bytes aligned on %d byte boundary: %p, stride is %ld bytes\n", total_bytes, IDM_ALIGNMENT, pdata, outstride);

    if ( indata ) {
        for ( size_t row = 0; row < num_rows; ++row ) {
            memcpy(pdata + row * outstride, ((uint8_t*)indata) + row * instride, num_columns * itemsize);
        }
    }
    else {
        memset((void*)pdata, 0, total_bytes);
    }

    npy_intp dimensions[2] = { static_cast<npy_intp>(num_rows), static_cast<npy_intp>(num_columns) };
    npy_intp outstrides[2] = { static_cast<npy_intp>(outstride), static_cast<npy_intp>(itemsize) };
    int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA;
    aligned = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dimensions, type, outstrides, (void*)pdata, static_cast<npy_intp>(itemsize), flags, /*obj*/nullptr);

    DPRINTFMT("NDIM =       %d\n", PyArray_NDIM(aligned));
    for ( int i = 0; i < PyArray_NDIM(aligned); ++i )
        DPRINTFMT("    DIM[%d] = %ld\n", i, PyArray_DIMS(aligned)[i]);
    DPRINTFMT("FLAGS = .... 0x%08X\n", PyArray_FLAGS(aligned));
    DPRINTFMT("TYPE =       %d\n", PyArray_TYPE(aligned));
    DPRINTFMT("DATA = ..... %p\n", PyArray_DATA(aligned));
    DPRINTFMT("ITEMSIZE =   %ld\n", PyArray_ITEMSIZE(aligned));
    DPRINTFMT("SIZE = ..... %ld\n", PyArray_SIZE(aligned));
    DPRINTFMT("NBYTES =     %ld\n", PyArray_NBYTES(aligned));
    for ( int i = 0; i < PyArray_NDIM(aligned); ++i )
        DPRINTFMT("    STRIDE[%d] = %ld\n", i, PyArray_STRIDES(aligned)[i]);

    return aligned;
}

