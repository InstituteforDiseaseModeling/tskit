#include "idmextensions.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL IDM_ARRAY_API
#include <numpy/arrayobject.h>

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "sha256.h"

#include <emmintrin.h>
#include <immintrin.h>

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

#define IDM_ALIGNMENT   32

template<typename T>
static void
_traverse(tsk_tree_t *tree, tsk_id_t node, tsk_id_t root, uint8_t* genome, size_t stride, size_t interval /*, int depth*/)
{
    // Visit node - store root at current interval for this node
    T* pibd = (T*)(genome + node * stride) + interval;
    *pibd = root;

    // Visit node's children
    for (tsk_id_t child = tree->left_child[node]; child != TSK_NULL; child = tree->right_sib[child]) {
        _traverse<T>(tree, child, root, genome, stride, interval /*, depth + 1 */);
    }
}

template<typename T>
static void
traverse_recursive(tsk_tree_t *tree, uint8_t* genome, size_t stride, size_t interval)
{
    // For each unique root, visit root (and it's children)
    for (tsk_id_t root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        _traverse<T>(tree, root, root, genome, stride, interval /*, 0*/);
    }
}

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

PyArrayObject *allocate_array(tsk_treeseq_t *tree_sequence);

PyObject *idm_get_genomes(tsk_treeseq_t *tree_sequence)
{
    tsk_tree_t tree;
    tsk_tree_init(&tree, tree_sequence, 0);
    PyArrayObject *array = allocate_array(tree_sequence);

    int iter = 0;
    size_t current_interval = 0;
    size_t elementsize = PyArray_ITEMSIZE(array);
    uint8_t *pdata = (uint8_t*)PyArray_DATA(array);
    size_t stride = PyArray_STRIDES(array)[0];
    // size_t percent = 0;
    // clock_t start = clock();
    for (iter = tsk_tree_first(&tree), current_interval = 0; iter == 1; iter = tsk_tree_next(&tree), ++current_interval) {
        // progress(percent, current_interval, num_trees, start);
        if ( elementsize == sizeof(uint16_t) ) {
            traverse_recursive<uint16_t>(&tree, pdata, stride, current_interval);
        } else {
            traverse_recursive<uint32_t>(&tree, pdata, stride, current_interval);
        }
    }

    Py_XDECREF(array);

    return (PyObject *)array;
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
    npy_intp strides[2] = { static_cast<npy_intp>(stride), static_cast<npy_intp>(elementsize) };

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

void calculate_ibd08(PyArrayObject *genomes, PyArrayObject *ids, PyArrayObject *intervals, PyArrayObject **ndibx, PyObject **digests, PyObject **mapping);
void calculate_ibx16(PyArrayObject *genomes, PyArrayObject *ids, PyArrayObject *intervals, PyArrayObject **ndibx, PyObject **digests, PyObject **mapping);
void calculate_ibd32(PyArrayObject *genomes, PyArrayObject *ids, PyArrayObject *intervals, PyArrayObject **ndibx, PyObject **digests, PyObject **mapping);

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
        if ( (PyObject *)ndvectorids == Py_None ) {
            printf("ndvectorids == Py_None, setting to NULL\n");
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
            printf("ndvectorids passed validation, incrementing refcount (ndvectorids[1] == %d)\n", ((uint32_t*)PyArray_DATA(ndvectorids))[1]);
            Py_XINCREF(ndvectorids);
        } else {
            npy_intp length = PyArray_DIMS(ndarraygenomes)[0];
            ndvectorids = (PyArrayObject *)PyArray_ZEROS(1, &length, NPY_UINT32, false);
            uint32_t *pid = (uint32_t *)PyArray_DATA(ndvectorids);
            for (uint32_t id = 0; id < length; ++id) {
                *pid++ = id;
            }
            printf("ndvectorids was NULL, allocated 'identity' vector (%d)\n", ((uint32_t *)PyArray_DATA(ndvectorids))[8]);
        }

        // ndvectorintervals is None or one dimension (vector) of uint32_t, == #columns of ndarraygenomes
        if ( (PyObject *)ndvectorintervals == Py_None ) {
            printf("ndvectorintervals == Py_None, setting to NULL\n");
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
            printf("ndvectorintervals passed validation, incrementing refcount (ndvectorintervals[1] == %d)\n", ((uint32_t*)PyArray_DATA(ndvectorintervals))[1]);
            Py_XINCREF(ndvectorintervals);
        } else {
            npy_intp length = PyArray_DIMS(ndarraygenomes)[1];
            ndvectorintervals = (PyArrayObject *)PyArray_ZEROS(1, &length, NPY_UINT32, false);
            uint32_t *pinterval = (uint32_t *)PyArray_DATA(ndvectorintervals);
            for (uint32_t i = 0; i < length; ++i) {
                *pinterval++ = 1;
            }
            printf("ndvectorintervals was NULL, allocated 'ones' vector (%d)\n", ((uint32_t *)PyArray_DATA(ndvectorintervals))[8]);
        }
    }

    switch (PyArray_ITEMSIZE(ndarraygenomes)) {
        case 1:
            //calculate_ibd08();
            break;

        case 2:
            calculate_ibx16(ndarraygenomes, ndvectorids, ndvectorintervals, &ibx, &hashes, &mapping);
            break;

        case 4:
            //calculate_ibd32();
            break;

        default:
            break;
    }

out:
    // NxN calculated values, unique hash values, entry->hash index map
    return Py_BuildValue("OOO", ibx, hashes, mapping);
}

void calculate_ibd08(PyArrayObject *genomes, PyArrayObject *ids, PyArrayObject *intervals, PyArrayObject **ndibx, PyObject **digests, PyObject **mapping)
{
    return;
}

void calculate_ibx16(PyArrayObject *ndgenomes, PyArrayObject *ndids, PyArrayObject *ndintervals, PyArrayObject **ndibx, PyObject **digests, PyObject **mapping)
{
    printf("In calculate_ibx16()...\n");
    size_t num_rows = PyArray_DIMS(ndids ? ndids : ndgenomes)[0];               printf("num_rows = %ld\n", num_rows);
    size_t num_intervals = PyArray_DIMS(ndgenomes)[1];                          printf("num_intervals = %ld\n", num_intervals);

    // Calculate SHA-256 hash for each genome by id in ndids

    uint32_t* pids = (uint32_t*)PyArray_DATA(ndids);
    uint8_t* pdata = (uint8_t*)PyArray_DATA(ndgenomes);
    size_t stride  = PyArray_STRIDES(ndgenomes)[0];                             printf("stride = %ld\n", stride);
    size_t count   = PyArray_DIMS(ndgenomes)[1] * PyArray_ITEMSIZE(ndgenomes);  printf("count = %ld\n", count);
    std::vector<std::string> hashes;

    clock_t start = clock();
    for ( size_t idindex = 0; idindex < num_rows; ++idindex ) {
        size_t row = pids[idindex];
        hashes.push_back(sha256(pdata + row * stride, count));
    }
    clock_t finish = clock();

    std::cout << double(finish - start) / CLOCKS_PER_SEC << " seconds to calculate SHA-256 hashes for " << num_rows << " genomes." << std::endl;

    // Create a Python list of hashes. Implicitly, list[N] == hash_value for ndids[N]
    *digests = nullptr;
    *digests = PyList_New(hashes.size());
    for ( size_t index = 0; index < hashes.size(); ++index) {
        PyObject *string = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, hashes[index].c_str(), hashes[index].size());
        /* int ret = */ PyList_SetItem(*digests, index, string);
    }

    // Determine unique hash values and sort them.
    std::set<std::string> unique(hashes.begin(), hashes.end());
    std::vector<std::string> sorted(unique.begin(), unique.end());
    std::sort(sorted.begin(), sorted.end());

    size_t num_unique = unique.size();
    std::cout << num_unique << " unique hash values" << std::endl;

    // Map each hash value to its position in the sorted list of unique hash values.
    std::map<std::string, size_t> index_for_hash;
    for ( size_t i = 0; i < sorted.size(); ++i ) {
        index_for_hash[sorted[i]] = i;
    }

    // Create a Python map of each hash value to its position in the sorted list.
    *mapping = nullptr;
    *mapping = PyDict_New();
    for ( auto& entry : index_for_hash ) {
        PyObject* key = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, entry.first.c_str(), entry.first.size());
        PyObject* value =  PyLong_FromUnsignedLong( entry.second );
        /* int ret = */ PyDict_SetItem(*mapping, key, value);
    }

    // Figure out how many genomes fit in the CPU cache.
    size_t genome_size = num_intervals * sizeof(uint16_t);  // GENOME is uint16_t
    std::cout << "genome_size is " << genome_size << " bytes." << std::endl;
    size_t cache_size = 19.25 * 1024 * 1024;    // 19.25MB L3 cache on Intel(R) Xeon(R) Gold 6126 CPU
    size_t num_genomes = cache_size / genome_size;
    std::cout << "num_genomes in 19.25MB cache is " << num_genomes << std::endl;
    size_t block_size = size_t(sqrt(num_genomes));
    std::cout << "block_size is " << block_size << " square." << std::endl;
    size_t num_blocks = (num_rows + block_size - 1) / block_size;
    std::cout << "num_blocks is " << num_blocks << std::endl;

    npy_intp dimensions[2] = { npy_intp(num_unique), npy_intp(num_unique) };
    *ndibx = (PyArrayObject *)PyArray_ZEROS(2, dimensions, NPY_UINT32, false);

    uint32_t* ibx = (uint32_t*)PyArray_DATA(*ndibx);
    memset((void*)ibx, 0xFF, size_t(PyArray_NBYTES(*ndibx)));

    uint32_t* pintervals = (uint32_t*)PyArray_DATA(ndintervals);

    start = clock();
    size_t percent = 0;

    for ( size_t yblock = 0; yblock < num_blocks; ++yblock ) {                      // We are processing banks of rows, each bank of size block_size. There are num_blocks of these banks.
        progress(percent, yblock, num_blocks, start);

        for ( size_t xblock = yblock; xblock < num_blocks; ++xblock ) {             // We are processing bundles of columns, each bundle of size block_size, there are num_blocks of these bundles.

            size_t ystart = yblock * block_size;                                    // Starting y for this bank is current block * block size
            size_t ystop = std::min(ystart + block_size, num_rows);                 // Ending y for this bank is smaller of next block or num_rows

            for ( size_t ygenome = ystart; ygenome < ystop; ++ygenome ) {           // Iterate from starting y to ending y

                size_t yibx = index_for_hash[ hashes[ ygenome ] ];                  // Index in result is index_for_hash[ hashes[ id index (ygenome) ] ]
                size_t xstart = xblock * block_size;                                // Starting x for this bundle is current block * block size
                size_t xstop = std::min(xstart + block_size, num_rows);             // Ending x for this bundle is smaller of nextg block or num_rows

                for ( size_t xgenome = xstart; xgenome < xstop; ++xgenome ) {       // Iterate from starting x to ending x

                    size_t xibx = index_for_hash[ hashes[ xgenome ] ];              // Index in result is index_for_hash[ hashes[ id index (xgenome) ]]

                    if ( ibx[ yibx * num_unique + xibx ] == 0xFFFFFFFF ) {          // If the value for this particular pair of hashes has not already been calculated...

                        uint32_t ibx_for_pair = 0;                                          // IBD is _always_ uint32_t
                        uint16_t* genomea = (uint16_t*)(pdata + pids[ygenome] * stride);    // GENOME in _this function_ is uint16_t (c.f. calculate_ibx32)
                        uint16_t* genomeb = (uint16_t*)(pdata + pids[xgenome] * stride);    // GENOME in _this function_ is uint16_t (c.f. calculate_ibx32)
                        size_t num_simd = num_intervals & ~7;                               // We calculate 8 at a time, round num_intervals down to largest multiple of 8
                        for ( size_t simd = 0; simd < num_simd; simd += 8 ) {
                            __m256i ga    = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)(genomea + simd)));    // load 8x uint16_t, convert to 8x uint32_t
                            __m256i gb    = _mm256_cvtepi16_epi32(_mm_load_si128((const __m128i*)(genomeb + simd)));    // load 8x uint16_t, convert to 8x uint32_t
                            __m256i span  = _mm256_load_si256((const __m256i*)(pintervals + simd));                     // load matching interval spans
                            __m256i eq    = _mm256_cmpeq_epi32(ga, gb);                                                 // compare each entry equal -> 0xFFFFFFFF, not equal -> 0x00000000
                            __m256i tmp   = _mm256_and_si256(eq, span);                                                 // bitwise and interval span with comparison result
                            ibx_for_pair += hsum256_epi32_avx(tmp);                                                     // sum resulting values and add to running total
                        }
                        for ( size_t iinterval = num_simd; iinterval < num_intervals; ++iinterval ) {
                            if ( genomea[iinterval] == genomeb[iinterval] ) {
                                ibx_for_pair += pintervals[iinterval];
                            }
                        }

                        ibx[ yibx * num_unique + xibx ] = ibx[ xibx * num_unique + yibx ] = ibx_for_pair;
                    }
                }
            }
        }
    }
    finish = clock();
    std::cout << double(finish - start) / CLOCKS_PER_SEC << " seconds to calculate IBx for all hash pairs." << std::endl;

out:
    return;
}

void calculate_ibd32(PyArrayObject *genomes, PyArrayObject *ids, PyArrayObject *intervals, PyArrayObject **ndibx, PyObject **digests, PyObject **mapping)
{
    return;
}
