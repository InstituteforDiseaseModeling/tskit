/*
 * License
 *
 * Copyright (c) 2021
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define IDM_BUG_ASSERT_MESSAGE                                                          \
    "Please open an issue on"                                                           \
    " GitHub, ideally with a reproducible example."                                     \
    " (https://github.com/institutefordiseasemodeling/tskit/issues)"

#include <Python.h>
#include <structmember.h>
#define PY_ARRAY_UNIQUE_SYMBOL IDM_ARRAY_API
#include <numpy/arrayobject.h>
#include <float.h>

#include "kastore.h"
#include "tskit.h"

#include "idmextensions.h"

#define SET_COLS 0
#define APPEND_COLS 1

/* IdmException is the superclass of all exceptions that can be thrown by
 * idm. We define it here in the low-level library so that exceptions defined
 * here and in the high-level library can inherit from it.
 */
static PyObject *IdmException;

typedef struct {
    PyObject_HEAD
    tsk_treeseq_t *tree_sequence;
} TreeSequence;

/*===================================================================
 * TreeSequence
 *===================================================================
 */

static int
TreeSequence_check_state(TreeSequence *self)
{
    int ret = 0;
    if (self->tree_sequence == NULL) {
        PyErr_SetString(PyExc_ValueError, "tree_sequence not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeSequence_dealloc(TreeSequence *self)
{
    if (self->tree_sequence != NULL) {
        tsk_treeseq_free(self->tree_sequence);
        PyMem_Free(self->tree_sequence);
        self->tree_sequence = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
TreeSequence_init(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    self->tree_sequence = NULL;
    return 0;
}

static PyObject *
TreeSequence_get_genomes(TreeSequence *self)
{
    PyObject *ret = NULL;

    if (TreeSequence_check_state(self) != 0) {
        goto out;
    }

    ret = idm_get_genomes(self->tree_sequence);

out:
    return ret;
}

static PyMethodDef TreeSequence_methods[] = {
    { .ml_name = "get_genomes",
        .ml_meth = (PyCFunction) TreeSequence_get_genomes,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the individual genomes of the tree as a NumPy array." },
    { NULL } /* Sentinel */
};

static PyTypeObject TreeSequenceType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_idm.TreeSequence",
    .tp_basicsize = sizeof(TreeSequence),
    .tp_dealloc = (destructor) TreeSequence_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "TreeSequence objects",
    .tp_methods = TreeSequence_methods,
    .tp_init = (initproc) TreeSequence_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Module level functions
 *===================================================================
 */

static PyObject *
calculate_ibx(PyObject *self, PyObject *args)
{
    PyObject *ret = NULL;

    ret = idm_calculate_ibx(args);

    return ret;
}

static PyMethodDef idm_methods[] = {
    { .ml_name = "calculate_ibx",
        .ml_meth = (PyCFunction) calculate_ibx,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Calculates IBD/IBS for a set of genomes." },
    { NULL } /* Sentinel */
};

static struct PyModuleDef idmmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_idm",
    .m_doc = "Low level interface for idm",
    .m_size = -1,
    .m_methods = idm_methods,
};

extern "C" {

PyObject *
PyInit__idm(void)
{
    PyObject *module = PyModule_Create(&idmmodule);
    if (module == NULL) {
        return NULL;
    }
    import_array();

    /* TreeSequence type */
    if (PyType_Ready(&TreeSequenceType) < 0) {
        return NULL;
    }
    Py_INCREF(&TreeSequenceType);
    PyModule_AddObject(module, "TreeSequence", (PyObject *) &TreeSequenceType);

    /* Errors and constants */
    IdmException = PyErr_NewException("_idm.IdmException", NULL, NULL);
    Py_INCREF(IdmException);
    PyModule_AddObject(module, "IdmException", IdmException);

    // PyModule_AddIntConstant(module, "NULL", 0);
    // PyModule_AddIntConstant(module, "MISSING_DATA", 13);

    return module;
}

} // extern "C"
