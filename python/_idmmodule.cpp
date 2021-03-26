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
// static PyObject *IdmException;

/*===================================================================
 * Module level functions
 *===================================================================
 */

static PyObject *
get_genomes(PyObject *self, PyObject *arg)
{
    PyObject *ret = idm_get_genomes(arg);

    return ret;
}

static PyObject *
calculate_ibx(PyObject *self, PyObject *args)
{
    PyObject *ret = idm_calculate_ibx(args);

    return ret;
}

static PyMethodDef idm_methods[] = {
    { .ml_name = "get_genomes",
        .ml_meth = (PyCFunction) get_genomes,
        .ml_flags = METH_O,
        .ml_doc = "Returns genomes (root of each interval) for a tree sequence as a NumPy array." },
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

    /* Errors and constants */
    // IdmException = PyErr_NewException("_idm.IdmException", NULL, NULL);
    // Py_INCREF(IdmException);
    // PyModule_AddObject(module, "IdmException", IdmException);

    // PyModule_AddIntConstant(module, "NULL", 0);
    // PyModule_AddIntConstant(module, "MISSING_DATA", 13);

    return module;
}

} // extern "C"
