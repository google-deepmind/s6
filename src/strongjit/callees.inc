// Note no include guard!

// Lists the allowlist of known callees for CallNativeInst.
//
// #include this file and define:
//   #define CALLEE(symbol): act on a symbol name in the global namespace,
//     extern "C".
//   #define CPP_CALLEE(namespace, symbol): act on a C++ symbol name in
//     `namespace`.
//
// If you want to also process extra callee information, you should also define
// CALLEE_INFO, CPP_CALLEE_INFO, INFO, ARGS, NOTRC, STEAL, BORROW,
// NOTNULL and MAYBENULL, but only CalleeInfo::Get should ever do that.
//
// All defines are undefined at the end of this file, so no need for the
// includer to do #undef CALLEE.

#ifndef CALLEE
#error Define CALLEE() before including callees.inc!
#endif
#ifndef CPP_CALLEE
#error Define CPP_CALLEE() before including callees.inc!
#endif

#ifndef CALLEE_INFO
#define CALLEE_INFO(symbol, info) CALLEE(symbol)
#endif
#ifndef CPP_CALLEE_INFO
#define CPP_CALLEE_INFO(namespace, symbol, info) CPP_CALLEE(namespace, symbol)
#endif

#ifndef CINFO
#define CINFO(...)
#endif

#ifndef NEWREF
#define NEWREF
#endif
#ifndef PLAIN
#define PLAIN
#endif

#ifndef MNULL
#define MNULL
#endif
#ifndef NOTNULL
#define NOTNULL
#endif
// Implies NOTNULL
#ifndef STOLEN
#define STOLEN
#endif
#ifndef STOLEN_MNULL
#define STOLEN_MNULL
#endif

// Arguments are NOTNULL by default,
// Return values are MAYBENULL by default.
// NOTRC implies MAYBENULL
// In case of variadic arguments, the information of the last argument is
// repeated. Adding a /*,...*/ is nice for the reader.

// Shortcuts:
#define BIN_INFO CINFO(NEWREF, {NOTNULL, NOTNULL})
#define UN_INFO CINFO(NEWREF, {NOTNULL})

// S6 symbols; keep sorted
CPP_CALLEE_INFO(s6, BuildListUnpackVararg,
                CINFO(NEWREF, {MNULL, MNULL, NOTNULL /*,...*/}))
CPP_CALLEE_INFO(s6, CallFunctionEx, CINFO(NEWREF, {NOTNULL, NOTNULL, MNULL}))
CPP_CALLEE_INFO(s6, CheckedGivenExceptionMatches,
                CINFO(PLAIN, {NOTNULL, NOTNULL}))
CPP_CALLEE_INFO(s6, CreateGenerator,
                CINFO(NEWREF, {NOTNULL, NOTNULL, STOLEN_MNULL /*,...*/}))
CPP_CALLEE_INFO(s6, FormatUnboundError, CINFO(PLAIN, {NOTNULL, MNULL, MNULL}))
CPP_CALLEE_INFO(s6, HandleRaiseVarargs,
                CINFO(PLAIN, {MNULL, STOLEN_MNULL, STOLEN_MNULL}))
CPP_CALLEE_INFO(s6, InitializeObjectDict,
                CINFO(PLAIN, {NOTNULL, NOTNULL, MNULL, NOTNULL}))
CPP_CALLEE_INFO(s6, IteratorNext, UN_INFO)
CPP_CALLEE_INFO(s6, LoadGlobal, CINFO(PLAIN, {NOTNULL, NOTNULL, NOTNULL}))
CPP_CALLEE(s6, RematerializeGetAttr)
CPP_CALLEE_INFO(s6, SetAttrForGlobalsDict,
                CINFO(PLAIN, {NOTNULL, NOTNULL, NOTNULL}))
CPP_CALLEE_INFO(s6, SinDouble, CINFO(PLAIN, {MNULL}))
CPP_CALLEE_INFO(s6, CosDouble, CINFO(PLAIN, {MNULL}))

// Python symbols; keep sorted
CALLEE(_PyCFunction_FastCallKeywords)
CALLEE_INFO(_PyDict_NewPresized, CINFO(NEWREF, {MNULL}))
CALLEE(_PyUnicode_JoinArray)
CALLEE_INFO(PyBool_FromLong, CINFO(NEWREF, {MNULL}))
CALLEE_INFO(PyDict_DelItem, CINFO(PLAIN, {NOTNULL, NOTNULL}))
CALLEE_INFO(PyDict_GetItem, CINFO(NEWREF, {NOTNULL, NOTNULL}))
CALLEE_INFO(PyDict_New, CINFO(NEWREF, {}))
CALLEE_INFO(PyDict_SetItem, CINFO(PLAIN, {NOTNULL, NOTNULL, NOTNULL}))
CALLEE_INFO(PyErr_Clear, CINFO(PLAIN, {}))
CALLEE_INFO(PyErr_ExceptionMatches, CINFO(PLAIN, {MNULL}))
CALLEE_INFO(PyErr_Format, CINFO(PLAIN, {MNULL, NOTNULL, MNULL /*,...*/}))
CALLEE_INFO(PyErr_GivenExceptionMatches, CINFO(PLAIN, {MNULL, MNULL}))
CALLEE_INFO(PyErr_Restore,
            CINFO(PLAIN, {STOLEN_MNULL, STOLEN_MNULL, STOLEN_MNULL}))
CALLEE(PyErr_SetString)
CALLEE_INFO(PyFloat_FromDouble, CINFO(NEWREF, {MNULL}))
CALLEE_INFO(PyFunction_NewWithQualName,
            CINFO(NEWREF, {NOTNULL, NOTNULL, MNULL}))
CALLEE_INFO(PyIter_Next, UN_INFO)
CALLEE_INFO(PyList_Append, CINFO(PLAIN, {NOTNULL, MNULL}))
CALLEE_INFO(PyList_AsTuple, CINFO(NEWREF, {MNULL}))
CALLEE_INFO(PyList_GetItem, CINFO(PLAIN, {NOTNULL, MNULL}))
CALLEE_INFO(PyList_New, CINFO(NEWREF, {MNULL}))
CALLEE_INFO(PyList_Size, CINFO(PLAIN, {NOTNULL}))
CALLEE_INFO(PyLong_FromLong, CINFO(NEWREF, {MNULL}))
// The PyNumber_Xxx callees in particular must remain sorted:
// there are occurrences of `callee >= PyNumber_Add && callee <= PyNumber_Xor`.
CALLEE_INFO(PyNumber_Add, BIN_INFO)
CALLEE_INFO(PyNumber_And, BIN_INFO)
CALLEE_INFO(PyNumber_FloorDivide, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceAdd, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceAnd, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceFloorDivide, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceLshift, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceMatrixMultiply, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceMultiply, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceOr, BIN_INFO)
CALLEE(PyNumber_InPlacePower)
CALLEE_INFO(PyNumber_InPlaceRemainder, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceRshift, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceSubtract, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceTrueDivide, BIN_INFO)
CALLEE_INFO(PyNumber_InPlaceXor, BIN_INFO)
CALLEE_INFO(PyNumber_Invert, UN_INFO)
CALLEE_INFO(PyNumber_Lshift, BIN_INFO)
CALLEE_INFO(PyNumber_MatrixMultiply, BIN_INFO)
CALLEE_INFO(PyNumber_Multiply, BIN_INFO)
CALLEE_INFO(PyNumber_Negative, UN_INFO)
CALLEE_INFO(PyNumber_Or, BIN_INFO)
CALLEE_INFO(PyNumber_Positive, UN_INFO)
CALLEE_INFO(PyNumber_Power, BIN_INFO)
CALLEE_INFO(PyNumber_Remainder, BIN_INFO)
CALLEE_INFO(PyNumber_Rshift, BIN_INFO)
CALLEE_INFO(PyNumber_Subtract, BIN_INFO)
CALLEE_INFO(PyNumber_TrueDivide, BIN_INFO)
CALLEE_INFO(PyNumber_Xor, BIN_INFO)
CALLEE_INFO(PyObject_DelItem, CINFO(PLAIN, {MNULL, MNULL}))
CALLEE_INFO(PyObject_GetAttr, BIN_INFO)
CALLEE_INFO(PyObject_GetItem, BIN_INFO)
CALLEE_INFO(PyObject_GetIter, UN_INFO)
CALLEE(PyObject_ASCII)
CALLEE(PyObject_Format)
CALLEE_INFO(PyObject_IsTrue, CINFO(PLAIN, {NOTNULL}))
CALLEE(PyObject_Repr)
CALLEE_INFO(PyObject_RichCompare, CINFO(NEWREF, {MNULL, MNULL, MNULL}))
CALLEE_INFO(PyObject_SetAttr, CINFO(PLAIN, {NOTNULL, NOTNULL, MNULL}))
CALLEE_INFO(PyObject_SetItem, CINFO(PLAIN, {MNULL, MNULL, MNULL}))
CALLEE(PyObject_Str)
CALLEE_INFO(PySequence_Contains, CINFO(PLAIN, {NOTNULL, MNULL}))
CALLEE_INFO(PySet_Add, CINFO(PLAIN, {NOTNULL, NOTNULL}))
CALLEE_INFO(PySet_New, CINFO(NEWREF, {MNULL}))
CALLEE_INFO(PySlice_New, CINFO(NEWREF, {MNULL, MNULL, MNULL}))
CALLEE(PyTuple_GetItem)
// Assume no allocation error, so this can't fail. Similar to BoxInst.
CALLEE_INFO(PyTuple_New, CINFO(NOTNULL, NEWREF, {MNULL}))
CALLEE_INFO(PyTuple_Size, CINFO(PLAIN, {NOTNULL}))
CALLEE(PyType_IsSubtype)
CALLEE(PyUnicode_AsUTF8)
CALLEE(PyUnicode_New)

#undef CALLEE
#undef CPP_CALLEE
#undef CALLEE_INFO
#undef CPP_CALLEE_INFO
#undef CINFO
#undef NOTNULL
#undef MNULL
#undef PLAIN
#undef NEWREF
#undef STOLEN
#undef STOLEN_MNULL
