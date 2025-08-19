/****************************************************************************
** Meta object code from reading C++ file 'BrainDeepLearnForm.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "BrainDeepLearnForm.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'BrainDeepLearnForm.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_BrainDeepLearnForm_t {
    QByteArrayData data[10];
    char stringdata0[136];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_BrainDeepLearnForm_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_BrainDeepLearnForm_t qt_meta_stringdata_BrainDeepLearnForm = {
    {
QT_MOC_LITERAL(0, 0, 18), // "BrainDeepLearnForm"
QT_MOC_LITERAL(1, 19, 7), // "AddData"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 4), // "data"
QT_MOC_LITERAL(4, 33, 9), // "CloseForm"
QT_MOC_LITERAL(5, 43, 20), // "TestPerceptionModule"
QT_MOC_LITERAL(6, 64, 19), // "TestAttentionModule"
QT_MOC_LITERAL(7, 84, 16), // "TestMemoryModule"
QT_MOC_LITERAL(8, 101, 18), // "TestDecisionModule"
QT_MOC_LITERAL(9, 120, 15) // "TestWorldModule"

    },
    "BrainDeepLearnForm\0AddData\0\0data\0"
    "CloseForm\0TestPerceptionModule\0"
    "TestAttentionModule\0TestMemoryModule\0"
    "TestDecisionModule\0TestWorldModule"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_BrainDeepLearnForm[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   49,    2, 0x0a /* Public */,
       4,    0,   52,    2, 0x08 /* Private */,
       5,    0,   53,    2, 0x08 /* Private */,
       6,    0,   54,    2, 0x08 /* Private */,
       7,    0,   55,    2, 0x08 /* Private */,
       8,    0,   56,    2, 0x08 /* Private */,
       9,    0,   57,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void BrainDeepLearnForm::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<BrainDeepLearnForm *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->AddData((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->CloseForm(); break;
        case 2: _t->TestPerceptionModule(); break;
        case 3: _t->TestAttentionModule(); break;
        case 4: _t->TestMemoryModule(); break;
        case 5: _t->TestDecisionModule(); break;
        case 6: _t->TestWorldModule(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject BrainDeepLearnForm::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_BrainDeepLearnForm.data,
    qt_meta_data_BrainDeepLearnForm,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *BrainDeepLearnForm::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *BrainDeepLearnForm::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_BrainDeepLearnForm.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int BrainDeepLearnForm::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
