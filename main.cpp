#include <iostream>
#include <python3.6/Python.h>
int main(int, char**)
{
     //设定参数值
    int a = 0;
    int b = 6;
 
    //第一步：初始化python环境
    Py_Initialize();
    //PyRun_SimpleString（）执行命令语句
    //测试python3的打印语句
    PyRun_SimpleString("print('Hello Python!')\n");
    PyRun_SimpleString("import os,sys");//执行import语句，把当前路径加入路径中，为了找到math_test.py
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("print(os.getcwd())");//测试打印当前路径
    
    PyObject *pModule;
    PyObject *pFunction;
    PyObject *pArgs;
    PyObject *pRetValue;
    //第二步：调用math_test脚本
    pModule = PyImport_ImportModule("math_test");
    if (!pModule) {
        printf("import python failed1!!\n");
        return -1;
    }
    //第三步：查找对应math_test.py中的def mainfunc(a,b)函数
    pFunction = PyObject_GetAttrString(pModule, "mainfunc");
    if (!pFunction) {
        printf("get python function failed!!!\n");
        return -1;
    }
    //第四步：新建python中的tuple对象（构建参数）
    pArgs = PyTuple_New(2);
    //PyTuple_SetItem(pArgs, 0, Py_BuildValue(""));
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", a));
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", b));
    //第五步：调用函数
    pRetValue=PyObject_CallObject(pFunction, pArgs);
 
    //第六步：清空PyObject 
    Py_DECREF(pModule);
    Py_DECREF(pFunction);
    Py_DECREF(pArgs);
    Py_DECREF(pRetValue);
 
    if (!pModule) {
        printf("import python failed!!\n");
        return -1;
    }
    //第七步：很关键的一步，如果是多次导入PyImport_ImportModule模块
    //只有最后一步才调用Py_Finalize()，此时python全局变量一直保存着
    Py_Finalize();
    //方便查看
    while (1);
 
    return 0;
} 
