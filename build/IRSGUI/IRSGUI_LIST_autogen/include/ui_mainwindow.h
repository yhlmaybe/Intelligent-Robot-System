/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QPushButton *SetServoNo_Button;
    QLabel *ServoNo_Lable;
    QLineEdit *ServoNo_LineEdit;
    QPushButton *ROSNodeInitiate_Button;
    QPlainTextEdit *MessageText;
    QPushButton *GetServoNo_Button;
    QPushButton *ServesInitiate_Button;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(400, 300);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        SetServoNo_Button = new QPushButton(centralWidget);
        SetServoNo_Button->setObjectName(QString::fromUtf8("SetServoNo_Button"));
        SetServoNo_Button->setGeometry(QRect(10, 30, 89, 25));
        ServoNo_Lable = new QLabel(centralWidget);
        ServoNo_Lable->setObjectName(QString::fromUtf8("ServoNo_Lable"));
        ServoNo_Lable->setGeometry(QRect(110, 80, 71, 21));
        ServoNo_LineEdit = new QLineEdit(centralWidget);
        ServoNo_LineEdit->setObjectName(QString::fromUtf8("ServoNo_LineEdit"));
        ServoNo_LineEdit->setGeometry(QRect(110, 30, 71, 25));
        ROSNodeInitiate_Button = new QPushButton(centralWidget);
        ROSNodeInitiate_Button->setObjectName(QString::fromUtf8("ROSNodeInitiate_Button"));
        ROSNodeInitiate_Button->setGeometry(QRect(260, 70, 121, 25));
        MessageText = new QPlainTextEdit(centralWidget);
        MessageText->setObjectName(QString::fromUtf8("MessageText"));
        MessageText->setGeometry(QRect(10, 120, 381, 111));
        MessageText->setReadOnly(true);
        GetServoNo_Button = new QPushButton(centralWidget);
        GetServoNo_Button->setObjectName(QString::fromUtf8("GetServoNo_Button"));
        GetServoNo_Button->setGeometry(QRect(10, 80, 89, 25));
        ServesInitiate_Button = new QPushButton(centralWidget);
        ServesInitiate_Button->setObjectName(QString::fromUtf8("ServesInitiate_Button"));
        ServesInitiate_Button->setGeometry(QRect(260, 30, 121, 25));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 400, 22));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        SetServoNo_Button->setText(QCoreApplication::translate("MainWindow", "SetServoNo", nullptr));
        ServoNo_Lable->setText(QCoreApplication::translate("MainWindow", "null", nullptr));
        ROSNodeInitiate_Button->setText(QCoreApplication::translate("MainWindow", "ROSNodeInitiate", nullptr));
        GetServoNo_Button->setText(QCoreApplication::translate("MainWindow", "GetServoNo", nullptr));
        ServesInitiate_Button->setText(QCoreApplication::translate("MainWindow", "ServesInitiate", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
