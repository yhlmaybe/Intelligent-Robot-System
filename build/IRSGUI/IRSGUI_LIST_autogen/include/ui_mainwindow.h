/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
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
    QPushButton *Initiate_Button;
    QPlainTextEdit *MessageText;
    QPushButton *GetServoNo_Button;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(400, 300);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        SetServoNo_Button = new QPushButton(centralWidget);
        SetServoNo_Button->setObjectName(QStringLiteral("SetServoNo_Button"));
        SetServoNo_Button->setGeometry(QRect(10, 30, 89, 25));
        ServoNo_Lable = new QLabel(centralWidget);
        ServoNo_Lable->setObjectName(QStringLiteral("ServoNo_Lable"));
        ServoNo_Lable->setGeometry(QRect(110, 80, 71, 21));
        ServoNo_LineEdit = new QLineEdit(centralWidget);
        ServoNo_LineEdit->setObjectName(QStringLiteral("ServoNo_LineEdit"));
        ServoNo_LineEdit->setGeometry(QRect(110, 30, 71, 25));
        Initiate_Button = new QPushButton(centralWidget);
        Initiate_Button->setObjectName(QStringLiteral("Initiate_Button"));
        Initiate_Button->setGeometry(QRect(290, 30, 89, 25));
        MessageText = new QPlainTextEdit(centralWidget);
        MessageText->setObjectName(QStringLiteral("MessageText"));
        MessageText->setGeometry(QRect(10, 120, 381, 111));
        MessageText->setReadOnly(true);
        GetServoNo_Button = new QPushButton(centralWidget);
        GetServoNo_Button->setObjectName(QStringLiteral("GetServoNo_Button"));
        GetServoNo_Button->setGeometry(QRect(10, 80, 89, 25));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 400, 22));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        SetServoNo_Button->setText(QApplication::translate("MainWindow", "SetServoNo", Q_NULLPTR));
        ServoNo_Lable->setText(QApplication::translate("MainWindow", "null", Q_NULLPTR));
        Initiate_Button->setText(QApplication::translate("MainWindow", "Initiate", Q_NULLPTR));
        GetServoNo_Button->setText(QApplication::translate("MainWindow", "GetServoNo", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
