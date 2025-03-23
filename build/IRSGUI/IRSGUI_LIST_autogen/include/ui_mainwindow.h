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
#include <QtWidgets/QComboBox>
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
    QPushButton *StateReset_Button;
    QPlainTextEdit *MessageText;
    QPushButton *GetServoNo_Button;
    QPushButton *Initiate_Button;
    QComboBox *jointName_comboBox;
    QLabel *jointName_label;
    QPushButton *setJointPosition_Button;
    QLabel *jointPosition_label;
    QLineEdit *jointPosition_lineEdit;
    QPushButton *setGoalPoint_Button;
    QLabel *goalPoint_label;
    QLineEdit *goalPoint_lineEdit;
    QPushButton *start_Button;
    QPushButton *end_Button;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(776, 489);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        SetServoNo_Button = new QPushButton(centralWidget);
        SetServoNo_Button->setObjectName(QString::fromUtf8("SetServoNo_Button"));
        SetServoNo_Button->setGeometry(QRect(680, 400, 91, 25));
        ServoNo_Lable = new QLabel(centralWidget);
        ServoNo_Lable->setObjectName(QString::fromUtf8("ServoNo_Lable"));
        ServoNo_Lable->setGeometry(QRect(580, 360, 91, 21));
        ServoNo_Lable->setLayoutDirection(Qt::LeftToRight);
        ServoNo_Lable->setAlignment(Qt::AlignCenter);
        ServoNo_LineEdit = new QLineEdit(centralWidget);
        ServoNo_LineEdit->setObjectName(QString::fromUtf8("ServoNo_LineEdit"));
        ServoNo_LineEdit->setGeometry(QRect(580, 400, 91, 25));
        StateReset_Button = new QPushButton(centralWidget);
        StateReset_Button->setObjectName(QString::fromUtf8("StateReset_Button"));
        StateReset_Button->setGeometry(QRect(580, 50, 91, 25));
        MessageText = new QPlainTextEdit(centralWidget);
        MessageText->setObjectName(QString::fromUtf8("MessageText"));
        MessageText->setGeometry(QRect(10, 10, 561, 421));
        MessageText->setReadOnly(true);
        GetServoNo_Button = new QPushButton(centralWidget);
        GetServoNo_Button->setObjectName(QString::fromUtf8("GetServoNo_Button"));
        GetServoNo_Button->setGeometry(QRect(680, 360, 91, 25));
        Initiate_Button = new QPushButton(centralWidget);
        Initiate_Button->setObjectName(QString::fromUtf8("Initiate_Button"));
        Initiate_Button->setGeometry(QRect(580, 10, 91, 25));
        jointName_comboBox = new QComboBox(centralWidget);
        jointName_comboBox->setObjectName(QString::fromUtf8("jointName_comboBox"));
        jointName_comboBox->setGeometry(QRect(580, 110, 191, 25));
        jointName_label = new QLabel(centralWidget);
        jointName_label->setObjectName(QString::fromUtf8("jointName_label"));
        jointName_label->setGeometry(QRect(580, 90, 191, 20));
        jointName_label->setAlignment(Qt::AlignCenter);
        setJointPosition_Button = new QPushButton(centralWidget);
        setJointPosition_Button->setObjectName(QString::fromUtf8("setJointPosition_Button"));
        setJointPosition_Button->setGeometry(QRect(580, 200, 191, 25));
        jointPosition_label = new QLabel(centralWidget);
        jointPosition_label->setObjectName(QString::fromUtf8("jointPosition_label"));
        jointPosition_label->setGeometry(QRect(580, 140, 191, 20));
        jointPosition_label->setAlignment(Qt::AlignCenter);
        jointPosition_lineEdit = new QLineEdit(centralWidget);
        jointPosition_lineEdit->setObjectName(QString::fromUtf8("jointPosition_lineEdit"));
        jointPosition_lineEdit->setGeometry(QRect(580, 160, 191, 25));
        setGoalPoint_Button = new QPushButton(centralWidget);
        setGoalPoint_Button->setObjectName(QString::fromUtf8("setGoalPoint_Button"));
        setGoalPoint_Button->setGeometry(QRect(580, 300, 191, 25));
        goalPoint_label = new QLabel(centralWidget);
        goalPoint_label->setObjectName(QString::fromUtf8("goalPoint_label"));
        goalPoint_label->setGeometry(QRect(580, 240, 191, 20));
        goalPoint_label->setAlignment(Qt::AlignCenter);
        goalPoint_lineEdit = new QLineEdit(centralWidget);
        goalPoint_lineEdit->setObjectName(QString::fromUtf8("goalPoint_lineEdit"));
        goalPoint_lineEdit->setGeometry(QRect(580, 260, 191, 25));
        start_Button = new QPushButton(centralWidget);
        start_Button->setObjectName(QString::fromUtf8("start_Button"));
        start_Button->setGeometry(QRect(680, 10, 91, 25));
        end_Button = new QPushButton(centralWidget);
        end_Button->setObjectName(QString::fromUtf8("end_Button"));
        end_Button->setGeometry(QRect(680, 50, 91, 25));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 776, 22));
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
        StateReset_Button->setText(QCoreApplication::translate("MainWindow", "StateReset", nullptr));
        GetServoNo_Button->setText(QCoreApplication::translate("MainWindow", "GetServoNo", nullptr));
        Initiate_Button->setText(QCoreApplication::translate("MainWindow", "Initiate", nullptr));
        jointName_label->setText(QCoreApplication::translate("MainWindow", "Joint Name", nullptr));
        setJointPosition_Button->setText(QCoreApplication::translate("MainWindow", "Set Joint Position", nullptr));
        jointPosition_label->setText(QCoreApplication::translate("MainWindow", "Joint Position", nullptr));
        setGoalPoint_Button->setText(QCoreApplication::translate("MainWindow", "Set Goal Point", nullptr));
        goalPoint_label->setText(QCoreApplication::translate("MainWindow", "Goal Point", nullptr));
        start_Button->setText(QCoreApplication::translate("MainWindow", "Start", nullptr));
        end_Button->setText(QCoreApplication::translate("MainWindow", "End", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
