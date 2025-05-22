/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionJoint_Datas;
    QAction *actionEnd_Effector_Datas;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QPushButton *start_Button;
    QPushButton *SetServoNo_Button;
    QLabel *jointName_label;
    QLineEdit *jointPosition_lineEdit;
    QPushButton *setGoalPoint_Button;
    QPushButton *end_Button;
    QPushButton *setJointPosition_Button;
    QLabel *goalPoint_label;
    QPushButton *StateReset_Button;
    QLabel *ServoNo_Lable;
    QLineEdit *goalPoint_lineEdit;
    QPushButton *GetServoNo_Button;
    QPlainTextEdit *MessageText;
    QPushButton *Initiate_Button;
    QLabel *jointPosition_label;
    QComboBox *jointName_comboBox;
    QLineEdit *ServoNo_LineEdit;
    QMenuBar *menuBar;
    QMenu *menuTools;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(809, 535);
        actionJoint_Datas = new QAction(MainWindow);
        actionJoint_Datas->setObjectName(QString::fromUtf8("actionJoint_Datas"));
        actionEnd_Effector_Datas = new QAction(MainWindow);
        actionEnd_Effector_Datas->setObjectName(QString::fromUtf8("actionEnd_Effector_Datas"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        start_Button = new QPushButton(centralWidget);
        start_Button->setObjectName(QString::fromUtf8("start_Button"));

        gridLayout->addWidget(start_Button, 0, 2, 1, 1);

        SetServoNo_Button = new QPushButton(centralWidget);
        SetServoNo_Button->setObjectName(QString::fromUtf8("SetServoNo_Button"));

        gridLayout->addWidget(SetServoNo_Button, 11, 2, 1, 1);

        jointName_label = new QLabel(centralWidget);
        jointName_label->setObjectName(QString::fromUtf8("jointName_label"));
        jointName_label->setAlignment(Qt::AlignBottom|Qt::AlignHCenter);

        gridLayout->addWidget(jointName_label, 2, 1, 1, 2);

        jointPosition_lineEdit = new QLineEdit(centralWidget);
        jointPosition_lineEdit->setObjectName(QString::fromUtf8("jointPosition_lineEdit"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(jointPosition_lineEdit->sizePolicy().hasHeightForWidth());
        jointPosition_lineEdit->setSizePolicy(sizePolicy);

        gridLayout->addWidget(jointPosition_lineEdit, 5, 1, 1, 2);

        setGoalPoint_Button = new QPushButton(centralWidget);
        setGoalPoint_Button->setObjectName(QString::fromUtf8("setGoalPoint_Button"));
        sizePolicy.setHeightForWidth(setGoalPoint_Button->sizePolicy().hasHeightForWidth());
        setGoalPoint_Button->setSizePolicy(sizePolicy);

        gridLayout->addWidget(setGoalPoint_Button, 9, 1, 1, 2);

        end_Button = new QPushButton(centralWidget);
        end_Button->setObjectName(QString::fromUtf8("end_Button"));

        gridLayout->addWidget(end_Button, 1, 2, 1, 1);

        setJointPosition_Button = new QPushButton(centralWidget);
        setJointPosition_Button->setObjectName(QString::fromUtf8("setJointPosition_Button"));
        sizePolicy.setHeightForWidth(setJointPosition_Button->sizePolicy().hasHeightForWidth());
        setJointPosition_Button->setSizePolicy(sizePolicy);

        gridLayout->addWidget(setJointPosition_Button, 6, 1, 1, 2);

        goalPoint_label = new QLabel(centralWidget);
        goalPoint_label->setObjectName(QString::fromUtf8("goalPoint_label"));
        goalPoint_label->setAlignment(Qt::AlignBottom|Qt::AlignHCenter);

        gridLayout->addWidget(goalPoint_label, 7, 1, 1, 2);

        StateReset_Button = new QPushButton(centralWidget);
        StateReset_Button->setObjectName(QString::fromUtf8("StateReset_Button"));

        gridLayout->addWidget(StateReset_Button, 1, 1, 1, 1);

        ServoNo_Lable = new QLabel(centralWidget);
        ServoNo_Lable->setObjectName(QString::fromUtf8("ServoNo_Lable"));
        ServoNo_Lable->setLayoutDirection(Qt::LeftToRight);
        ServoNo_Lable->setAlignment(Qt::AlignBottom|Qt::AlignHCenter);

        gridLayout->addWidget(ServoNo_Lable, 10, 1, 1, 1);

        goalPoint_lineEdit = new QLineEdit(centralWidget);
        goalPoint_lineEdit->setObjectName(QString::fromUtf8("goalPoint_lineEdit"));
        sizePolicy.setHeightForWidth(goalPoint_lineEdit->sizePolicy().hasHeightForWidth());
        goalPoint_lineEdit->setSizePolicy(sizePolicy);

        gridLayout->addWidget(goalPoint_lineEdit, 8, 1, 1, 2);

        GetServoNo_Button = new QPushButton(centralWidget);
        GetServoNo_Button->setObjectName(QString::fromUtf8("GetServoNo_Button"));

        gridLayout->addWidget(GetServoNo_Button, 10, 2, 1, 1);

        MessageText = new QPlainTextEdit(centralWidget);
        MessageText->setObjectName(QString::fromUtf8("MessageText"));
        MessageText->setReadOnly(true);

        gridLayout->addWidget(MessageText, 0, 0, 12, 1);

        Initiate_Button = new QPushButton(centralWidget);
        Initiate_Button->setObjectName(QString::fromUtf8("Initiate_Button"));

        gridLayout->addWidget(Initiate_Button, 0, 1, 1, 1);

        jointPosition_label = new QLabel(centralWidget);
        jointPosition_label->setObjectName(QString::fromUtf8("jointPosition_label"));
        jointPosition_label->setAlignment(Qt::AlignBottom|Qt::AlignHCenter);

        gridLayout->addWidget(jointPosition_label, 4, 1, 1, 2);

        jointName_comboBox = new QComboBox(centralWidget);
        jointName_comboBox->setObjectName(QString::fromUtf8("jointName_comboBox"));
        sizePolicy.setHeightForWidth(jointName_comboBox->sizePolicy().hasHeightForWidth());
        jointName_comboBox->setSizePolicy(sizePolicy);

        gridLayout->addWidget(jointName_comboBox, 3, 1, 1, 2);

        ServoNo_LineEdit = new QLineEdit(centralWidget);
        ServoNo_LineEdit->setObjectName(QString::fromUtf8("ServoNo_LineEdit"));
        sizePolicy.setHeightForWidth(ServoNo_LineEdit->sizePolicy().hasHeightForWidth());
        ServoNo_LineEdit->setSizePolicy(sizePolicy);

        gridLayout->addWidget(ServoNo_LineEdit, 11, 1, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 809, 22));
        menuTools = new QMenu(menuBar);
        menuTools->setObjectName(QString::fromUtf8("menuTools"));
        MainWindow->setMenuBar(menuBar);

        menuBar->addAction(menuTools->menuAction());
        menuTools->addSeparator();
        menuTools->addAction(actionJoint_Datas);
        menuTools->addAction(actionEnd_Effector_Datas);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        actionJoint_Datas->setText(QApplication::translate("MainWindow", "Joint Datas", nullptr));
        actionEnd_Effector_Datas->setText(QApplication::translate("MainWindow", "End Effector Datas", nullptr));
        start_Button->setText(QApplication::translate("MainWindow", "Start", nullptr));
        SetServoNo_Button->setText(QApplication::translate("MainWindow", "SetServoNo", nullptr));
        jointName_label->setText(QApplication::translate("MainWindow", "Joint Name", nullptr));
        setGoalPoint_Button->setText(QApplication::translate("MainWindow", "Set Goal Point", nullptr));
        end_Button->setText(QApplication::translate("MainWindow", "End", nullptr));
        setJointPosition_Button->setText(QApplication::translate("MainWindow", "Set Joint Position", nullptr));
        goalPoint_label->setText(QApplication::translate("MainWindow", "Goal Point", nullptr));
        StateReset_Button->setText(QApplication::translate("MainWindow", "StateReset", nullptr));
        ServoNo_Lable->setText(QApplication::translate("MainWindow", "null", nullptr));
        GetServoNo_Button->setText(QApplication::translate("MainWindow", "GetServoNo", nullptr));
        Initiate_Button->setText(QApplication::translate("MainWindow", "Initiate", nullptr));
        jointPosition_label->setText(QApplication::translate("MainWindow", "Joint Position", nullptr));
        menuTools->setTitle(QApplication::translate("MainWindow", "Tools", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
