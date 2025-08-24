/********************************************************************************
** Form generated from reading UI file 'BrainDeepLearnForm.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BRAINDEEPLEARNFORM_H
#define UI_BRAINDEEPLEARNFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_BrainDeepLearnForm
{
public:
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout;
    QTextBrowser *message_textBrowser;
    QVBoxLayout *verticalLayout;
    QPushButton *trainModule_pushButton;
    QPushButton *testPerceptionModule_pushButton;
    QPushButton *testAttentionModulel_pushButton;
    QPushButton *testMemoryModule_pushButton;
    QPushButton *testDecisionModule_pushButton;
    QPushButton *testWorldModule_pushButton;
    QPushButton *testValueEstimationModule_pushButton;
    QPushButton *close_pushButton;

    void setupUi(QWidget *BrainDeepLearnForm)
    {
        if (BrainDeepLearnForm->objectName().isEmpty())
            BrainDeepLearnForm->setObjectName(QString::fromUtf8("BrainDeepLearnForm"));
        BrainDeepLearnForm->resize(626, 495);
        gridLayout = new QGridLayout(BrainDeepLearnForm);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        message_textBrowser = new QTextBrowser(BrainDeepLearnForm);
        message_textBrowser->setObjectName(QString::fromUtf8("message_textBrowser"));

        horizontalLayout->addWidget(message_textBrowser);


        gridLayout->addLayout(horizontalLayout, 0, 0, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        trainModule_pushButton = new QPushButton(BrainDeepLearnForm);
        trainModule_pushButton->setObjectName(QString::fromUtf8("trainModule_pushButton"));

        verticalLayout->addWidget(trainModule_pushButton);

        testPerceptionModule_pushButton = new QPushButton(BrainDeepLearnForm);
        testPerceptionModule_pushButton->setObjectName(QString::fromUtf8("testPerceptionModule_pushButton"));

        verticalLayout->addWidget(testPerceptionModule_pushButton);

        testAttentionModulel_pushButton = new QPushButton(BrainDeepLearnForm);
        testAttentionModulel_pushButton->setObjectName(QString::fromUtf8("testAttentionModulel_pushButton"));

        verticalLayout->addWidget(testAttentionModulel_pushButton);

        testMemoryModule_pushButton = new QPushButton(BrainDeepLearnForm);
        testMemoryModule_pushButton->setObjectName(QString::fromUtf8("testMemoryModule_pushButton"));

        verticalLayout->addWidget(testMemoryModule_pushButton);

        testDecisionModule_pushButton = new QPushButton(BrainDeepLearnForm);
        testDecisionModule_pushButton->setObjectName(QString::fromUtf8("testDecisionModule_pushButton"));

        verticalLayout->addWidget(testDecisionModule_pushButton);

        testWorldModule_pushButton = new QPushButton(BrainDeepLearnForm);
        testWorldModule_pushButton->setObjectName(QString::fromUtf8("testWorldModule_pushButton"));

        verticalLayout->addWidget(testWorldModule_pushButton);

        testValueEstimationModule_pushButton = new QPushButton(BrainDeepLearnForm);
        testValueEstimationModule_pushButton->setObjectName(QString::fromUtf8("testValueEstimationModule_pushButton"));

        verticalLayout->addWidget(testValueEstimationModule_pushButton);


        gridLayout->addLayout(verticalLayout, 0, 1, 1, 1);

        close_pushButton = new QPushButton(BrainDeepLearnForm);
        close_pushButton->setObjectName(QString::fromUtf8("close_pushButton"));

        gridLayout->addWidget(close_pushButton, 1, 0, 1, 1);


        retranslateUi(BrainDeepLearnForm);

        QMetaObject::connectSlotsByName(BrainDeepLearnForm);
    } // setupUi

    void retranslateUi(QWidget *BrainDeepLearnForm)
    {
        BrainDeepLearnForm->setWindowTitle(QApplication::translate("BrainDeepLearnForm", "BrainDeepLearnForm", nullptr));
        trainModule_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TrainModule", nullptr));
        testPerceptionModule_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TestPerceptionModule", nullptr));
        testAttentionModulel_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TestAttentionModule", nullptr));
        testMemoryModule_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TestMemoryModule", nullptr));
        testDecisionModule_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TestDecisionModule", nullptr));
        testWorldModule_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TestWorldModule", nullptr));
#ifndef QT_NO_SHORTCUT
        testWorldModule_pushButton->setShortcut(QApplication::translate("BrainDeepLearnForm", "Ctrl+S", nullptr));
#endif // QT_NO_SHORTCUT
        testValueEstimationModule_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "TestValueEstimationModule", nullptr));
        close_pushButton->setText(QApplication::translate("BrainDeepLearnForm", "Close", nullptr));
    } // retranslateUi

};

namespace Ui {
    class BrainDeepLearnForm: public Ui_BrainDeepLearnForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BRAINDEEPLEARNFORM_H
