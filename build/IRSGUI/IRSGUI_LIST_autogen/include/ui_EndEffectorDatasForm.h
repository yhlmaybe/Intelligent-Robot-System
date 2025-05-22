/********************************************************************************
** Form generated from reading UI file 'EndEffectorDatasForm.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ENDEFFECTORDATASFORM_H
#define UI_ENDEFFECTORDATASFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_EndEffectorDatasForm
{
public:
    QGridLayout *gridLayout;
    QTextBrowser *datas_textBrowser;
    QPushButton *close_pushButton;

    void setupUi(QWidget *EndEffectorDatasForm)
    {
        if (EndEffectorDatasForm->objectName().isEmpty())
            EndEffectorDatasForm->setObjectName(QString::fromUtf8("EndEffectorDatasForm"));
        EndEffectorDatasForm->resize(425, 350);
        gridLayout = new QGridLayout(EndEffectorDatasForm);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        datas_textBrowser = new QTextBrowser(EndEffectorDatasForm);
        datas_textBrowser->setObjectName(QString::fromUtf8("datas_textBrowser"));

        gridLayout->addWidget(datas_textBrowser, 0, 0, 1, 1);

        close_pushButton = new QPushButton(EndEffectorDatasForm);
        close_pushButton->setObjectName(QString::fromUtf8("close_pushButton"));

        gridLayout->addWidget(close_pushButton, 1, 0, 1, 1);


        retranslateUi(EndEffectorDatasForm);

        QMetaObject::connectSlotsByName(EndEffectorDatasForm);
    } // setupUi

    void retranslateUi(QWidget *EndEffectorDatasForm)
    {
        EndEffectorDatasForm->setWindowTitle(QApplication::translate("EndEffectorDatasForm", "EndEffectorDatasForm", nullptr));
        close_pushButton->setText(QApplication::translate("EndEffectorDatasForm", "Close", nullptr));
    } // retranslateUi

};

namespace Ui {
    class EndEffectorDatasForm: public Ui_EndEffectorDatasForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ENDEFFECTORDATASFORM_H
