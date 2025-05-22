/********************************************************************************
** Form generated from reading UI file 'JointDatasForm.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_JOINTDATASFORM_H
#define UI_JOINTDATASFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_JointDatasForm
{
public:
    QGridLayout *gridLayout;
    QTextBrowser *datas_textBrowser;
    QPushButton *close_pushButton;

    void setupUi(QWidget *JointDatasForm)
    {
        if (JointDatasForm->objectName().isEmpty())
            JointDatasForm->setObjectName(QString::fromUtf8("JointDatasForm"));
        JointDatasForm->resize(400, 300);
        gridLayout = new QGridLayout(JointDatasForm);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        datas_textBrowser = new QTextBrowser(JointDatasForm);
        datas_textBrowser->setObjectName(QString::fromUtf8("datas_textBrowser"));

        gridLayout->addWidget(datas_textBrowser, 0, 0, 1, 1);

        close_pushButton = new QPushButton(JointDatasForm);
        close_pushButton->setObjectName(QString::fromUtf8("close_pushButton"));

        gridLayout->addWidget(close_pushButton, 1, 0, 1, 1);


        retranslateUi(JointDatasForm);

        QMetaObject::connectSlotsByName(JointDatasForm);
    } // setupUi

    void retranslateUi(QWidget *JointDatasForm)
    {
        JointDatasForm->setWindowTitle(QApplication::translate("JointDatasForm", "JointDatasForm", nullptr));
        close_pushButton->setText(QApplication::translate("JointDatasForm", "Close", nullptr));
    } // retranslateUi

};

namespace Ui {
    class JointDatasForm: public Ui_JointDatasForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_JOINTDATASFORM_H
