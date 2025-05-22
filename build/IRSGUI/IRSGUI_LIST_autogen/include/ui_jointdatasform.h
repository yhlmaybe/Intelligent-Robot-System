/********************************************************************************
** Form generated from reading UI file 'jointdatasform.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_JOINTDATASFORM_H
#define UI_JOINTDATASFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_JointDatasForm
{
public:
    QTextBrowser *jointdatas_textBrowser;

    void setupUi(QWidget *JointDatasForm)
    {
        if (JointDatasForm->objectName().isEmpty())
            JointDatasForm->setObjectName(QString::fromUtf8("JointDatasForm"));
        JointDatasForm->resize(530, 399);
        jointdatas_textBrowser = new QTextBrowser(JointDatasForm);
        jointdatas_textBrowser->setObjectName(QString::fromUtf8("jointdatas_textBrowser"));
        jointdatas_textBrowser->setGeometry(QRect(10, 10, 511, 381));

        retranslateUi(JointDatasForm);

        QMetaObject::connectSlotsByName(JointDatasForm);
    } // setupUi

    void retranslateUi(QWidget *JointDatasForm)
    {
        JointDatasForm->setWindowTitle(QApplication::translate("JointDatasForm", "JointDatas", nullptr));
    } // retranslateUi

};

namespace Ui {
    class JointDatasForm: public Ui_JointDatasForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_JOINTDATASFORM_H
