#include <QLayout>
#include "InitiateUI.h"

Dialog::Dialog(QWidget *parent) : QDialog(parent)
{
    this->setWindowTitle("hello");

    lable = new QLabel(this);
    lable->setText("helloworld");

    QGridLayout *main_laylot = new QGridLayout(this);
    main_laylot->addWidget(lable, 0, 0);

}


Dialog::~Dialog()
{

}