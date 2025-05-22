#include "JointDatasForm.h"
#include "ui_JointDatasForm.h"

JointDatasForm::JointDatasForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::JointDatasForm)
{
    ui->setupUi(this);

    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setAttribute(Qt::WA_DeleteOnClose);

    connect(ui->close_pushButton, SIGNAL(clicked()), this, SLOT(CloseForm()));

}

JointDatasForm::~JointDatasForm()
{
    delete ui;
}

void JointDatasForm::AddData(QString data)
{
    if(this->isHidden()) return;
    std::lock_guard<std::mutex> lock(message_mtx);
    ui->datas_textBrowser->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
    ui->datas_textBrowser->insertPlainText("\n");
    ui->datas_textBrowser->insertPlainText(data);
    QScrollBar *scrollbar = ui->datas_textBrowser->verticalScrollBar();
    if(scrollbar)  
    {
        scrollbar->setSliderPosition(scrollbar->maximum());
    }  
}

void JointDatasForm::CloseForm()
{
    this->hide();
}
