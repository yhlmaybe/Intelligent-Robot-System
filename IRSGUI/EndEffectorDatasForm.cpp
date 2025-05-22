#include "EndEffectorDatasForm.h"
#include "ui_EndEffectorDatasForm.h"

EndEffectorDatasForm::EndEffectorDatasForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::EndEffectorDatasForm)
{
    ui->setupUi(this);

    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setAttribute(Qt::WA_DeleteOnClose);

    connect(ui->close_pushButton, SIGNAL(clicked()), this, SLOT(CloseForm()));
}

EndEffectorDatasForm::~EndEffectorDatasForm()
{
    delete ui;
}

void EndEffectorDatasForm::AddData(QString data)
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

void EndEffectorDatasForm::CloseForm()
{
    this->hide();
}
