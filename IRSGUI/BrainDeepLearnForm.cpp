#include "BrainDeepLearnForm.h"
#include "ui_BrainDeepLearnForm.h"

BrainDeepLearnForm::BrainDeepLearnForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::BrainDeepLearnForm)
{
    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setAttribute(Qt::WA_DeleteOnClose);

    ui->setupUi(this);

    brainDeepLearn = std::make_shared<BrainDeepLearnInterface>();

    brainDeepLearn->SetPrintCallback([this](std::string data) {this->SetMessageToTextBrowser(data);});

    connect(ui->testPerceptionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestPerceptionModule()));
}

BrainDeepLearnForm::~BrainDeepLearnForm()
{
    delete ui;
}

void BrainDeepLearnForm::SetMessageToTextBrowser(std::string message)
{
    std::lock_guard<std::mutex> lock(message_mtx);
    QString qstr = QString::fromStdString(message);
    AddData(qstr);
}

void BrainDeepLearnForm::AddData(QString data)
{
    if(this->isHidden()) return;
    
    ui->message_textBrowser->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
    ui->message_textBrowser->insertPlainText("\n");
    ui->message_textBrowser->insertPlainText(data);
    QScrollBar *scrollbar = ui->message_textBrowser->verticalScrollBar();
    if(scrollbar)  
    {
        scrollbar->setSliderPosition(scrollbar->maximum());
    }  
}

void BrainDeepLearnForm::CloseForm()
{
    this->hide();
}

void BrainDeepLearnForm::TestPerceptionModule()
{
    brainDeepLearn->TestPerceptionModule();
}