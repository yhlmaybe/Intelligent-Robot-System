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


    connect(ui->testPerceptionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestPerceptionModule()));
    connect(ui->testAttentionModulel_pushButton, SIGNAL(clicked()), this, SLOT(TestAttentionModule()));
    connect(ui->testMemoryModule_pushButton, SIGNAL(clicked()), this, SLOT(TestMemoryModule()));
    connect(ui->testDecisionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestDecisionModule()));
    connect(ui->testWorldModule_pushButton, SIGNAL(clicked()), this, SLOT(TestWorldModule()));
    connect(ui->testValueEstimationModule_pushButton, SIGNAL(clicked()), this, SLOT(TestValueEstimationModule()));
    connect(ui->testTrainAndDeploy_pushButton, SIGNAL(clicked()), this, SLOT(TestTrainAndDeploy()));

    connect(ui->close_pushButton, SIGNAL(clicked()), this, SLOT(CloseForm()));
}

BrainDeepLearnForm::~BrainDeepLearnForm()
{
    delete ui;
}

void BrainDeepLearnForm::AddData(QString data)
{
    if(this->isHidden()) return;
    std::lock_guard<std::mutex> lock(message_mtx);
    
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

void BrainDeepLearnForm::TestAttentionModule()
{
    brainDeepLearn->TestAttentionModule();
}

void BrainDeepLearnForm::TestMemoryModule()
{
    brainDeepLearn->TestMemoryModule();
}

void BrainDeepLearnForm::TestDecisionModule()
{
    brainDeepLearn->TestDecisionModule();
}

void BrainDeepLearnForm::TestWorldModule()
{
    brainDeepLearn->TestWorldModule();
}

void BrainDeepLearnForm::TestValueEstimationModule()
{
    brainDeepLearn->TestValueEstimationModule();
}

void BrainDeepLearnForm::TestTrainAndDeploy()
{
    brainDeepLearn->TestTrainAndDeploy();
}