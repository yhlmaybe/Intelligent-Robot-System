#include "BrainDeepLearnForm.h"
#include "ui_BrainDeepLearnForm.h"

BrainDeepLearnForm::BrainDeepLearnForm(std::shared_ptr<PythonInteraction::Manager> mag, QWidget *parent):
    QWidget(parent),
    ui(new Ui::BrainDeepLearnForm)
{
    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setAttribute(Qt::WA_DeleteOnClose);

    qRegisterMetaType<QTextCursor>("QTextCursor");

    ui->setupUi(this);

    brainDeepLearn = std::make_shared<BrainDeepLearnInterface>(mag, [this](std::string msg)
    {
        QString qMsg = QString::fromStdString(msg);

        QMetaObject::invokeMethod(
            this,
            [this, qMsg]() 
            {
                AddData(qMsg);
            },
            Qt::QueuedConnection);
    });

    connect(ui->testPerceptionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestPerceptionModule()));
    connect(ui->testAttentionModulel_pushButton, SIGNAL(clicked()), this, SLOT(TestAttentionModule()));
    connect(ui->testMemoryModule_pushButton, SIGNAL(clicked()), this, SLOT(TestMemoryModule()));
    connect(ui->testDecisionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestDecisionModule()));
    connect(ui->testWorldModule_pushButton, SIGNAL(clicked()), this, SLOT(TestWorldModule()));
    connect(ui->testValueEstimationModule_pushButton, SIGNAL(clicked()), this, SLOT(TestValueEstimationModule()));
    connect(ui->testTrainModule_pushButton, SIGNAL(clicked()), this, SLOT(TestTrainModule()));
    connect(ui->testTrainOCRModule_pushButton, SIGNAL(clicked()), this, SLOT(TestTrainOCRModule()));
    connect(ui->testConsciousnessModule_pushButton, SIGNAL(clicked()), this, SLOT(TestConsciousnessModule()));
    connect(ui->testIntentionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestIntentionModule()));
    connect(ui->testOCRModule_pushButton, SIGNAL(clicked()), this, SLOT(TestOCRModule()));

    connect(ui->trainModule_pushButton, SIGNAL(clicked()), this, SLOT(TrainModule()));
    connect(ui->trainOCRModule_pushButton, SIGNAL(clicked()), this, SLOT(TrainOCRModule()));
    connect(ui->deployModule_pushButton, SIGNAL(clicked()), this, SLOT(DeployModule()));
    connect(ui->stop_pushButton, SIGNAL(clicked()), this, SLOT(StopModule()));
    connect(ui->pause_pushButton, SIGNAL(clicked()), this, SLOT(PauseModule()));
    connect(ui->resume_pushButton, SIGNAL(clicked()), this, SLOT(ResumeModule()));
    connect(ui->saveParameter_pushButton, SIGNAL(clicked()), this, SLOT(ExportParmFromCheckpoint()));
    connect(ui->resetState_pushButton, SIGNAL(clicked()), this, SLOT(ResetHebbianMemory()));

    connect(ui->clear_pushButton, SIGNAL(clicked()), this, SLOT(ClearText()));
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

void BrainDeepLearnForm::ClearText()
{
    ui->message_textBrowser->clear();
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

void BrainDeepLearnForm::TestTrainModule()
{
    if(ui->onlineLearning_checkBox->isChecked())
    {
        brainDeepLearn->TestModuleTrain(true);
    }
    else
    {
        brainDeepLearn->TestModuleTrain(false);
    }
}

void BrainDeepLearnForm::TestTrainOCRModule()
{
    if(ui->onlineLearning_checkBox->isChecked())
    {
        brainDeepLearn->TestOCRModuleTrain(true);
    }
    else
    {
        brainDeepLearn->TestOCRModuleTrain(false);
    }
}

void BrainDeepLearnForm::TestConsciousnessModule()
{
    brainDeepLearn->TestConsciousnessModule();
}

void BrainDeepLearnForm::TestIntentionModule()
{
    brainDeepLearn->TestIntentionModule();
}

void BrainDeepLearnForm::TestOCRModule()
{
    brainDeepLearn->TestOCRModule();
}

void BrainDeepLearnForm::TrainModule()
{
    brainDeepLearn->TrainModule();
}

void BrainDeepLearnForm::TrainOCRModule()
{
    brainDeepLearn->TrainOCRModule();
}


void BrainDeepLearnForm::ExportParmFromCheckpoint()
{
    brainDeepLearn->ExportParmFromCheckpoint(ui->saveParmOverride_checkBox->isChecked());
}

void BrainDeepLearnForm::DeployModule()
{
    brainDeepLearn->DeployModule();
}

void BrainDeepLearnForm::StopModule()
{
    brainDeepLearn->Stop();
}

void BrainDeepLearnForm::PauseModule()
{
    brainDeepLearn->Pause();
}

void BrainDeepLearnForm::ResumeModule()
{
    brainDeepLearn->Resume();
}

void BrainDeepLearnForm::ResetHebbianMemory()
{
    brainDeepLearn->ResetHebbianMemory();
}
