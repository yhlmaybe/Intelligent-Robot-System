#include "BrainDeepLearnForm.h"
#include "ui_BrainDeepLearnForm.h"

#include <QDir>
#include <QFileDialog>
#include <QStringList>
#include <QTextDocument>

BrainDeepLearnForm::BrainDeepLearnForm(std::shared_ptr<PythonInteraction::Manager> mag, QWidget *parent):
    QWidget(parent),
    ui(new Ui::BrainDeepLearnForm)
{
    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setAttribute(Qt::WA_DeleteOnClose);

    qRegisterMetaType<QTextCursor>("QTextCursor");

    ui->setupUi(this);
    ui->message_textBrowser->document()->setMaximumBlockCount(1000);

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
    connect(ui->testTrainOCRRecognitionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestTrainOCRRecognitionModule()));
    connect(ui->testConsciousnessModule_pushButton, SIGNAL(clicked()), this, SLOT(TestConsciousnessModule()));
    connect(ui->testIntentionModule_pushButton, SIGNAL(clicked()), this, SLOT(TestIntentionModule()));
    connect(ui->testOCRModule_pushButton, SIGNAL(clicked()), this, SLOT(TestOCRModule()));

    connect(ui->trainModule_pushButton, SIGNAL(clicked()), this, SLOT(TrainModule()));
    connect(ui->trainOCRModule_pushButton, SIGNAL(clicked()), this, SLOT(TrainOCRModule()));
    connect(ui->OCRTrainPicture_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRTrainPicturePath()));
    connect(ui->OCRTrainText_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRTrainTextPath()));
    connect(ui->OCRRecognizeTrainPicture_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRRecognizeTrainPicturePath()));
    connect(ui->OCRRecognizeTrainText_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRRecognizeTrainTextPath()));
    connect(ui->OCRCheckPoint_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRCheckPointPath()));
    connect(ui->OCRRecognizeCheckPoint_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRRecognizeCheckPointPath()));
    connect(ui->OCRParameter_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRParameterPath()));
    connect(ui->OCRRecognizeParameter_pushButton, SIGNAL(clicked()), this, SLOT(SetOCRRecognizeParameterPath()));
    connect(ui->deployModule_pushButton, SIGNAL(clicked()), this, SLOT(DeployModule()));
    connect(ui->stop_pushButton, SIGNAL(clicked()), this, SLOT(StopModule()));
    connect(ui->pause_pushButton, SIGNAL(clicked()), this, SLOT(PauseModule()));
    connect(ui->resume_pushButton, SIGNAL(clicked()), this, SLOT(ResumeModule()));
    connect(ui->saveParameter_pushButton, SIGNAL(clicked()), this, SLOT(ExportParmFromCheckpoint()));
    connect(ui->resetState_pushButton, SIGNAL(clicked()), this, SLOT(ResetHebbianMemory()));

    connect(ui->clear_pushButton, SIGNAL(clicked()), this, SLOT(ClearText()));
    connect(ui->close_pushButton, SIGNAL(clicked()), this, SLOT(CloseForm()));

    RefreshPathBrowserFromParameters();
}


BrainDeepLearnForm::~BrainDeepLearnForm()
{
    delete ui;
}

bool BrainDeepLearnForm::GetCurrentVisualData(QImage& image, QString& text, double& updatedAt)
{
    BrainDeepLearnInterface::VisualStatus status;
    if (!brainDeepLearn->GetCurrentVisualStatus(status))
    {
        return false;
    }

    text = QString::fromStdString(status.text);
    updatedAt = status.updatedAt;

    const std::size_t expectedSize = static_cast<std::size_t>(status.width) * static_cast<std::size_t>(status.height) * 3;
    if (status.width <= 0 || status.height <= 0 || status.bitmapRgb.size() < expectedSize)
    {
        image = QImage();
        return true;
    }

    QImage rawImage(
        status.bitmapRgb.data(),
        status.width,
        status.height,
        status.width * 3,
        QImage::Format_RGB888);
    image = rawImage.copy();
    return true;
}

bool BrainDeepLearnForm::SetVisualStateEnabled(bool enabled)
{
    return brainDeepLearn->SetVisualStateEnabled(enabled);
}

QString BrainDeepLearnForm::StatusValueToQString(BrainDeepLearnInterface::StatusValue& value)
{
    if (int* v = boost::get<int>(&value))
    {
        return QString::number(*v);
    }

    if (double* v = boost::get<double>(&value))
    {
        return QString::number(*v);
    }

    if (std::string* v = boost::get<std::string>(&value))
    {
        return QString::fromStdString(*v);
    }

    return QString();
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
    brainDeepLearn->TestOCRModuleTrain(); 
}

void BrainDeepLearnForm::TestTrainOCRRecognitionModule()
{
    brainDeepLearn->TestOCRRecognitionTrain();
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

void BrainDeepLearnForm::SetOCRTrainPicturePath()
{
    ChooseFolderAndSetParameter("Select OCR Train Picture Folder", "OCR_FRAMES_PATH");
}

void BrainDeepLearnForm::SetOCRTrainTextPath()
{
    ChooseFolderAndSetParameter("Select OCR Train Text Folder", "OCR_TEXTS_PATH");
}

void BrainDeepLearnForm::SetOCRRecognizeTrainPicturePath()
{
    ChooseFolderAndSetParameter("Select OCR Recognize Train Picture Folder", "OCR_RECOGNIZER_FRAMES_PATH");
}

void BrainDeepLearnForm::SetOCRRecognizeTrainTextPath()
{
    ChooseFolderAndSetParameter("Select OCR Recognize Train Text Folder", "OCR_RECOGNIZER_TEXTS_PATH");
}

void BrainDeepLearnForm::SetOCRCheckPointPath()
{
    ChooseFolderAndSetParameter("Select OCR Check Point Folder", "OCR_CKPT_PATH_TRAIN", "ocr_training_checkpoint.pth");
}

void BrainDeepLearnForm::SetOCRRecognizeCheckPointPath()
{
    ChooseFolderAndSetParameter("Select OCR Recognize Check Point Folder", "OCR_RECOGNIZER_CKPT_PATH_TRAIN", "ocr_recognizer_training_checkpoint.pth");
}

void BrainDeepLearnForm::SetOCRParameterPath()
{
    ChooseFolderAndSetParameter("Select OCR Parameter Folder", "OCR_MODULEPARAMETER_PATH", "ocr_module_parameter.pth");
}

void BrainDeepLearnForm::SetOCRRecognizeParameterPath()
{
    ChooseFolderAndSetParameter("Select OCR Recognize Parameter Folder", "OCR_RECOGNIZER_MODULEPARAMETER_PATH", "ocr_recognizer_parameter.pth");
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

void BrainDeepLearnForm::ChooseFolderAndSetParameter(QString title, std::string parameterName, QString fileName)
{
    QString selectedDir = QFileDialog::getExistingDirectory(
        this,
        title,
        QDir::homePath(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (selectedDir.isEmpty())
    {
        return;
    }

    QString finalPath = selectedDir;
    if (!fileName.isEmpty())
    {
        finalPath = QDir(selectedDir).filePath(fileName);
    }

    if (brainDeepLearn->SetBasicParameters(parameterName, finalPath.toStdString()))
    {
        RefreshPathBrowserFromParameters();
        AddData(QString("Set %1 = %2").arg(QString::fromStdString(parameterName), finalPath));
    }
    else
    {
        AddData(QString("Set %1 failed").arg(QString::fromStdString(parameterName)));
    }
}

void BrainDeepLearnForm::RefreshPathBrowserFromParameters()
{
    BrainDeepLearnInterface::StatusMap parameters;
    if (!brainDeepLearn->GetBasicParametersDict(parameters))
    {
        ui->path_textBrowser->setPlainText("GetBasicParametersDict failed");
        return;
    }

    QStringList lines;
    for (std::pair<std::string, BrainDeepLearnInterface::StatusValue> item : parameters)
    {
        QString key = QString::fromStdString(item.first);
        if (!key.contains("PATH"))
        {
            continue;
        }

        QString value = StatusValueToQString(item.second);
        lines << QString("%1 = %2").arg(key, value);
    }

    ui->path_textBrowser->setPlainText(lines.join("\n\n"));
}
