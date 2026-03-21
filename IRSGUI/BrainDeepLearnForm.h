#ifndef BRAINDEEPLEARNFORM_H
#define BRAINDEEPLEARNFORM_H

#include <python3.8/Python.h>//it must be placed before <QWidget>, otherwise an error will be reported
#include <QWidget>
#include <QScrollBar>
#include <mutex>

#include "../BrainDeepLearn/Interface.h"
#include "../include/PythonInteraction.h"

namespace Ui {
class BrainDeepLearnForm;
}

class BrainDeepLearnForm : public QWidget
{
    Q_OBJECT

public:
    explicit BrainDeepLearnForm(std::shared_ptr<PythonInteraction::Manager> mag, QWidget *parent = nullptr);
    ~BrainDeepLearnForm();

private:
    Ui::BrainDeepLearnForm *ui;
    mutable std::mutex message_mtx;

    std::shared_ptr<PythonInteraction::Manager> pyManager = nullptr;

    BrainDeepLearnForm(const BrainDeepLearnForm&) = delete;
    BrainDeepLearnForm& operator = (const BrainDeepLearnForm&) = delete;

    std::shared_ptr<BrainDeepLearnInterface> brainDeepLearn;

public slots:    
    void AddData(QString data);


private slots:

    void CloseForm();

    void ClearText();

    void TestPerceptionModule();

    void TestAttentionModule();

    void TestMemoryModule();

    void TestDecisionModule();

    void TestWorldModule();

    void TestValueEstimationModule();

    void TestTrainModule();

    void TestTrainOCRModule();

    void TestTrainOCRRecognitionModule();

    void TestConsciousnessModule();

    void TestIntentionModule();

    void TestOCRModule();

    void TrainModule();

    void TrainOCRModule();

    void ExportParmFromCheckpoint();

    void SetOCRTrainPicturePath();

    void SetOCRTrainTextPath();

    void SetOCRRecognizeTrainPicturePath();

    void SetOCRRecognizeTrainTextPath();

    void SetOCRCheckPointPath();

    void SetOCRRecognizeCheckPointPath();

    void SetOCRParameterPath();

    void SetOCRRecognizeParameterPath();

    void DeployModule();

    void StopModule();

    void PauseModule();

    void ResumeModule();

    void ResetHebbianMemory();

private:
    QString StatusValueToQString(BrainDeepLearnInterface::StatusValue& value);
    void ChooseFolderAndSetParameter(QString title, std::string parameterName, QString fileName = "");
    void RefreshPathBrowserFromParameters();
};

#endif // BRAINDEEPLEARNFORM_H
