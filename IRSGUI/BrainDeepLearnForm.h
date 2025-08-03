#ifndef BRAINDEEPLEARNFORM_H
#define BRAINDEEPLEARNFORM_H

#include <python3.8/Python.h>//it must be placed before <QWidget>, otherwise an error will be reported
#include <QWidget>
#include <QScrollBar>
#include <mutex>

#include "../BrainDeepLearn/Interface.h"

namespace Ui {
class BrainDeepLearnForm;
}

class BrainDeepLearnForm : public QWidget
{
    Q_OBJECT

public:
    explicit BrainDeepLearnForm(QWidget *parent = nullptr);
    ~BrainDeepLearnForm();

private:
    Ui::BrainDeepLearnForm *ui;
    mutable std::mutex message_mtx;

    BrainDeepLearnForm(const BrainDeepLearnForm&) = delete;
    BrainDeepLearnForm& operator = (const BrainDeepLearnForm&) = delete;

    std::shared_ptr<BrainDeepLearnInterface> brainDeepLearn;

    void SetMessageToTextBrowser(std::string message);

public slots:    
    void AddData(QString data);


private slots:

    void CloseForm();

    void TestPerceptionModule();
};

#endif // BRAINDEEPLEARNFORM_H
