#ifndef JOINTDATASFORM_H
#define JOINTDATASFORM_H

#include <python3.8/Python.h>//it must be placed before <QWidget>, otherwise an error will be reported
#include <QWidget>
#include <QScrollBar>
#include <mutex>

namespace Ui {
class JointDatasForm;
}

class JointDatasForm : public QWidget
{
    Q_OBJECT

public:
    explicit JointDatasForm(QWidget *parent = nullptr);
    ~JointDatasForm();
    
public slots:    
    void AddData(QString data);

private:
    Ui::JointDatasForm *ui;

    mutable std::mutex message_mtx;

    JointDatasForm(const JointDatasForm&) = delete;
    JointDatasForm& operator = (const JointDatasForm&) = delete;

private slots:

    void CloseForm();
};

#endif // JOINTDATASFORM_H
