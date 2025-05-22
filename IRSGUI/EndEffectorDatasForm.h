#ifndef ENDEFFECTORDATASFORM_H
#define ENDEFFECTORDATASFORM_H

#include <python3.8/Python.h>//it must be placed before <QWidget>, otherwise an error will be reported
#include <QWidget>
#include <QScrollBar>
#include <mutex>

namespace Ui {
class EndEffectorDatasForm;
}

class EndEffectorDatasForm : public QWidget
{
    Q_OBJECT

public:
    explicit EndEffectorDatasForm(QWidget *parent = nullptr);
    ~EndEffectorDatasForm();

public slots:    
    void AddData(QString data);

private:
    Ui::EndEffectorDatasForm *ui;
    mutable std::mutex message_mtx;

    EndEffectorDatasForm(const EndEffectorDatasForm&) = delete;
    EndEffectorDatasForm& operator = (const EndEffectorDatasForm&) = delete;

private slots:

    void CloseForm();
};

#endif // ENDEFFECTORDATASFORM_H
