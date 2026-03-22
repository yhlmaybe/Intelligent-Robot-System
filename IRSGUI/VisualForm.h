#ifndef VISUALFORM_H
#define VISUALFORM_H

#include <python3.8/Python.h>//it must be placed before <QWidget>, otherwise an error will be reported
#include <QWidget>
#include <QImage>
#include <QPixmap>

namespace Ui {
class VisualForm;
}

class QResizeEvent;
class QShowEvent;
class QHideEvent;

class VisualForm : public QWidget
{
    Q_OBJECT

public:
    explicit VisualForm(QWidget *parent = nullptr);
    ~VisualForm();

public slots:
    void SetBitmap(QImage bitmap);
    void SetBitmap(QPixmap bitmap);
    void SetText(QString text);
    void SetVisualData(QImage bitmap, QString text);

signals:
    void VisualShown();
    void VisualHidden();

protected:
    void resizeEvent(QResizeEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private:
    Ui::VisualForm *ui;
    QPixmap currentPixmap;

    VisualForm(const VisualForm&) = delete;
    VisualForm& operator = (const VisualForm&) = delete;

    void RefreshBitmap();

private slots:
    void CloseForm();
};

#endif // VISUALFORM_H
