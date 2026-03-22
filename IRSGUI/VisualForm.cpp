#include "VisualForm.h"
#include "ui_VisualForm.h"

#include <QResizeEvent>
#include <QShowEvent>
#include <QHideEvent>

VisualForm::VisualForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::VisualForm)
{
    ui->setupUi(this);

    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowTitleHint);
    setAttribute(Qt::WA_DeleteOnClose);

    connect(ui->close_pushButton, SIGNAL(clicked()), this, SLOT(CloseForm()));
}

VisualForm::~VisualForm()
{
    delete ui;
}

void VisualForm::SetBitmap(QImage bitmap)
{
    if (this->isHidden())
        return;
    SetBitmap(QPixmap::fromImage(bitmap));
}

void VisualForm::SetBitmap(QPixmap bitmap)
{
    if (this->isHidden())
        return;
    currentPixmap = bitmap;
    RefreshBitmap();
}

void VisualForm::SetText(QString text)
{
    if (this->isHidden())
        return;
    ui->message_textBrowser->setPlainText(text);
}

void VisualForm::SetVisualData(QImage bitmap, QString text)
{
    if (this->isHidden())
        return;
    SetBitmap(bitmap);
    SetText(text);
}

void VisualForm::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    RefreshBitmap();
}

void VisualForm::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    emit VisualShown();
}

void VisualForm::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    emit VisualHidden();
}

void VisualForm::RefreshBitmap()
{
    if (currentPixmap.isNull())
    {
        ui->bitmap_label->clear();
        ui->bitmap_label->setText("No Image");
        return;
    }

    ui->bitmap_label->setText(QString());
    ui->bitmap_label->setPixmap(
        currentPixmap.scaled(
            ui->bitmap_label->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
}

void VisualForm::CloseForm()
{
    this->hide();
}
