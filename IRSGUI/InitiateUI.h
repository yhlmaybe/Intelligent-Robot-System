#ifndef _MAIN_DIALOG_
#define _MAIN_DIALOG_


#include <QDialog>
#include <QLabel>

class Dialog : public QDialog
{

    Q_OBJECT

    public:
    Dialog(QWidget *parent = 0);
    ~Dialog();

    private:
    QLabel *lable;

};


#endif