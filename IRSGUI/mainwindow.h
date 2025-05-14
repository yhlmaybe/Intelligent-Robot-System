#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <python3.8/Python.h>//it must be placed before <QMainWindow>, otherwise an error will be reported
#include <QMainWindow>
#include <rclcpp/rclcpp.hpp>
#include <QScrollBar>
#include <QSharedMemory>
#include <QMessageBox>

#include "../IRSManager/IRSCoreManager.h"
#include "../include/IRSFunction.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    static MainWindow* GetInstance();
    Ui::MainWindow* GetUI();

private:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void closeEvent(QCloseEvent *event) override;

    MainWindow(const MainWindow&) = delete;
    MainWindow& operator = (const MainWindow&);

    Ui::MainWindow *ui;

    mutable std::mutex message_mtx;

    bool is_running = false;

    bool is_initial = false;

public slots:
    void SetMessage(QString QMessage);

private slots:

    void SetServoNo();

    void GetServoNo();

    void Initiate();

    void Start();

    void End();

    void StateReset();

    void SetJointPosition();

    void SetGoalPoint();
};

#endif // MAINWINDOW_H
