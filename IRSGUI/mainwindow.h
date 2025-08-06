#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <python3.8/Python.h>//it must be placed before <QMainWindow>, otherwise an error will be reported
#include <QMainWindow>
#include <rclcpp/rclcpp.hpp>
#include <QScrollBar>
#include <QSharedMemory>
#include <QMessageBox>
#include <QPointer>

#include "../IRSManager/IRSCoreManager.h"
#include "../include/IRSFunction.h"
#include "../include/PythonInteraction.h"
#include "JointDatasForm.h"
#include "EndEffectorDatasForm.h"
#include "BrainDeepLearnForm.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    static MainWindow* GetInstance();
    Ui::MainWindow* GetUI();
    
signals:
    void SetJointDatasFormDatas(QString QMessage);
    void SetEndEffectorDatasFormDatas(QString QMessage);
    void SetBrainDeepLearnFormDatas(QString QMessage);

private:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void closeEvent(QCloseEvent *event) override;

    MainWindow(const MainWindow&) = delete;
    MainWindow& operator = (const MainWindow&);

    Ui::MainWindow *ui;

    std::shared_ptr<PythonInteraction::Manager> py_manager;

    std::vector<QWidget*> forms;

    JointDatasForm *joint_datas_form;

    EndEffectorDatasForm *end_effector_datas_form;

    BrainDeepLearnForm *brain_deep_learn_form;

    mutable std::mutex message_mtx;

    bool is_running = false;

    bool is_initial = false;

public slots:

    void SetMessage(QString QMessage);

private slots:

    void OpenJointDatasForm();

    void OpenEndEffectDatasForm();

    void OpenBrainDeepLearnForm();

    void SetServoNo();

    void GetServoNo();

    void Initiate();

    void Start();

    void End();

    void StateReset();

    void SetJointPosition();

    void SetGoalPoint();

    void SetPythonMessageToTextBrowser(const char* data, std::size_t len, const std::string& mouduleName);
};

#endif // MAINWINDOW_H
