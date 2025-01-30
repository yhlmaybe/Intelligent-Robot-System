#ifndef KINEMATICTOOL_H
#define KINEMATICTOOL_H

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <nlopt.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <thread>
#include <mutex>
#include <memory>
#include <boost/date_time.hpp>
#include <chrono>
#include <vector>

#include <boost/date_time.hpp>
#include <Eigen/Geometry>
#include <limits>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>
#include <cmath>

#include "IRSFunction.h"
#include "Math3D.h"


namespace IRS_IK
{
    class IRS_IK;
}

namespace KDL
{

enum BasicJointType { RotJoint, TransJoint, Continuous };

class ChainIkSolverPos_TL
{
  friend class IRS_IK::IRS_IK;

public:
  ChainIkSolverPos_TL(
    const Chain & chain, const JntArray & q_min, const JntArray & q_max,
    double maxtime = 0.005, double eps = 1e-3, bool random_restart = false,
    bool try_jl_wrap = false);

  ~ChainIkSolverPos_TL();

  int CartToJnt(
    const KDL::JntArray & q_init, const KDL::Frame & p_in, KDL::JntArray & q_out,
    const KDL::Twist bounds = KDL::Twist::Zero());

  inline void setMaxtime(double t)
  {
    maxtime = std::chrono::duration<double>(t);
  }

private:
  const Chain chain;
  JntArray q_min;
  JntArray q_max;

  KDL::Twist bounds;

  KDL::ChainIkSolverVel_pinv vik_solver;
  KDL::ChainFkSolverPos_recursive fksolver;
  JntArray delta_q;
  std::chrono::duration<double> maxtime;

  double eps;

  bool rr;
  bool wrap;

  std::vector<KDL::BasicJointType> types;

  inline void abort()
  {
    aborted = true;
  }

  inline void reset()
  {
    aborted = false;
  }

  bool aborted;

  Frame f;
  Twist delta_twist;

  inline static double fRand(double min, double max)
  {
    double f = static_cast<double>(rand()) / RAND_MAX;  // NOLINT
    return min + f * (max - min);
  }
};

/**
 * determines the rotation axis necessary to rotate from frame b1 to the
 * orientation of frame b2 and the vector necessary to translate the origin
 * of b1 to the origin of b2, and stores the result in a Twist
 * datastructure.  The result is w.r.t. frame b1.
 * \param F_a_b1 frame b1 expressed with respect to some frame a.
 * \param F_a_b2 frame b2 expressed with respect to some frame a.
 * \warning The result is not a real Twist!
 * \warning In contrast to standard KDL diff methods, the result of
 * diffRelative is w.r.t. frame b1 instead of frame a.
 */
IMETHOD Twist diffRelative(const Frame & F_a_b1, const Frame & F_a_b2, double dt = 1)
{
  return Twist(
    F_a_b1.M.Inverse() * diff(F_a_b1.p, F_a_b2.p, dt),
    F_a_b1.M.Inverse() * diff(F_a_b1.M, F_a_b2.M, dt));
}

} 

namespace NLOPT_IK
{

enum OptType { Joint, DualQuat, SumSq, L2 };


class NLOPT_IK
{
  friend class IRS_IK::IRS_IK;

public:
  NLOPT_IK(
    const KDL::Chain & chain, const KDL::JntArray & q_min, const KDL::JntArray & q_max,
    double maxtime = 0.005, double eps = 1e-3, OptType type = SumSq);

  ~NLOPT_IK() {}
  int CartToJnt(
    const KDL::JntArray & q_init, const KDL::Frame & p_in, KDL::JntArray & q_out,
    const KDL::Twist bounds = KDL::Twist::Zero(),
    const KDL::JntArray & q_desired = KDL::JntArray());

  double minJoints(const std::vector<double> & x, std::vector<double> & grad);
  //  void cartFourPointError(const std::vector<double>& x, double error[]);
  void cartSumSquaredError(const std::vector<double> & x, double error[]);
  void cartDQError(const std::vector<double> & x, double error[]);
  void cartL2NormError(const std::vector<double> & x, double error[]);

  inline void setMaxtime(double t)
  {
    maxtime = std::chrono::duration<double>(t);
  }

private:
  inline void abort()
  {
    aborted = true;
  }

  inline void reset()
  {
    aborted = false;
  }


  std::vector<double> lb;
  std::vector<double> ub;

  const KDL::Chain chain;
  std::vector<double> des;


  KDL::ChainFkSolverPos_recursive fksolver;

  std::chrono::duration<double> maxtime;
  double eps;
  int iter_counter;
  OptType TYPE;

  KDL::Frame targetPose;
  KDL::Frame z_up;
  KDL::Frame x_out;
  KDL::Frame y_out;
  KDL::Frame z_target;
  KDL::Frame x_target;
  KDL::Frame y_target;

  std::vector<KDL::BasicJointType> types;

  nlopt::opt opt;

  KDL::Frame currentPose;

  std::vector<double> best_x;
  int progress;
  bool aborted;

  KDL::Twist bounds;

  inline static double fRand(double min, double max)
  {
    double f = static_cast<double>(rand()) / RAND_MAX;  // NOLINT
    return min + f * (max - min);
  }
};

} 

namespace IRS_IK
{

enum SolveType { Speed, Distance, Manip1, Manip2 };

class  IRS_IK
{
public:
  
  IRS_IK(
    const KDL::Chain & _chain, const KDL::JntArray & _q_min, const KDL::JntArray & _q_max,
    double _maxtime = 0.005, double _eps = 1e-5, SolveType _type = Speed);

  
  IRS_IK(
    const std::string & base_link, const std::string & tip_link,
    const std::string & urdf_xml = "", double _maxtime = 0.005, double _eps = 1e-5,
    SolveType _type = Speed);

  
  ~IRS_IK();

  
  bool getKDLChain(KDL::Chain & chain_)
  {
    chain_ = chain;
    return initialized;
  }

  
  bool getKDLLimits(KDL::JntArray & lb_, KDL::JntArray & ub_)
  {
    lb_ = lb;
    ub_ = ub;
    return initialized;
  }

  
  // Requires a previous call to CartToJnt()
  bool getSolutions(std::vector<KDL::JntArray> & solutions_)
  {
    solutions_ = solutions;
    return initialized && !solutions.empty();
  }

  
  bool getSolutions(
    std::vector<KDL::JntArray> & solutions_, std::vector<std::pair<double,
    uint>> & errors_)
  {
    errors_ = errors;
    return getSolutions(solutions_);
  }

  
  bool setKDLLimits(KDL::JntArray & lb_, KDL::JntArray & ub_)
  {
    lb = lb_;
    ub = ub_;
    nl_solver.reset(new NLOPT_IK::NLOPT_IK(chain, lb, ub, maxtime.count(), eps, NLOPT_IK::SumSq));
    iksolver.reset(new KDL::ChainIkSolverPos_TL(chain, lb, ub, maxtime.count(), eps, true, true));
    return true;
  }

  
  static double JointErr(const KDL::JntArray & arr1, const KDL::JntArray & arr2)
  {
    double err = 0;
    for (uint i = 0; i < arr1.data.size(); i++) {
      err += pow(arr1(i) - arr2(i), 2);
    }

    return err;
  }

  
  int CartToJnt(
    const KDL::JntArray & q_init, const KDL::Frame & p_in, KDL::JntArray & q_out,
    const KDL::Twist & bounds = KDL::Twist::Zero());

  inline void SetSolveType(SolveType _type)
  {
    solvetype = _type;
  }

  inline SolveType GetSolveType() const
  {
    return solvetype;
  }

private:
  bool initialized;
  KDL::Chain chain;
  KDL::JntArray lb, ub;
  std::unique_ptr<KDL::ChainJntToJacSolver> jacsolver;
  double eps;
  std::chrono::duration<double> maxtime;
  SolveType solvetype;

  std::unique_ptr<NLOPT_IK::NLOPT_IK> nl_solver;
  std::unique_ptr<KDL::ChainIkSolverPos_TL> iksolver;

  std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> start_time;

  template<typename T1, typename T2>
  bool runSolver(
    T1 & solver, T2 & other_solver,
    const KDL::JntArray & q_init,
    const KDL::Frame & p_in);

  bool runKDL(const KDL::JntArray & q_init, const KDL::Frame & p_in);
  bool runNLOPT(const KDL::JntArray & q_init, const KDL::Frame & p_in);

  void normalize_seed(const KDL::JntArray & seed, KDL::JntArray & solution);
  void normalize_limits(const KDL::JntArray & seed, KDL::JntArray & solution);

  std::vector<KDL::BasicJointType> types;

  std::mutex mtx_;
  std::vector<KDL::JntArray> solutions;
  std::vector<std::pair<double, uint>> errors;

  std::thread task1, task2;
  KDL::Twist bounds;

  bool unique_solution(const KDL::JntArray & sol);

  inline static double fRand(double min, double max)
  {
    double f = static_cast<double>(rand()) / RAND_MAX;  // NOLINT
    return min + f * (max - min);
  }


  double manipPenalty(const KDL::JntArray &);
  double ManipValue1(const KDL::JntArray &);
  double ManipValue2(const KDL::JntArray &);

  inline bool myEqual(const KDL::JntArray & a, const KDL::JntArray & b)
  {
    return (a.data - b.data).isZero(1e-4);
  }

  void initialize();
};

inline bool IRS_IK::runKDL(const KDL::JntArray & q_init, const KDL::Frame & p_in)
{
  return runSolver(*iksolver.get(), *nl_solver.get(), q_init, p_in);
}

inline bool IRS_IK::runNLOPT(const KDL::JntArray & q_init, const KDL::Frame & p_in)
{
  return runSolver(*nl_solver.get(), *iksolver.get(), q_init, p_in);
}

} 
#endif
