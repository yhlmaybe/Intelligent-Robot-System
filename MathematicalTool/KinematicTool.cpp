#include <KinematicTool.h>

namespace KDL
{
    ChainIkSolverPos_TL::ChainIkSolverPos_TL(
        const Chain &_chain, const JntArray &_q_min,
        const JntArray &_q_max, double _maxtime, double _eps,
        bool _random_restart, bool _try_jl_wrap)
        : chain(_chain), q_min(_q_min), q_max(_q_max), vik_solver(chain), fksolver(chain), delta_q(
                                                                                               chain.getNrOfJoints()),
          maxtime(std::chrono::duration<double>(_maxtime)), eps(_eps), rr(_random_restart), wrap(
                                                                                                _try_jl_wrap)
    {
        assert(chain.getNrOfJoints() == _q_min.data.size());
        assert(chain.getNrOfJoints() == _q_max.data.size());

        reset();

        for (uint i = 0; i < chain.segments.size(); i++)
        {
            std::string type = chain.segments[i].getJoint().getTypeName();
            if (type.find("Rot") != std::string::npos)
            {
                if (_q_max(types.size()) >= std::numeric_limits<float>::max() &&
                    _q_min(types.size()) <= std::numeric_limits<float>::lowest())
                {
                    types.push_back(KDL::BasicJointType::Continuous);
                }
                else
                {
                    types.push_back(KDL::BasicJointType::RotJoint);
                }
            }
            else if (type.find("Trans") != std::string::npos)
            {
                types.push_back(KDL::BasicJointType::TransJoint);
            }
        }

        assert(types.size() == static_cast<long unsigned int>(_q_max.data.size())); // NOLINT
    }

    int ChainIkSolverPos_TL::CartToJnt(
        const KDL::JntArray &q_init, const KDL::Frame &p_in,
        KDL::JntArray &q_out, const KDL::Twist _bounds)
    {
        if (aborted)
        {
            return -3;
        }

        std::chrono::duration<double> timediff;
        std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> start_time(
            std::chrono::system_clock::now());
        q_out = q_init;
        bounds = _bounds;

        do
        {
            fksolver.JntToCart(q_out, f);
            delta_twist = diffRelative(p_in, f);

            if (std::abs(delta_twist.vel.x()) <= std::abs(bounds.vel.x()))
            {
                delta_twist.vel.x(0);
            }

            if (std::abs(delta_twist.vel.y()) <= std::abs(bounds.vel.y()))
            {
                delta_twist.vel.y(0);
            }

            if (std::abs(delta_twist.vel.z()) <= std::abs(bounds.vel.z()))
            {
                delta_twist.vel.z(0);
            }

            if (std::abs(delta_twist.rot.x()) <= std::abs(bounds.rot.x()))
            {
                delta_twist.rot.x(0);
            }

            if (std::abs(delta_twist.rot.y()) <= std::abs(bounds.rot.y()))
            {
                delta_twist.rot.y(0);
            }

            if (std::abs(delta_twist.rot.z()) <= std::abs(bounds.rot.z()))
            {
                delta_twist.rot.z(0);
            }

            if (Equal(delta_twist, Twist::Zero(), eps))
            {
                return 1;
            }

            delta_twist = diff(f, p_in);

            vik_solver.CartToJnt(q_out, delta_twist, delta_q);
            KDL::JntArray q_curr;

            Add(q_out, delta_q, q_curr);

            for (unsigned int j = 0; j < q_min.data.size(); j++)
            {
                if (types[j] == KDL::BasicJointType::Continuous)
                {
                    continue;
                }
                if (q_curr(j) < q_min(j))
                {
                    if (!wrap || types[j] == KDL::BasicJointType::TransJoint)
                    {
                        // KDL's default
                        q_curr(j) = q_min(j);
                    }
                    else
                    {
                        // Find actual wrapped angle between limit and joint
                        double diffangle = fmod(q_min(j) - q_curr(j), 2 * M_PI);
                        // Subtract that angle from limit and go into the range by a
                        // revolution
                        double curr_angle = q_min(j) - diffangle + 2 * M_PI;
                        if (curr_angle > q_max(j))
                        {
                            q_curr(j) = q_min(j);
                        }
                        else
                        {
                            q_curr(j) = curr_angle;
                        }
                    }
                }
            }

            for (unsigned int j = 0; j < q_max.data.size(); j++)
            {
                if (types[j] == KDL::BasicJointType::Continuous)
                {
                    continue;
                }

                if (q_curr(j) > q_max(j))
                {
                    if (!wrap || types[j] == KDL::BasicJointType::TransJoint)
                    {
                        // KDL's default
                        q_curr(j) = q_max(j);
                    }
                    else
                    {
                        // Find actual wrapped angle between limit and joint
                        double diffangle = fmod(q_curr(j) - q_max(j), 2 * M_PI);
                        // Add that angle to limit and go into the range by a revolution
                        double curr_angle = q_max(j) + diffangle - 2 * M_PI;
                        if (curr_angle < q_min(j))
                        {
                            q_curr(j) = q_max(j);
                        }
                        else
                        {
                            q_curr(j) = curr_angle;
                        }
                    }
                }
            }

            Subtract(q_out, q_curr, q_out);

            if (q_out.data.isZero(std::numeric_limits<float>::epsilon()))
            {
                if (rr)
                {
                    for (unsigned int j = 0; j < q_out.data.size(); j++)
                    {
                        if (types[j] == KDL::BasicJointType::Continuous)
                        {
                            q_curr(j) = fRand(q_curr(j) - 2 * M_PI, q_curr(j) + 2 * M_PI);
                        }
                        else
                        {
                            q_curr(j) = fRand(q_min(j), q_max(j));
                        }
                    }
                }

                // Below would be an optimization to the normal KDL, where when it
                // gets stuck, it returns immediately.  Don't use to compare KDL with
                // random restarts or TRAC-IK to plain KDL.

                // else {
                //   q_out=q_curr;
                //   return -3;
                // }
            }

            q_out = q_curr;

            timediff = std::chrono::system_clock::now() - start_time;
        } while (timediff < maxtime && !aborted);

        return -3;
    }

    ChainIkSolverPos_TL::~ChainIkSolverPos_TL()
    {
    }

}

namespace NLOPT_IK
{
    Math3D::dual_quaternion targetDQ;

    double minfunc(const std::vector<double> &x, std::vector<double> &grad, void *data)
    {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.

        NLOPT_IK *c = reinterpret_cast<NLOPT_IK *>(data);

        return c->minJoints(x, grad);
    }

    double minfuncDQ(const std::vector<double> &x, std::vector<double> &grad, void *data)
    {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.
        NLOPT_IK *c = reinterpret_cast<NLOPT_IK *>(data);

        std::vector<double> vals(x);

        double jump = std::numeric_limits<float>::epsilon();
        double result[1];
        c->cartDQError(vals, result);

        if (!grad.empty())
        {
            double v1[1];
            for (uint i = 0; i < x.size(); i++)
            {
                double original = vals[i];

                vals[i] = original + jump;
                c->cartDQError(vals, v1);

                vals[i] = original;
                grad[i] = (v1[0] - result[0]) / (2 * jump);
            }
        }

        return result[0];
    }

    double minfuncSumSquared(const std::vector<double> &x, std::vector<double> &grad, void *data)
    {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.

        NLOPT_IK *c = reinterpret_cast<NLOPT_IK *>(data);

        std::vector<double> vals(x);

        double jump = std::numeric_limits<float>::epsilon();
        double result[1];
        c->cartSumSquaredError(vals, result);

        if (!grad.empty())
        {
            double v1[1];
            for (uint i = 0; i < x.size(); i++)
            {
                double original = vals[i];

                vals[i] = original + jump;
                c->cartSumSquaredError(vals, v1);

                vals[i] = original;
                grad[i] = (v1[0] - result[0]) / (2.0 * jump);
            }
        }

        return result[0];
    }

    double minfuncL2(const std::vector<double> &x, std::vector<double> &grad, void *data)
    {
        // Auxilory function to minimize (Sum of Squared joint angle error
        // from the requested configuration).  Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.

        NLOPT_IK *c = reinterpret_cast<NLOPT_IK *>(data);

        std::vector<double> vals(x);

        double jump = std::numeric_limits<float>::epsilon();
        double result[1];
        c->cartL2NormError(vals, result);

        if (!grad.empty())
        {
            double v1[1];
            for (uint i = 0; i < x.size(); i++)
            {
                double original = vals[i];

                vals[i] = original + jump;
                c->cartL2NormError(vals, v1);

                vals[i] = original;
                grad[i] = (v1[0] - result[0]) / (2.0 * jump);
            }
        }

        return result[0];
    }

    void constrainfuncm(uint m, double *result, uint n, const double *x, double *grad, void *data)
    {
        // Equality constraint auxilary function for Euclidean distance .
        // This also uses a small walk to approximate the gradient of the
        // constraint function at the current joint angles.

        NLOPT_IK *c = reinterpret_cast<NLOPT_IK *>(data);

        std::vector<double> vals(n);

        for (uint i = 0; i < n; i++)
        {
            vals[i] = x[i];
        }

        double jump = std::numeric_limits<float>::epsilon();

        c->cartSumSquaredError(vals, result);

        if (grad != NULL)
        {
            std::vector<double> v1(m);
            for (uint i = 0; i < n; i++)
            {
                double o = vals[i];
                vals[i] = o + jump;
                c->cartSumSquaredError(vals, v1.data());
                vals[i] = o;
                for (uint j = 0; j < m; j++)
                {
                    grad[j * n + i] = (v1[j] - result[j]) / (2 * jump);
                }
            }
        }
    }

    NLOPT_IK::NLOPT_IK(
        const KDL::Chain &_chain, const KDL::JntArray &_q_min,
        const KDL::JntArray &_q_max, double _maxtime, double _eps, OptType _type)
        : chain(_chain), fksolver(chain), maxtime(std::chrono::duration<double>(_maxtime)),
          eps(std::abs(_eps)), TYPE(_type)
    {
        assert(chain.getNrOfJoints() == _q_min.data.size());
        assert(chain.getNrOfJoints() == _q_max.data.size());

        // Constructor for an IK Class.  Takes in a Chain to operate on,
        // the min and max joint limits, an (optional) maximum number of
        // iterations, and an (optional) desired error.
        reset();

        if (chain.getNrOfJoints() < 2)
        {

            IRS_MESSAGE(

                "NLOpt_IK can only be run for chains of length 2 or more");
            return;
        }
        opt = nlopt::opt(nlopt::LD_SLSQP, _chain.getNrOfJoints());

        for (uint i = 0; i < chain.getNrOfJoints(); i++)
        {
            lb.push_back(_q_min(i));
            ub.push_back(_q_max(i));
        }

        for (uint i = 0; i < chain.segments.size(); i++)
        {
            std::string type = chain.segments[i].getJoint().getTypeName();
            if (type.find("Rot") != std::string::npos)
            {
                if (_q_max(types.size()) >= std::numeric_limits<float>::max() &&
                    _q_min(types.size()) <= std::numeric_limits<float>::lowest())
                {
                    types.push_back(KDL::BasicJointType::Continuous);
                }
                else
                {
                    types.push_back(KDL::BasicJointType::RotJoint);
                }
            }
            else if (type.find("Trans") != std::string::npos)
            {
                types.push_back(KDL::BasicJointType::TransJoint);
            }
        }

        assert(types.size() == lb.size());

        std::vector<double> tolerance(1, std::numeric_limits<float>::epsilon());
        opt.set_xtol_abs(tolerance[0]);

        switch (TYPE)
        {
        case Joint:
            opt.set_min_objective(minfunc, this);
            opt.add_equality_mconstraint(constrainfuncm, this, tolerance);
            break;
        case DualQuat:
            opt.set_min_objective(minfuncDQ, this);
            break;
        case SumSq:
            opt.set_min_objective(minfuncSumSquared, this);
            break;
        case L2:
            opt.set_min_objective(minfuncL2, this);
            break;
        }
    }

    double NLOPT_IK::minJoints(const std::vector<double> &x, std::vector<double> &grad)
    {
        // Actual function to compute the error between the current joint
        // configuration and the desired.  The SSE is easy to provide a
        // closed form gradient for.

        bool gradient = !grad.empty();

        double err = 0;
        for (uint i = 0; i < x.size(); i++)
        {
            err += pow(x[i] - des[i], 2);
            if (gradient)
            {
                grad[i] = 2.0 * (x[i] - des[i]);
            }
        }

        return err;
    }

    void NLOPT_IK::cartSumSquaredError(const std::vector<double> &x, double error[])
    {
        // Actual function to compute Euclidean distance error.  This uses
        // the KDL Forward Kinematics solver to compute the Cartesian pose
        // of the current joint configuration and compares that to the
        // desired Cartesian pose for the IK solve.

        if (aborted || progress != -3)
        {
            opt.force_stop();
            return;
        }

        KDL::JntArray q(x.size());

        for (uint i = 0; i < x.size(); i++)
        {
            q(i) = x[i];
        }

        int rc = fksolver.JntToCart(q, currentPose);

        if (rc < 0)
        {
            IRS_MESSAGE("KDL FKSolver is failing ");
        }

        if (std::isnan(currentPose.p.x()))
        {
            IRS_MESSAGE("NaNs from NLOpt!!");
            error[0] = std::numeric_limits<float>::max();
            progress = -1;
            return;
        }

        KDL::Twist delta_twist = KDL::diffRelative(targetPose, currentPose);

        for (int i = 0; i < 6; i++)
        {
            if (std::abs(delta_twist[i]) <= std::abs(bounds[i]))
            {
                delta_twist[i] = 0.0;
            }
        }

        error[0] =
            KDL::dot(delta_twist.vel, delta_twist.vel) + KDL::dot(delta_twist.rot, delta_twist.rot);

        if (KDL::Equal(delta_twist, KDL::Twist::Zero(), eps))
        {
            progress = 1;
            best_x = x;
            return;
        }
    }

    void NLOPT_IK::cartL2NormError(const std::vector<double> &x, double error[])
    {
        // Actual function to compute Euclidean distance error.  This uses
        // the KDL Forward Kinematics solver to compute the Cartesian pose
        // of the current joint configuration and compares that to the
        // desired Cartesian pose for the IK solve.

        if (aborted || progress != -3)
        {
            opt.force_stop();
            return;
        }

        KDL::JntArray q(x.size());

        for (uint i = 0; i < x.size(); i++)
        {
            q(i) = x[i];
        }

        int rc = fksolver.JntToCart(q, currentPose);

        if (rc < 0)
        {
            IRS_MESSAGE("KDL FKSolver is failing ");
        }

        if (std::isnan(currentPose.p.x()))
        {
            IRS_MESSAGE("NaNs from NLOpt!!");
            error[0] = std::numeric_limits<float>::max();
            progress = -1;
            return;
        }

        KDL::Twist delta_twist = KDL::diffRelative(targetPose, currentPose);

        for (int i = 0; i < 6; i++)
        {
            if (std::abs(delta_twist[i]) <= std::abs(bounds[i]))
            {
                delta_twist[i] = 0.0;
            }
        }

        error[0] =
            std::sqrt(
                KDL::dot(
                    delta_twist.vel,
                    delta_twist.vel) +
                KDL::dot(delta_twist.rot, delta_twist.rot));

        if (KDL::Equal(delta_twist, KDL::Twist::Zero(), eps))
        {
            progress = 1;
            best_x = x;
            return;
        }
    }

    void NLOPT_IK::cartDQError(const std::vector<double> &x, double error[])
    {
        // Actual function to compute Euclidean distance error.  This uses
        // the KDL Forward Kinematics solver to compute the Cartesian pose
        // of the current joint configuration and compares that to the
        // desired Cartesian pose for the IK solve.

        if (aborted || progress != -3)
        {
            opt.force_stop();
            return;
        }

        KDL::JntArray q(x.size());

        for (uint i = 0; i < x.size(); i++)
        {
            q(i) = x[i];
        }

        int rc = fksolver.JntToCart(q, currentPose);

        if (rc < 0)
        {
            IRS_MESSAGE("KDL FKSolver is failing ");
        }

        if (std::isnan(currentPose.p.x()))
        {
            IRS_MESSAGE("NaNs from NLOpt!!");
            error[0] = std::numeric_limits<float>::max();
            progress = -1;
            return;
        }

        KDL::Twist delta_twist = KDL::diffRelative(targetPose, currentPose);

        for (int i = 0; i < 6; i++)
        {
            if (std::abs(delta_twist[i]) <= std::abs(bounds[i]))
            {
                delta_twist[i] = 0.0;
            }
        }

        Math3D::matrix3x3<double> currentRotationMatrix(currentPose.M.data);
        Math3D::quaternion<double> currentQuaternion = Math3D::rot_matrix_to_quaternion<double>(
            currentRotationMatrix);
        Math3D::point3d currentTranslation(currentPose.p.data);
        Math3D::dual_quaternion currentDQ = Math3D::dual_quaternion::rigid_transformation(
            currentQuaternion,
            currentTranslation);

        Math3D::dual_quaternion errorDQ = (currentDQ * !targetDQ).normalize();
        errorDQ.log();
        error[0] = 4.0f * Math3D::dot(errorDQ, errorDQ);

        if (KDL::Equal(delta_twist, KDL::Twist::Zero(), eps))
        {
            progress = 1;
            best_x = x;
            return;
        }
    }

    int NLOPT_IK::CartToJnt(
        const KDL::JntArray &q_init, const KDL::Frame &p_in,
        KDL::JntArray &q_out, const KDL::Twist _bounds,
        const KDL::JntArray &q_desired)
    {
        // User command to start an IK solve.  Takes in a seed
        // configuration, a Cartesian pose, and (optional) a desired
        // configuration.  If the desired is not provided, the seed is
        // used.  Outputs the joint configuration found that solves the
        // IK.

        // Returns -3 if a configuration could not be found within the eps
        // set up in the constructor.

        std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> start_time(
            std::chrono::system_clock::now());

        bounds = _bounds;
        q_out = q_init;

        if (chain.getNrOfJoints() < 2)
        {

            IRS_MESSAGE(

                "NLOpt_IK can only be run for chains of length 2 or more");
            return -3;
        }

        if (static_cast<long unsigned int>(q_init.data.size()) != types.size())
        { // NOLINT

            IRS_MESSAGE(

                "IK seeded with wrong number of joints.  Expected %d but got %d",
                (int)types.size(), (int)q_init.data.size());
            return -3;
        }

        opt.set_maxtime(maxtime.count());

        double minf; /* the minimum objective value, upon return */

        targetPose = p_in;

        if (TYPE == 1)
        { // DQ
            Math3D::matrix3x3<double> targetRotationMatrix(targetPose.M.data);
            Math3D::quaternion<double> targetQuaternion = Math3D::rot_matrix_to_quaternion<double>(
                targetRotationMatrix);
            Math3D::point3d targetTranslation(targetPose.p.data);
            targetDQ = Math3D::dual_quaternion::rigid_transformation(targetQuaternion, targetTranslation);
        }
        // else if (TYPE == 1)
        // {
        //   z_target = targetPose*z_up;
        //   x_target = targetPose*x_out;
        //   y_target = targetPose*y_out;
        // }

        //    fksolver.JntToCart(q_init,currentPose);

        std::vector<double> x(chain.getNrOfJoints());

        for (uint i = 0; i < x.size(); i++)
        {
            x[i] = q_init(i);

            if (types[i] == KDL::BasicJointType::Continuous)
            {
                continue;
            }

            if (types[i] == KDL::BasicJointType::TransJoint)
            {
                x[i] = std::min(x[i], ub[i]);
                x[i] = std::max(x[i], lb[i]);
            }
            else
            {
                // Below is to handle bad seeds outside of limits

                if (x[i] > ub[i])
                {
                    // Find actual angle offset
                    double diffangle = fmod(x[i] - ub[i], 2 * M_PI);
                    // Add that to upper bound and go back a full rotation
                    x[i] = ub[i] + diffangle - 2 * M_PI;
                }

                if (x[i] < lb[i])
                {
                    // Find actual angle offset
                    double diffangle = fmod(lb[i] - x[i], 2 * M_PI);
                    // Subtract that from lower bound and go forward a full rotation
                    x[i] = lb[i] - diffangle + 2 * M_PI;
                }

                if (x[i] > ub[i])
                {
                    x[i] = (ub[i] + lb[i]) / 2.0;
                }
            }
        }

        best_x = x;
        progress = -3;

        std::vector<double> artificial_lower_limits(lb.size());

        for (uint i = 0; i < lb.size(); i++)
        {
            if (types[i] == KDL::BasicJointType::Continuous)
            {
                artificial_lower_limits[i] = best_x[i] - 2 * M_PI;
            }
            else if (types[i] == KDL::BasicJointType::TransJoint)
            {
                artificial_lower_limits[i] = lb[i];
            }
            else
            {
                artificial_lower_limits[i] = std::max(lb[i], best_x[i] - 2 * M_PI);
            }
        }

        opt.set_lower_bounds(artificial_lower_limits);

        std::vector<double> artificial_upper_limits(lb.size());

        for (uint i = 0; i < ub.size(); i++)
        {
            if (types[i] == KDL::BasicJointType::Continuous)
            {
                artificial_upper_limits[i] = best_x[i] + 2 * M_PI;
            }
            else if (types[i] == KDL::BasicJointType::TransJoint)
            {
                artificial_upper_limits[i] = ub[i];
            }
            else
            {
                artificial_upper_limits[i] = std::min(ub[i], best_x[i] + 2 * M_PI);
            }
        }

        opt.set_upper_bounds(artificial_upper_limits);

        if (q_desired.data.size() == 0)
        {
            des = x;
        }
        else
        {
            des.resize(x.size());
            for (uint i = 0; i < des.size(); i++)
            {
                des[i] = q_desired(i);
            }
        }

        try
        {
            opt.optimize(x, minf);
        }
        catch (...)
        {
        }

        if (progress == -1)
        { // Got NaNs
            progress = -3;
        }

        if (!aborted && progress < 0)
        {
            std::chrono::duration<double> diff(std::chrono::system_clock::now() - start_time);

            while (diff < maxtime && !aborted && progress < 0)
            {
                for (uint i = 0; i < x.size(); i++)
                {
                    x[i] = fRand(artificial_lower_limits[i], artificial_upper_limits[i]);
                }

                opt.set_maxtime((maxtime - diff).count());

                try
                {
                    opt.optimize(x, minf);
                }
                catch (...)
                {
                }

                if (progress == -1)
                { // Got NaNs
                    progress = -3;
                }

                diff = std::chrono::system_clock::now() - start_time;
            }
        }

        for (uint i = 0; i < x.size(); i++)
        {
            q_out(i) = best_x[i];
        }

        return progress;
    }

}

namespace IRS_IK
{
    IRS_IK::IRS_IK(
        const std::string &base_link, const std::string &tip_link,
        const std::string &urdf_xml, double _maxtime, double _eps, SolveType _type)
        : initialized(false),
          eps(_eps),
          maxtime(std::chrono::duration<double>(_maxtime)),
          solvetype(_type)
    {
        urdf::Model robot_model;
        if (!robot_model.initString(urdf_xml))
        {
            IRS_MESSAGE("Unable to initialize urdf::Model from robot description.");
        }

        IRS_MESSAGE("Reading joints and links from URDF");

        KDL::Tree tree;

        if (!kdl_parser::treeFromUrdfModel(robot_model, tree))
        {
            IRS_MESSAGE("Failed to extract kdl tree from xml robot description");
        }

        if (!tree.getChain(base_link, tip_link, chain))
        {
            IRS_MESSAGE("Couldn't find chain ");
        }

        std::vector<KDL::Segment> chain_segs = chain.segments;

        urdf::JointConstSharedPtr joint;

        std::vector<double> l_bounds, u_bounds;

        lb.resize(chain.getNrOfJoints());
        ub.resize(chain.getNrOfJoints());

        uint joint_num = 0;
        for (unsigned int i = 0; i < chain_segs.size(); ++i)
        {
            joint = robot_model.getJoint(chain_segs[i].getJoint().getName());
            if (joint->type != urdf::Joint::UNKNOWN && joint->type != urdf::Joint::FIXED)
            {
                joint_num++;
                float lower, upper;
                int hasLimits;
                if (joint->type != urdf::Joint::CONTINUOUS)
                {
                    if (joint->safety)
                    {
                        lower = std::max(joint->limits->lower, joint->safety->soft_lower_limit);
                        upper = std::min(joint->limits->upper, joint->safety->soft_upper_limit);
                    }
                    else
                    {
                        lower = joint->limits->lower;
                        upper = joint->limits->upper;
                    }
                    hasLimits = 1;
                }
                else
                {
                    hasLimits = 0;
                }
                if (hasLimits)
                {
                    lb(joint_num - 1) = lower;
                    ub(joint_num - 1) = upper;
                }
                else
                {
                    lb(joint_num - 1) = std::numeric_limits<float>::lowest();
                    ub(joint_num - 1) = std::numeric_limits<float>::max();
                }
            }
        }

        initialize();
    }

    IRS_IK::IRS_IK(
        const KDL::Chain &_chain, const KDL::JntArray &_q_min,
        const KDL::JntArray &_q_max, double _maxtime, double _eps, SolveType _type)
        : initialized(false),
          chain(_chain),
          lb(_q_min),
          ub(_q_max),
          eps(_eps),
          maxtime(std::chrono::duration<double>(_maxtime)),
          solvetype(_type)
    {
        initialize();
    }

    void IRS_IK::initialize()
    {
        assert(chain.getNrOfJoints() == lb.data.size());
        assert(chain.getNrOfJoints() == ub.data.size());

        jacsolver.reset(new KDL::ChainJntToJacSolver(chain));
        nl_solver.reset(new NLOPT_IK::NLOPT_IK(chain, lb, ub, maxtime.count(), eps, NLOPT_IK::SumSq));
        iksolver.reset(new KDL::ChainIkSolverPos_TL(chain, lb, ub, maxtime.count(), eps, true, true));

        for (uint i = 0; i < chain.segments.size(); i++)
        {
            std::string type = chain.segments[i].getJoint().getTypeName();
            if (type.find("Rot") != std::string::npos)
            {
                if (ub(types.size()) >= std::numeric_limits<float>::max() &&
                    lb(types.size()) <= std::numeric_limits<float>::lowest())
                {
                    types.push_back(KDL::BasicJointType::Continuous);
                }
                else
                {
                    types.push_back(KDL::BasicJointType::RotJoint);
                }
            }
            else if (type.find("Trans") != std::string::npos)
            {
                types.push_back(KDL::BasicJointType::TransJoint);
            }
        }

        assert(types.size() == static_cast<long unsigned int>(lb.data.size())); // NOLINT

        initialized = true;
    }

    bool IRS_IK::unique_solution(const KDL::JntArray &sol)
    {
        for (uint i = 0; i < solutions.size(); i++)
        {
            if (myEqual(sol, solutions[i]))
            {
                return false;
            }
        }
        return true;
    }

    inline void normalizeAngle(double &val, const double &min, const double &max)
    {
        if (val > max)
        {
            // Find actual angle offset
            double diffangle = fmod(val - max, 2 * M_PI);
            // Add that to upper bound and go back a full rotation
            val = max + diffangle - 2 * M_PI;
        }

        if (val < min)
        {
            // Find actual angle offset
            double diffangle = fmod(min - val, 2 * M_PI);
            // Add that to upper bound and go back a full rotation
            val = min - diffangle + 2 * M_PI;
        }
    }

    inline void normalizeAngle(double &val, const double &target)
    {
        double new_target = target + M_PI;
        if (val > new_target)
        {
            // Find actual angle offset
            double diffangle = fmod(val - new_target, 2 * M_PI);
            // Add that to upper bound and go back a full rotation
            val = new_target + diffangle - 2 * M_PI;
        }

        new_target = target - M_PI;
        if (val < new_target)
        {
            // Find actual angle offset
            double diffangle = fmod(new_target - val, 2 * M_PI);
            // Add that to upper bound and go back a full rotation
            val = new_target - diffangle + 2 * M_PI;
        }
    }

    template <typename T1, typename T2>
    bool IRS_IK::runSolver(
        T1 &solver, T2 &other_solver,
        const KDL::JntArray &q_init,
        const KDL::Frame &p_in)
    {
        KDL::JntArray q_out;

        std::chrono::duration<double> fulltime(maxtime);
        KDL::JntArray seed = q_init;

        while (true)
        {
            std::chrono::duration<double> timediff(std::chrono::system_clock::now() - start_time);

            if (timediff >= fulltime)
            {
                break;
            }

            solver.setMaxtime((fulltime - timediff).count());

            int RC = solver.CartToJnt(seed, p_in, q_out, bounds);
            if (RC >= 0)
            {
                switch (solvetype)
                {
                case Manip1:
                case Manip2:
                    normalize_limits(q_init, q_out);
                    break;
                default:
                    normalize_seed(q_init, q_out);
                    break;
                }
                mtx_.lock();
                if (unique_solution(q_out))
                {
                    solutions.push_back(q_out);
                    uint curr_size = solutions.size();
                    errors.resize(curr_size);
                    mtx_.unlock();
                    double err, penalty;
                    switch (solvetype)
                    {
                    case Manip1:
                        penalty = manipPenalty(q_out);
                        err = penalty * IRS_IK::ManipValue1(q_out);
                        break;
                    case Manip2:
                        penalty = manipPenalty(q_out);
                        err = penalty * IRS_IK::ManipValue2(q_out);
                        break;
                    default:
                        err = IRS_IK::JointErr(q_init, q_out);
                        break;
                    }
                    mtx_.lock();
                    errors[curr_size - 1] = std::make_pair(err, curr_size - 1);
                }
                mtx_.unlock();
            }

            if (!solutions.empty() && solvetype == Speed)
            {
                break;
            }

            for (unsigned int j = 0; j < seed.data.size(); j++)
            {
                if (types[j] == KDL::BasicJointType::Continuous)
                {
                    seed(j) = fRand(q_init(j) - 2 * M_PI, q_init(j) + 2 * M_PI);
                }
                else
                {
                    seed(j) = fRand(lb(j), ub(j));
                }
            }
        }
        other_solver.abort();

        solver.setMaxtime(fulltime.count());

        return true;
    }

    void IRS_IK::normalize_seed(const KDL::JntArray &seed, KDL::JntArray &solution)
    {
        // Make sure rotational joint values are within 1 revolution of seed; then
        // ensure joint limits are met.

        for (uint i = 0; i < lb.data.size(); i++)
        {
            if (types[i] == KDL::BasicJointType::TransJoint)
            {
                continue;
            }

            double target = seed(i);
            double val = solution(i);

            normalizeAngle(val, target);

            if (types[i] == KDL::BasicJointType::Continuous)
            {
                solution(i) = val;
                continue;
            }

            normalizeAngle(val, lb(i), ub(i));

            solution(i) = val;
        }
    }

    void IRS_IK::normalize_limits(const KDL::JntArray &seed, KDL::JntArray &solution)
    {
        // Make sure rotational joint values are within 1 revolution of middle of
        // limits; then ensure joint limits are met.

        for (uint i = 0; i < lb.data.size(); i++)
        {
            if (types[i] == KDL::BasicJointType::TransJoint)
            {
                continue;
            }

            double target = seed(i);

            if (types[i] == KDL::BasicJointType::RotJoint && types[i] != KDL::BasicJointType::Continuous)
            {
                target = (ub(i) + lb(i)) / 2.0;
            }

            double val = solution(i);

            normalizeAngle(val, target);

            if (types[i] == KDL::BasicJointType::Continuous)
            {
                solution(i) = val;
                continue;
            }

            normalizeAngle(val, lb(i), ub(i));

            solution(i) = val;
        }
    }

    double IRS_IK::manipPenalty(const KDL::JntArray &arr)
    {
        double penalty = 1.0;
        for (uint i = 0; i < arr.data.size(); i++)
        {
            if (types[i] == KDL::BasicJointType::Continuous)
            {
                continue;
            }
            double range = ub(i) - lb(i);
            penalty *= ((arr(i) - lb(i)) * (ub(i) - arr(i)) / (range * range));
        }
        return std::max(0.0, 1.0 - exp(-1 * penalty));
    }

    double IRS_IK::ManipValue1(const KDL::JntArray &arr)
    {
        KDL::Jacobian jac(arr.data.size());

        jacsolver->JntToJac(arr, jac);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdsolver(jac.data);
        Eigen::MatrixXd singular_values = svdsolver.singularValues();

        double error = 1.0;
        for (unsigned int i = 0; i < singular_values.rows(); ++i)
        {
            error *= singular_values(i, 0);
        }
        return error;
    }

    double IRS_IK::ManipValue2(const KDL::JntArray &arr)
    {
        KDL::Jacobian jac(arr.data.size());

        jacsolver->JntToJac(arr, jac);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdsolver(jac.data);
        Eigen::MatrixXd singular_values = svdsolver.singularValues();

        return singular_values.minCoeff() / singular_values.maxCoeff();
    }

    int IRS_IK::CartToJnt(
        const KDL::JntArray &q_init, const KDL::Frame &p_in, KDL::JntArray &q_out,
        const KDL::Twist &_bounds)
    {
        if (!initialized)
        {
            return -1;
        }

        start_time = std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>>(
            std::chrono::system_clock::now());

        nl_solver->reset();
        iksolver->reset();

        solutions.clear();
        errors.clear();

        bounds = _bounds;

        task1 = std::thread(&IRS_IK::runKDL, this, q_init, p_in);
        task2 = std::thread(&IRS_IK::runNLOPT, this, q_init, p_in);

        task1.join();
        task2.join();

        if (solutions.empty())
        {
            q_out = q_init;
            return -3;
        }

        switch (solvetype)
        {
        case Manip1:
        case Manip2:
            std::sort(errors.rbegin(), errors.rend()); // rbegin/rend to sort by max
            break;
        default:
            std::sort(errors.begin(), errors.end());
            break;
        }

        q_out = solutions[errors[0].second];

        return solutions.size();
    }

    IRS_IK::~IRS_IK()
    {
        if (task1.joinable())
        {
            task1.join();
        }
        if (task2.joinable())
        {
            task2.join();
        }
    }
}
