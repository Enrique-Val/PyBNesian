#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_whitelists.hpp>
#include <util/validate_options.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;;
using models::GaussianNetwork, models::BayesianNetworkType;
using learning::scores::ScoreType, learning::scores::BIC, learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood;
using learning::operators::OperatorSet, learning::operators::OperatorSetType, 
      learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet;

using util::ArcStringVector;


namespace learning::algorithms {

    std::unique_ptr<BayesianNetworkBase> hc(const DataFrame& df,
                                            const BayesianNetworkBase* start,
                                            const std::string& bn_str,
                                            const std::optional<std::string>& score_str,
                                            const std::optional<std::vector<std::string>>& operators_str,
                                            const ArcStringVector& arc_blacklist,
                                            const ArcStringVector& arc_whitelist,
                                            const FactorStringTypeVector& type_whitelist,
                                            int max_indegree,
                                            int max_iters,
                                            double epsilon,
                                            int patience,
                                            std::optional<unsigned int> seed,
                                            int num_folds,
                                            double test_holdout_ratio,
                                            int verbose) {
        
        auto iseed = [seed]() {
            if (seed) return *seed;
            else return std::random_device{}();
        }();

        auto bn_type = [start, &bn_str]() {
            if (start) return start->type();
            else return util::check_valid_bn_string(bn_str);
        }();

        auto score_type = util::check_valid_score_string(score_str, bn_type);
        auto operators_type = util::check_valid_operators_string(operators_str, bn_type);

        auto operators = util::check_valid_operators(bn_type, operators_type, 
                                arc_blacklist, arc_whitelist, max_indegree, type_whitelist);
        
        if (max_iters == 0) max_iters = std::numeric_limits<int>::max();

        std::unique_ptr<BayesianNetworkBase> created_start_model = [start, bn_type, &df]() -> std::unique_ptr<BayesianNetworkBase> {
            if (!start) {
                switch (bn_type) {
                    case BayesianNetworkType::Gaussian:
                        return std::make_unique<GaussianNetwork>(df.column_names());
                    case BayesianNetworkType::Semiparametric:
                        return std::make_unique<SemiparametricBN>(df.column_names());
                    default:
                        throw std::invalid_argument("Wrong BayesianNetwork type. Unreachable code!");
                }
            } else {
                return nullptr;
            }
        }();

        const auto start_model = [start, &created_start_model]() -> const BayesianNetworkBase* {
            if (start) return start;
            else return created_start_model.get();
        }();

        GreedyHillClimbing hc;

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            HoldoutLikelihood validation_score(df, test_holdout_ratio, iseed);
            auto score = util::check_valid_score(validation_score.training_data(), 
                            bn_type, score_type, iseed, num_folds, test_holdout_ratio);
    
            return hc.estimate_validation(*operators, *score, validation_score, *start_model, arc_blacklist, arc_whitelist, 
                                        type_whitelist, max_indegree, max_iters, epsilon, patience, verbose);

        } else {
            auto score = util::check_valid_score(df, bn_type, score_type, iseed, num_folds, test_holdout_ratio);

            return hc.estimate(*operators, *score, *start_model, arc_blacklist, arc_whitelist,
                                max_indegree, max_iters, epsilon, verbose);
        }
    }

    std::unique_ptr<BayesianNetworkBase> GreedyHillClimbing::estimate(OperatorSet& op_set,
                                                                      Score& score,
                                                                      const BayesianNetworkBase& start,
                                                                      const ArcStringVector& arc_blacklist,
                                                                      const ArcStringVector& arc_whitelist,
                                                                      int max_indegree,
                                                                      int max_iters,
                                                                      double epsilon,
                                                                      int verbose) {
        auto restrictions = util::validate_restrictions(start, arc_blacklist, arc_whitelist);

        return estimate_hc(op_set,
                           score,
                           start,
                           restrictions.arc_blacklist,
                           restrictions.arc_whitelist,
                           max_indegree,
                           max_iters,
                           epsilon,
                           verbose);
    }

    std::unique_ptr<ConditionalBayesianNetworkBase> GreedyHillClimbing::estimate(
                                                        OperatorSet& op_set,
                                                        Score& score,
                                                        const ConditionalBayesianNetworkBase& start,
                                                        const ArcStringVector& arc_blacklist,
                                                        const ArcStringVector& arc_whitelist,
                                                        int max_indegree,
                                                        int max_iters, 
                                                        double epsilon,
                                                        int verbose) {
        auto restrictions = util::validate_restrictions(start, arc_blacklist, arc_whitelist);

        return estimate_hc(op_set,
                           score,
                           start,
                           restrictions.arc_blacklist,
                           restrictions.arc_whitelist,
                           max_indegree,
                           max_iters,
                           epsilon,
                           verbose);
    }

    std::unique_ptr<BayesianNetworkBase> GreedyHillClimbing::estimate_validation(OperatorSet& op_set,
                                                                                 Score& score,
                                                                                 Score& validation_score,
                                                                                 const BayesianNetworkBase& start,
                                                                                 const ArcStringVector& arc_blacklist,
                                                                                 const ArcStringVector& arc_whitelist,
                                                                                 const FactorStringTypeVector& type_whitelist,
                                                                                 int max_indegree,
                                                                                 int max_iters,
                                                                                 double epsilon, 
                                                                                 int patience,
                                                                                 int verbose) {
        auto restrictions = util::validate_restrictions(start, arc_blacklist, arc_whitelist);
        util::check_node_type_list(start, type_whitelist);

        return estimate_validation_hc(op_set,
                                      score,
                                      validation_score,
                                      start,
                                      restrictions.arc_blacklist,
                                      restrictions.arc_whitelist,
                                      type_whitelist,
                                      max_indegree,
                                      max_iters,
                                      epsilon,
                                      patience,
                                      verbose);

    }

    std::unique_ptr<BayesianNetworkBase> GreedyHillClimbing::estimate_validation(OperatorSet& op_set,
                                                                                 Score& score,
                                                                                 Score& validation_score,
                                                                                 const ConditionalBayesianNetworkBase& start,
                                                                                 const ArcStringVector& arc_blacklist,
                                                                                 const ArcStringVector& arc_whitelist,
                                                                                 const FactorStringTypeVector& type_whitelist,
                                                                                 int max_indegree,
                                                                                 int max_iters,
                                                                                 double epsilon, 
                                                                                 int patience,
                                                                                 int verbose) {
        auto restrictions = util::validate_restrictions(start, arc_blacklist, arc_whitelist);
        util::check_node_type_list(start, type_whitelist);

        return estimate_validation_hc(op_set,
                                      score,
                                      validation_score,
                                      start,
                                      restrictions.arc_blacklist,
                                      restrictions.arc_whitelist,
                                      type_whitelist,
                                      max_indegree,
                                      max_iters,
                                      epsilon,
                                      patience,
                                      verbose);

    }
}