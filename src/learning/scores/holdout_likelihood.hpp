#ifndef PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/holdout_adaptator.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <learning/scores/scores.hpp>

using dataset::HoldOut;
using models::GaussianNetwork, models::SemiparametricBN;
using learning::scores::ScoreSPBN;

namespace learning::scores {

    class HoldoutLikelihood : public ScoreSPBN {
    public:
        HoldoutLikelihood(const DataFrame& df, double test_ratio) : m_holdout(df, test_ratio) { }
        HoldoutLikelihood(const DataFrame& df, double test_ratio, unsigned int seed) : m_holdout(df, test_ratio, seed) { }

        double local_score(const BayesianNetworkBase& model, int variable) const override {
            return local_score(model, model.name(variable));
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            auto parents = model.parents(variable);
            return local_score(model, variable, parents);
        }

        double local_score(const BayesianNetworkBase& model, int variable, const std::vector<int>& evidence) const override {
            std::vector<std::string> evidence_str;
            for (auto ev : evidence) {
                evidence_str.push_back(model.name(ev));
            }

            return local_score(model, model.name(variable), evidence_str);
        }

        double local_score(const BayesianNetworkBase& model,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;

        double local_score(FactorType variable_type,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;


        const DataFrame& training_data() const { return m_holdout.training_data(); }
        const DataFrame& test_data() const { return m_holdout.test_data(); }

        const HoldOut& holdout() { return m_holdout; }

        std::string ToString() const override {
            return "HoldoutLikelihood";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::HOLDOUT_LIKELIHOOD;
        }
        
        bool compatible_bn(const BayesianNetworkBase& model) const override{
            return m_holdout.training_data().has_columns(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return m_holdout.training_data().has_columns(model.all_nodes());
        }
    private:
        HoldOut m_holdout;
    };

    using DynamicHoldoutLikelihood = DynamicScoreAdaptator<HoldoutLikelihood>;
}

#endif //PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP