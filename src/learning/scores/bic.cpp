#include <learning/scores/bic.hpp>

namespace learning::scores {
   
    double BIC::local_score(const BayesianNetworkBase& model,
                            const std::string& variable,
                            const std::vector<std::string>& evidence) const {
        
        switch (model.type()) {
            case BayesianNetworkType::Gaussian: {
                MLE<LinearGaussianCPD> mle;

                auto mle_params = mle.estimate(m_df, variable, evidence);

                auto rows = m_df.valid_rows(variable, evidence);
                auto num_evidence = evidence.size();
                auto loglik = 0.5 * (1 + static_cast<double>(num_evidence) - static_cast<double>(rows)) 
                                - 0.5 * rows*std::log(2*util::pi<double>) 
                                - rows * std::log(std::sqrt(mle_params.variance));

                return loglik - std::log(rows) * 0.5 * (num_evidence + 2);
            }
            default:
               throw std::invalid_argument("Bayesian network type " + 
                    models::BayesianNetworkType_ToString(model.type()) + " not valid for score BIC");
        }

    }
}