#ifndef PGM_DATASET_INDEPENDENCE_HPP
#define PGM_DATASET_INDEPENDENCE_HPP

namespace learning::independences {


    class IndependenceTest {
    public:
        virtual ~IndependenceTest() {};
        virtual double pvalue(int v1, int v2) const = 0;
        virtual double pvalue(const std::string& v1, const std::string& v2) const = 0;

        virtual double pvalue(int v1, int v2, int cond) const = 0;
        virtual double pvalue(const std::string& v1, const std::string& v2, const std::string& cond) const = 0;

        virtual double pvalue(int v1, int v2, 
                        const typename std::vector<int>::const_iterator evidence_begin, 
                        const typename std::vector<int>::const_iterator evidence_end) const = 0;

        virtual double pvalue(const std::string& v1, const std::string& v2, 
                        const typename std::vector<std::string>::const_iterator evidence_begin, 
                        const typename std::vector<std::string>::const_iterator evidence_end) const = 0;
    };
}

#endif //PGM_DATASET_LINEARCORRELATION_HPP