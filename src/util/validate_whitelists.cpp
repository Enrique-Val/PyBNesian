#include <util/validate_whitelists.hpp>

using graph::PartiallyDirectedGraph;

namespace util {


    ListRestrictions check_whitelists(const PartiallyDirectedGraph& g,
                                      const ArcStringVector& varc_blacklist, 
                                      const ArcStringVector& varc_whitelist,
                                      const EdgeStringVector& vedge_blacklist,
                                      const EdgeStringVector& vedge_whitelist) {
        
        ListRestrictions r;

        for (const auto& edge : vedge_blacklist) {
            r.edge_blacklist.insert({g.index(edge.first), g.index(edge.second)});
        }

        for (const auto& edge : vedge_whitelist) {
            auto e1 = g.index(edge.first);
            auto e2 = g.index(edge.second);
            
            if (r.edge_blacklist.count({e1, e2}) > 0) {
                throw std::invalid_argument("Edge " + edge.first + " -- " + edge.second 
                                            + " in blacklist and whitelist");
            }

            r.edge_whitelist.insert({g.index(edge.first), g.index(edge.second)});
        }

        for (const auto& arc : varc_whitelist) {
            auto s = g.index(arc.first);
            auto t = g.index(arc.second);
            
            // Edge blacklist + Arc whitelist =  Not possible
            if (r.edge_blacklist.count({s, t}) > 0) {
                throw std::invalid_argument("Edge blacklist " + arc.first + " -- " + arc.second 
                                            + " is incompatible with arc whitelist" + arc.first + " -> " + arc.second);
            }
            
            // Edge whitelist + Arc whitelist = Arc whitelist
            if (r.edge_whitelist.count({s, t}) > 0) {
                r.edge_whitelist.erase({s, t});
            }

            r.arc_whitelist.insert({g.index(arc.first), g.index(arc.second)});
        }


        for (const auto& arc : varc_blacklist) {
            auto s = g.index(arc.first);
            auto t = g.index(arc.second);

            // Arc whitelist + Arc blacklist = Not possible
            if (r.arc_whitelist.count({s, t}) > 0) {
                throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second 
                                            + " in blacklist and whitelist");
            }

            // Edge whitelist + Arc blacklist = Arc whitelist in opposite direction.
            if (r.edge_whitelist.count({s, t}) > 0) {
                r.arc_whitelist.insert({t, s});
                r.edge_whitelist.erase({s, t});
            } 
            
            // Edge blacklist + Arc blacklist  = Edge blacklist -> do nothing.
            if (r.edge_blacklist.count({s, t}) == 0)
                r.arc_blacklist.insert({g.index(arc.first), g.index(arc.second)});
        }

        // PartiallyDirectedGraph acyclic()
        return r;
    }
}