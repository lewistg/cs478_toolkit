#ifndef _ID3TREEPLOTTER_H_
#define _ID3TREEPLOTTER_H_

#include <fstream>
#include <map>

class ID3Node;

namespace ID3TreePlot
{
	void plotTree(const ID3Node& root);

	void explore(const ID3Node& node, std::ofstream& treePlot, std::map<size_t, 
            std::string>& nodeAnnot, std::map<std::string, std::string>& edgeAnnot);
};

#endif