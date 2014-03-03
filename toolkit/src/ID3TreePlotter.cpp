#include <string>
#include <sstream>
#include "ID3TreePlotter.h"
#include "ID3Node.h"

void ID3TreePlot::plotTree(const ID3Node& root)
{
    std::ofstream treePlot("tree_plot.py", std::ios::trunc);
	treePlot << "import networkx as nx" << std::endl;
    treePlot << "import matplotlib.pyplot as plt" << std::endl << std::endl;
	treePlot << "G = nx.Graph()" << std::endl;
    treePlot << "labels = {}" << std::endl;

	std::map<size_t, std::string> nodeAnnot;
	explore(root, treePlot, nodeAnnot);

	treePlot << "pos=nx.graphviz_layout(G,prog='dot')" << std::endl;
	treePlot << "nx.draw(G, pos, arrows = False, with_labels=False)" << std::endl;
	treePlot << "for key in pos:" << std::endl;
    treePlot << "\tpos[key] = (pos[key][0] + 10, pos[key][1] + 5)" << std::endl;

	for(std::map<size_t, std::string>::iterator itr = nodeAnnot.begin(); itr != nodeAnnot.end(); itr++)
	{
		//treePlot << "plt.text(pos[key][0], pos[key][1], 'hi there', family='serif', style='italic', ha='left')" << std::endl;
		treePlot << "plt.text(pos[" << itr->first << "][0], pos[" << itr->first << "][1], " 
				<< itr->second << ", family='serif', style='italic', ha='left')" << std::endl;
	}

	treePlot << "nx.draw_networkx_labels(G,pos,labels)" << std::endl;
	treePlot << "plt.show()" << std::endl;
}

namespace
{
	std::string getNodeLabel(const ID3Node& node, size_t nodeId)
	{
		std::stringstream ss;

		if(node.getTargetAttr() != -1)
			ss << "\"splits on:" << node.getTargetAttr() << "-" << node.getTargetAttrName() << "\"";
		else
			ss << "\"assigns label: " << node.getLabelToAssign() << "-" << node.getLabelToAssignName() << "\"";

		return ss.str();
	}	
}

void ID3TreePlot::explore(const ID3Node& node, std::ofstream& treePlot, std::map<size_t, std::string>& nodeAnnot)
{
	static size_t nodeCounter = 0;
	nodeCounter += 1;

	int nodeId = nodeCounter;
	treePlot << "G.add_node(" << nodeId << ")" << std::endl;
	nodeAnnot[nodeId] = getNodeLabel(node, nodeId);
	/*if(node.getTargetAttr() != -1)
	{
		treePlot << "labels[" << nodeId << "] = \" splits on:" 
				<< node.getTargetAttr() << "-" << node.getTargetAttrName() << "\"" << std::endl;
	}
	else
	{
		treePlot << "labels[" << nodeId << "] = \"" 
				<< "assigns: " << node.getLabelToAssign() << "-" << node.getLabelToAssignName() << "\"" << std::endl;
	}*/

	for(size_t i = 0; i < node.getNumChildNodes(); i++)	
	{
		treePlot << "G.add_edge(" << nodeId << ", " << nodeCounter + 1 << ")" << std::endl;
		explore(node.getChildNode(i), treePlot, nodeAnnot);
	}
}