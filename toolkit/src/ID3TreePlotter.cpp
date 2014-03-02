#include <fstream>
#include <map>
#include "ID3TreePlotter.h"
#include "ID3Node.h"

void ID3TreePlot::plotTree(const ID3Node& root)
{
    std::ofstream treePlot("tree_plot.py", std::ios::trunc);
	treePlot << "import networkx as nx" << std::endl;
    treePlot << "import matplotlib.pyplot as plt" << std::endl << std::endl;
	treePlot << "G = nx.Graph()" << std::endl;
    treePlot << "labels = {}" << std::endl;
	explore(root, treePlot);
	treePlot << "pos=nx.graphviz_layout(G,prog='dot')" << std::endl;
	treePlot << "nx.draw(G, pos, arrows = False, with_labels=False)" << std::endl;
	treePlot << "for key in pos:" << std::endl;
    treePlot << "\tpos[key] = (pos[key][0] + 10, pos[key][1] + 5)" << std::endl;
	treePlot << "nx.draw_networkx_labels(G,pos,labels)" << std::endl;
	treePlot << "plt.show()" << std::endl;
}

void ID3TreePlot::explore(const ID3Node& node, std::ofstream& treePlot)
{
	static size_t nodeCounter = 0;
	nodeCounter += 1;

	int nodeId = nodeCounter;
	treePlot << "G.add_node(" << nodeId << ")" << std::endl;
    treePlot << "labels[" << nodeId << "] = " << node._targetAttr << std::endl;
	for(size_t i = 0; i < node._attrToNode.size(); i++)	
	{
		treePlot << "G.add_edge(" << nodeId << ", " << nodeCounter + 1 << ")" << std::endl;
		explore(node._attrToNode[i], treePlot);
	}
}