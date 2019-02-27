#include "trainingSet.h"
#include "neuron.h"
#include "net.h"

void showVectorVals(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

int main()
{
	trainingSet trainingData("testData.txt");
	std::vector<unsigned> topology;
	trainingData.getTopology(topology);
	net net(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while (!trainingData.isEOF())
	{
		++trainingPass;
		std::cout << std::endl << "Pass: " << trainingPass << std::endl;

		if (trainingData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals("Input:", inputVals);
		net.feedForward(inputVals);

		trainingData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		net.getResults(resultVals);
		showVectorVals("Outputs", resultVals);

		net.backProp(targetVals);

		std::cout << "Net average error: " << net.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;
#if defined(_MSC_VER) || defined(_WIN32)
	system("PAUSE");
#endif
	return(0);
}