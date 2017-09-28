#include "trainingSet.h"
#include "neuron.h"
#include "net.h"


double net::recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

net::net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];


		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			layers.back().push_back(neuron(numOutputs, neuronNum));
		}

		layers.back().back().setOutputVal(1.0);
	}
}


void net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		resultVals.push_back(layers.back()[n].getOutputVal());
	}
}

void net::backProp(const std::vector<double> &targetVals)
{

	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1; 
	error = sqrt(error); 


	recentAverageError =
		(recentAverageError * recentAverageSmoothingFactor + error)
		/ (recentAverageSmoothingFactor + 1.0);


	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}


	for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void net::feedForward(const std::vector<double> &inputVals)
{
	assert(inputVals.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); ++i) {
		layers[0][i].setOutputVal(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < layers.size(); ++layerNum) {
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n) {
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

