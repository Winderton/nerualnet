#pragma once
#include <vector>

class neuron;

typedef std::vector<neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class neuron
{
public:
	neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { outputVal = val; }
	double getOutputVal() const { return outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta;
	static double alpha;
	static double randomWeight() { return rand() / double(RAND_MAX); }
	static double activationFunction(double x);
	static double activationFunctionDerivative(double x);
	double sumDOW(const Layer &nextLayer) const;
	double outputVal;
	std::vector<Connection> outputWeights;
	unsigned m_myIndex;
	double gradient;
};