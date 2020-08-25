#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct
{
	unsigned int inputs; // number of inputs
	unsigned int outputs; // number of outputs
	unsigned int layers; // number of layers
	unsigned int* layerSizes; // layer sizes

	float*** weights; // weights of each connection
	float** biases; // biases of each neuron
} Network;

// ml_initialize: create Network struct
Network* ml_initialize (unsigned int inputs, unsigned int outputs, unsigned int layers, unsigned int* layerSizes)
{
	Network* network = calloc (1, sizeof (Network)); // allocate and initialize memory for struct
	network -> inputs = inputs; // set inputs
	network -> outputs = outputs; // set outputs
	network -> layers = layers; // set layers
	network -> layerSizes = calloc (layers, sizeof (unsigned int)); // set layer sizes memory address
	network -> weights = calloc (layers + 1, sizeof (float**)); // set weights memory address
	network -> biases = calloc (layers + 1, sizeof (float*)); // set biases memory address

	for (unsigned int i = 0; i <= layers; i++) // go through each layer
	{
		network -> layerSizes[i] = layerSizes[i]; // set layer size

		unsigned int prevLayerSize = (i > 0 ? layerSizes[i - 1] : inputs); // determine connections layer size
		unsigned int layerSize = (i == layers ? outputs : layerSizes[i]); // determine current layer size

		network -> weights[i] = calloc (layerSize, sizeof (float**)); // set layer
		network -> biases[i] = calloc (layerSize, sizeof (float*)); // set layer

		for (unsigned int j = 0; j < layerSize; j++) // go through each neuron
			network -> weights[i][j] = calloc (prevLayerSize, sizeof (float)); // set neuron connections
	}

	return network;
}

// ml_terminate: delete all memory used by network
void ml_terminate (Network* network)
{
	for (unsigned int i = 0; i < network -> layers; i++) // go through each layer
	{
		free (network -> biases[i]); // free bias layer
		for (unsigned int j = 0; j < network -> layerSizes[i]; j++) // go through each neuron
			free (network -> weights[i][j]); // free neuron connection weights
		free (network -> weights[i]); // free layer connection pointers
	}

	free (network -> biases); // free bias pointers
	free (network -> weights); // free weight points

	free (network -> layerSizes); // free layer size pointers

	free (network); // free memory used by network
}

// ml_fill: fill weights and biases with a constant value
void ml_fill (float value, Network* network)
{	
	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine current layer size
		for (unsigned int j = 0; j < layerSize; j++) // go through each neuron
		{
			network -> biases[i][j] = value; // fill with random value

			unsigned int prevLayerSize = (i > 0 ? network -> layerSizes[i - 1] : network -> inputs); // determine layer size
			for (unsigned int k = 0; k < prevLayerSize; k++) // go through each connection
				network -> weights[i][j][k] = (float) rand() / RAND_MAX; // fill with random value
		}
	}
}

// ml_fill_random: fill weights and biases with random values
void ml_fill_random (unsigned int seed, Network* network)
{
	srand (seed);
	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine current layer size
		for (unsigned int j = 0; j < layerSize; j++) // go through each neuron
		{
			network -> biases[i][j] = (float) rand() / RAND_MAX; // fill with random value

			unsigned int prevLayerSize = (i > 0 ? network -> layerSizes[i - 1] : network -> inputs); // determine layer size
			for (unsigned int k = 0; k < prevLayerSize; k++) // go through each connection
				network -> weights[i][j][k] = (float) rand() / RAND_MAX; // fill with random value
		}
	}
}

// ml_save_network: save network to binary file
void ml_save_network (const char* fileName, Network* network)
{
	FILE* file = fopen (fileName, "wb"); // open file to write binary

	fwrite (&network -> inputs, sizeof (unsigned int), 1, file); // write inputs to file
	fwrite (&network -> outputs, sizeof (unsigned int), 1, file); // write outputs to file
	fwrite (&network -> layers, sizeof (unsigned int), 1, file); // write layers to file
	fwrite (&network -> layerSizes[0], sizeof (unsigned int), network -> layers, file); // write layer sizes to file

	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine current layer size
		for (unsigned int j = 0; j < layerSize; j++) // go through each neuron
		{
			unsigned int prevLayerSize = (i > 0 ? network -> layerSizes[i - 1] : network -> inputs); // determine layer size
			fwrite (&network -> weights[i][j][0], sizeof (unsigned int), prevLayerSize, file); // write weights layer to file
		}
	}

	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs: network -> layerSizes[i]); // determine current layer size
		fwrite (&network -> biases[i][0], sizeof (unsigned int), layerSize, file); // write biases layer to file
	}

	fclose (file); // close file
}

// ml_load_network: load network from a saved binary file
Network* ml_load_network (const char* fileName)
{
	FILE* file = fopen (fileName, "rb"); // open file to read binary
	Network* network = calloc (1, sizeof (Network)); // allocate memory for network

	fread (&network -> inputs, sizeof (unsigned int), 1, file); // read inputs from file
	fread (&network -> outputs, sizeof (unsigned int), 1, file); // read outputs from file
	fread (&network -> layers, sizeof (unsigned int), 1, file); // read layers from memory

	network -> layerSizes = calloc (network -> layers, sizeof (unsigned int)); // allocate memory for array of layer sizes
	fread (&network -> layerSizes[0], sizeof (unsigned int), network -> layers, file); // read layer sizes from memory
	
	network -> weights = calloc (network -> layers + 1, sizeof (float**)); // allocate memory for weights

	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine current layer size
		network -> weights[i] = calloc (layerSize, sizeof (float*));
		
		for (unsigned int j = 0; j < layerSize; j++) // go through each neuron
		{
			unsigned int prevLayerSize = (i > 0 ? network -> layerSizes[i - 1] : network -> inputs); // determine previous layer size

			network -> weights[i][j] = calloc (prevLayerSize, sizeof (float)); // allocate memory for layer of connections
			fread (&network -> weights[i][j][0], sizeof (float), prevLayerSize, file); // read layer from file
		}
	}

	network -> biases = calloc (network -> layers + 1, sizeof (float*)); // allocate memory for biases

	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine current layer size
		network -> biases[i] = calloc (layerSize, sizeof (float)); // allocate memory for layer

		fread (&network -> biases[i][0], layerSize, sizeof (float), file); // read layer of biases from file
	}

	return network; // return structure
}

// ml_test: passes inputs through the network and returns the outputs
float* ml_test (float* inputs, Network* network)
{
	float* outputs = calloc (network -> outputs, sizeof (float)); // allocate memory for outputs
	float** neurons = calloc (network -> layers + 1, sizeof (float*)); // allocate memory for pointers to each layer

	for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
	{
		unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine layer size
		neurons[i] = calloc (layerSize, sizeof (float)); // allocate and initialize memory for current layer

		for (unsigned int j = 0; j < layerSize; j++) // go through each neuron
		{
			unsigned int prevLayerSize = (i > 0 ? network -> layerSizes[i - 1] : network -> inputs); // determine previous layer size
			for (unsigned int k = 0; k < prevLayerSize; k++) // go through each connection
				neurons[i][j] += network -> weights[i][j][k] * (i == 0 ? inputs[k] : neurons[i - 1][k]); // change neuron relative to the weight
			neurons[i][j] += network -> biases[i][j]; // change neuron by bias
		}

		if (i > 0) // if there is a previous layer
			free (neurons[i - 1]); // free the previous layer
	}

	memcpy (&outputs[0], &neurons[network -> layers][0], sizeof (float) * network -> outputs); // copy memory from last layer to outputs
	
	free (neurons[network -> layers]); // free last layer
	free (neurons); // free pointers

	return outputs; // return outputs
}

// ml_cost: calculate cost for outputs compared to the expected values
float ml_cost (float* output, float* expected, Network* network)
{
	float cost = 0.0f; // set default cost to 0
	for (unsigned int i = 0; i < network -> outputs; i++) // go through each output
		cost += (output[i] - expected[i]) * (output[i] - expected[i]); // change cost
	return cost; // return cost
}
