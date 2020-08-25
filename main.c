#include "ml.h"
#include <time.h>

#include <stdio.h>
#include <stdlib.h>

int main ()
{
	unsigned int layerSizes[] = { 16, 16 }; // layer sizes
	Network* network = ml_initialize(784, 10, 2, layerSizes); // initialize network
	
	ml_fill_random (time (NULL), network); // fill network with random values
	ml_save_network("network.bin.old", network);

	printf ("Starting...\n"); // indicate that training has started

	FILE* labels = fopen ("labels", "rb"); // open labels file
	FILE* images = fopen ("images", "rb"); // open images file

	unsigned int iterations = 10; // number of iterations to do

	srand (time (NULL)); // seed random
	unsigned int startTime = time (NULL); // get current time in seconds
	for (unsigned int i = 0; i < iterations; i++) // training iterations
	{
		unsigned int st = time (NULL); // get current time in seconds

		float increment = iterations - (i * i) / 4 > i / i * i ? iterations - (i * i) / 4 : i / i * i; // determine the increment
		unsigned int index = rand() % 60000; // pick random image

		fseek (labels, sizeof (int) * 2 + index * sizeof (unsigned char), SEEK_SET); // seek

		unsigned char number; // what number we're expecting
		fread (&number, 1, 1, labels); // read expected number from labels file
		
		float* expected = calloc (10, sizeof (float)); // create expected values
		for (unsigned int j = 0; j < 10; j++) expected[j] = number == i ? 1.0f : 0.0f; // set expected values

		unsigned char image_ub[28][28]; // create image ub memory
		float image[28][28]; // create image memory

		fseek (images, sizeof (int) * 4 + 28 * 28 * sizeof (unsigned char) * index, SEEK_SET); // seek
		fread (&image_ub, sizeof (unsigned char), 784, images); // read image ub from file

		for (unsigned int x = 0; x < 28; x++) // go through each row
			for (unsigned int y = 0; y < 28; y++) // go through each column
				image[x][y] = (float) image_ub[x][y] / 255.0f; // set pixel

		// next block determines whether going positive or negative positively influences the outputs
		for (unsigned int i = 0; i <= network -> layers; i++) // go through each layer
		{
			unsigned int layerSize = (i == network -> layers ? network -> outputs : network -> layerSizes[i]); // determine current layer size
			for (unsigned int j = 0; j < layerSize; j++)
			{
				unsigned int prevLayerSize = (i > 0 ? network -> layerSizes[i - 1] : network -> inputs); // determine previous  layer size
				for (unsigned int k = 0; k < prevLayerSize; k++) // go through each connection
				{
					// test negative first
					network -> weights[i][j][k] -= increment; // decrease by increment
					float* negOutput = ml_test (&image[0][0], network); // get negative output
					float negCost = ml_cost (negOutput, expected, network); // find negative cost
					free (negOutput); // free memory used by outputs

					// test positive
					network -> weights[i][j][k] += increment * 2; // increase by increment * 2
					float* posOutput = ml_test (&image[0][0], network); // get positive output
					float posCost = ml_cost (posOutput, expected, network); // find positive cost
					free (posOutput); // free memory used by outputs

					if (posOutput > negOutput) network -> weights[i][j][k] -= increment * 2; // revert back to negative state if the cost is lower
				}

				// test negative first
				network -> biases[i][j] -= increment; // decrease by increment
				float* negOutput = ml_test (&image[0][0], network); // get negative output
				float negCost = ml_cost (negOutput, expected, network); // find negative cost
				free (negOutput); // free memory used by outputs

				// test positive
				network -> biases[i][j] += increment * 2; // increase by increment * 2
				float* posOutput = ml_test (&image[0][0], network); // get positive output
				float posCost = ml_cost (posOutput, expected, network); // find positive cost
				free (posOutput); // free memory used by outputs

				if (posOutput > negOutput) network -> biases[i][j] -= increment * 2; // revert back to negative state if the cost is lower
			}
		}

		float* output = ml_test (&image[0][0], network); // get outputs
		float cost = ml_cost (output, expected, network); // get cost

		unsigned char picked = 0; // picked number
		for (unsigned int i = 0; i < 10; i++) // go through each number 0-9
			if (output[i] > output[picked]) // if this is higher than the current one
				picked = i; // set picked number

		printf("expected: %i, output: ", number); // print
		for (unsigned int i = 0; i < 10; i ++) printf("%f ", output[i]); // print current node
		printf("(%lis)\n", time (NULL) - st); // second elapsed & newline

		free (output); // free memory used by outputs
		free (expected); // free memory used by expected values
	}

	printf("Done (%lis)\n", time (NULL) - startTime); // indicate that training has finished

	ml_save_network ("network.bin", network); // save network to file
	ml_terminate (network); // free memory

	return 0;
}
