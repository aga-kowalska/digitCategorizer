package digitCategorization;

import java.util.Random;

public class MLP {

	public final static int NEURONS_INPUT_LAYER = 64;
	public final static int NEURONS_HIDDEN_LAYER = 15;
	public final static int NEURONS_OUTPUT_LAYER = 10;

	public double[] biasesL2 = new double[NEURONS_HIDDEN_LAYER];
	public double[] biasesL3 = new double[NEURONS_OUTPUT_LAYER];

	public double[][] weights1to2 = new double[NEURONS_HIDDEN_LAYER][NEURONS_INPUT_LAYER];
	public double[][] weights2to3 = new double[NEURONS_OUTPUT_LAYER][NEURONS_HIDDEN_LAYER];

	public void init() {

		// biases and weights randomly initialised for hidden and output layers
		Random random = new Random();

		for (int bias = 0; bias < biasesL2.length; bias++) {
			biasesL2[bias] = random.nextGaussian();

		}

		for (int bias = 0; bias < biasesL3.length; bias++) {
			biasesL3[bias] = random.nextGaussian();

		}

		for (int neuron = 0; neuron < weights1to2.length; neuron++) {
			for (int weight = 0; weight < weights1to2[neuron].length; weight++) {
				weights1to2[neuron][weight] = random.nextGaussian();

			}
		}

		for (int neuron = 0; neuron < weights2to3.length; neuron++) {
			for (int weight = 0; weight < weights2to3[neuron].length; weight++) {
				weights2to3[neuron][weight] = random.nextGaussian();

			}
		}

	}

	public static double logistic(double input) {
		return 1.0 / (1.0 + Math.exp(-input));
	}

	public static double[] feed(int currentLayerNeurons, double[] outputPreviousLayer, double[][] weigths,
			double[] biases) {
		
		double[] currentLayerOutput = new double[currentLayerNeurons];
		
		//for every neuron in the current layer: add weighted inputs, add bias and apply logistic function
		for (int neuron = 0; neuron < currentLayerNeurons; neuron++) {
			for (int input = 0; input < outputPreviousLayer.length; input++) {
				currentLayerOutput[neuron] += outputPreviousLayer[input] * weigths[neuron][input];
			}
			currentLayerOutput[neuron] += biases[neuron];
			currentLayerOutput[neuron] = logistic(currentLayerOutput[neuron]);
		}
		
		return currentLayerOutput;
	}
	
	//loss function
	public static double loss(int category, double[] actualOutput) {
		double[] expectedOutput = new double[NEURONS_OUTPUT_LAYER];
		expectedOutput[category] = 1.0;
		
		
	}
	
	public static double euclideanDistance(double[] expectedOutput, double[] actualOutput) {
		double distance = 0.0;
		
		double sum = 0.0;
		
		for (int output = 0; output < expectedOutput.length; output++) {
			sum += Math.pow(expectedOutput[output] - actualOutput[output], 2);
		}
		
		distance = Math.sqrt(sum);
		
		return distance;
	}

}
