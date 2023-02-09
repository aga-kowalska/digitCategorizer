package digitCategorization;

import java.util.Random;

/*UNDONE
try mini-batches 
 * */

public class MLP {

	public final static int NEURONS_INPUT_LAYER = 64;
	public final static int NEURONS_HIDDEN_LAYER = 15;
	public final static int NEURONS_OUTPUT_LAYER = 10;

	public double[] biasesL2 = new double[NEURONS_HIDDEN_LAYER];
	public double[] biasesL3 = new double[NEURONS_OUTPUT_LAYER];

	public double[][] weights1to2 = new double[NEURONS_HIDDEN_LAYER][DigitCategorizer.INPUT_DIMENSION];
	public double[][] weights2to3 = new double[NEURONS_OUTPUT_LAYER][NEURONS_HIDDEN_LAYER];

	public final static double LEARNING_RATE = 0.15;

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

	// compute the activation of every neuron in a given layer
	public static double[] feedForwardLayer(int currentLayerNeurons, double[] outputPreviousLayer, double[][] weigths,
			double[] biases, boolean isFirst) {
		double[] currentLayerOutput = new double[currentLayerNeurons];
		// for every neuron in the current layer: add weighted inputs, add bias and
		// apply logistic function
		for (int neuron = 0; neuron < currentLayerNeurons; neuron++) {
			// if it is the first layer skip the last feature (it is a category label)
			if (isFirst) {
				for (int input = 0; input < DigitCategorizer.INPUT_DIMENSION; input++) {
					currentLayerOutput[neuron] += outputPreviousLayer[input] * weigths[neuron][input];
				}
			} else {
				for (int input = 0; input < outputPreviousLayer.length; input++) {
					currentLayerOutput[neuron] += outputPreviousLayer[input] * weigths[neuron][input];
				}
			}
			currentLayerOutput[neuron] += biases[neuron];
			currentLayerOutput[neuron] = logistic(currentLayerOutput[neuron]);
		}
		return currentLayerOutput;
	}

//	// loss function for the individual input
//	public double loss(int category, double[] actualOutput) {
//		double[] expectedOutput = new double[NEURONS_OUTPUT_LAYER];
//		expectedOutput[category] = 1.0;
//		return squaredError(expectedOutput, actualOutput);
//	}
//
//	// used to calculate the loss function
//	public double squaredError(double[] expectedOutput, double[] actualOutput) {
//		double sum = 0.0;
//		for (int output = 0; output < expectedOutput.length; output++) {
//			sum += Math.pow(expectedOutput[output] - actualOutput[output], 2);
//		}
//		return sum;
//	}

	public double[][] convertToDouble(int[][] dataSet) {
		double[][] newDataset = new double[dataSet.length][dataSet[0].length];
		for (int row = 0; row < dataSet.length; row++) {
			for (int column = 0; column < dataSet[row].length; column++) {
				newDataset[row][column] = (double) dataSet[row][column];
			}
		}
		return newDataset;
	}

	// feed the network with mini-batch and return the total cost of the network
	// UNDONE: mini-batches
	public void updateWeights(int[][] trainigSet) {
		int[] categories = DigitCategorizer.extractCategories(trainigSet);
		double[][] trainSetDouble = convertToDouble(trainigSet);
		for (int input = 0; input < trainigSet.length; input++) {

			// feed hidden layer
			double[] outputL2 = new double[NEURONS_HIDDEN_LAYER];
			outputL2 = feedForwardLayer(NEURONS_HIDDEN_LAYER, trainSetDouble[input], weights1to2, biasesL2, true);

			// feed output layer
			double[] outputL3 = new double[NEURONS_OUTPUT_LAYER];
			outputL3 = feedForwardLayer(NEURONS_OUTPUT_LAYER, outputL2, weights2to3, biasesL3, false);

			// backpropagation
			double[] expectedOutput = new double[NEURONS_OUTPUT_LAYER];
			expectedOutput[categories[input]] = 1.0;
			// outer layer
			double[] errorOuterLayer = errorOuterLayer(outputL3, expectedOutput);
			backpropagation(weights2to3, errorOuterLayer, outputL2, LEARNING_RATE);
			// inner layer
			double[] errorInnerLayer = errorInnerLayer(outputL2, errorOuterLayer, weights2to3);
			backpropagation(weights1to2, errorInnerLayer, trainSetDouble[input], LEARNING_RATE);

		}

	}

	public double test(int[][] testSet) {
		int[] categories = DigitCategorizer.extractCategories(testSet);
		double[][] testSetDouble = convertToDouble(testSet);
		int correctOutputCounter = 0;
		for (int input = 0; input < testSet.length; input++) {

			// feed hidden layer
			double[] outputL2 = new double[NEURONS_HIDDEN_LAYER];
			outputL2 = feedForwardLayer(NEURONS_HIDDEN_LAYER, testSetDouble[input], weights1to2, biasesL2, true);

			// feed output layer
			double[] outputL3 = new double[NEURONS_OUTPUT_LAYER];
			outputL3 = feedForwardLayer(NEURONS_OUTPUT_LAYER, outputL2, weights2to3, biasesL3, false);

			// evaluate
			double[] expectedOutput = new double[NEURONS_OUTPUT_LAYER];
			expectedOutput[categories[input]] = 1.0;

			if (outputCat(outputL3) == categories[input])
				correctOutputCounter++;

//			//TEST
//			if (input % 250 == 0) {
//				System.out.println("Input no. " + input);
//				System.out.println("Correct category: " + categories[input]);
//				System.out.println("Output category: " + outputCat(outputL3));
//				for (int neuron = 0; neuron < outputL3.length; neuron++) {
//					System.out.println(neuron + ". " + outputL3[neuron]);
//				}
//				System.out.println();
//			}

		}

		return ((double) correctOutputCounter / testSet.length) * 100.00;
	}

	public int outputCat(double[] output) {
		double max = 0.0;
		int indexMax = 0;
		for (int neuron = 0; neuron < output.length; neuron++) {
			if (max < output[neuron]) {
				max = output[neuron];
				indexMax = neuron;
			}
		}
		return indexMax;
	}

	//I could do that
	public int[][] miniBatch(int factor, int offset, int[][] trainingSet) {
		int[][] miniBatch = new int[offset][DigitCategorizer.COLUMNS];
		for (int digit = factor * offset, digMiniBatch = 0; digit < offset; digit++, digMiniBatch++) {
			miniBatch[digMiniBatch] = trainingSet[digit];
		}
		return miniBatch;
	}

	public static double[] errorOuterLayer(double[] output, double[] target) {

		double[] error = new double[output.length];
		for (int neuron = 0; neuron < error.length; neuron++) {
			error[neuron] = (output[neuron] - target[neuron]) * output[neuron] * (1 - output[neuron]);
		}
		return error;

	}

	public static double[] errorInnerLayer(double[] output, double[] errorOuter, double[][] weights) {

		double[] error = new double[output.length];

		for (int neuron = 0; neuron < output.length; neuron++) {
			double weightedSum = 0.0;
			for (int neuronOuter = 0; neuronOuter < errorOuter.length; neuronOuter++) {
				weightedSum += weights[neuronOuter][neuron] * errorOuter[neuronOuter];
			}
			error[neuron] = weightedSum * output[neuron] * (1 - output[neuron]);
		}

		return error;

	}

	public static void backpropagation(double[][] weights, double[] error, double[] outputPreviousLayer,
			double learningRate) {

		for (int neuron = 0; neuron < error.length; neuron++) {
			for (int weight = 0; weight < weights[neuron].length; weight++) {
				weights[neuron][weight] -= learningRate * outputPreviousLayer[weight] * error[neuron];
			}
		}

	}


}
