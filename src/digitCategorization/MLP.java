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

	public static double[] feedForward(int currentLayerNeurons, double[] outputPreviousLayer, double[][] weigths,
			double[] biases, boolean isFirst) {
		
		double[] currentLayerOutput = new double[currentLayerNeurons];
		
		//for every neuron in the current layer: add weighted inputs, add bias and apply logistic function
		for (int neuron = 0; neuron < currentLayerNeurons; neuron++) {
			//if it is the first layer skip the last feature (it is a category label)
			if (isFirst) {
				for (int input = 0; input < outputPreviousLayer.length-1; input++) {
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
	
	//loss function for the individual output
	public double loss(int category, double[] actualOutput) {
		
		double[] expectedOutput = new double[NEURONS_OUTPUT_LAYER];
		expectedOutput[category] = 1.0;
		return euclideanDistance(expectedOutput, actualOutput);
		
	}
	
	//used to calculate the loss function
	public double euclideanDistance(double[] expectedOutput, double[] actualOutput) {
		
		double sum = 0.0;
		for (int output = 0; output < expectedOutput.length; output++) {
			sum += Math.pow(expectedOutput[output] - actualOutput[output], 2);
		}
		return sum;
	}
	
	public double[][] convertToDouble(int[][] dataSet) {
		double[][] newDataset = new double[dataSet.length][dataSet[0].length];
		for (int row = 0; row < dataSet.length; row++) {
			for (int column = 0; column < dataSet[row].length; column++) {
				newDataset[row][column] = (double)dataSet[row][column];
			}
		}
		return newDataset;
	}
	

	//feed the network once with the whole training set and return the total loss of the network
	//by adding the cost after every network input and calculating the average (dividing by the number of input items)
	public double train(int[][] trainigSet) {
		double cost = 0.0;
		int[] categories = DigitCategorizer.extractCategories(trainigSet);
		double[][] trainSetDouble = convertToDouble(trainigSet);
		for (int input = 0; input < trainigSet.length; input++) {
			double[] outputL2 = new double[NEURONS_HIDDEN_LAYER];
			//feed hidden layer
			outputL2 = feedForward(NEURONS_HIDDEN_LAYER, trainSetDouble[input], weights1to2, biasesL2, true);
			//feed output layer
			double[] outputL3 = new double[NEURONS_OUTPUT_LAYER];
			outputL3 = feedForward(NEURONS_OUTPUT_LAYER, outputL2, weights2to3, biasesL3, false);
			cost += loss(categories[input], outputL3);
			
			for (int row = 0; row < outputL3.length; row++) {
					System.out.print(input + ". " + outputL3[row]);
				
			}
			System.out.println();
		}
		return cost/trainigSet.length;
	}
	


}
