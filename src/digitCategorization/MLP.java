package digitCategorization;

import java.util.Random;

public class MLP {
	
	public final static int NEURONS_INPUT_LAYER = 64;
	public final static int NEURONS_HIDDEN_LAYER = 15;
	public final static int NEURONS_OUTPUT_LAYER = 10; 
	
	public static double[] biasesHL = new double[NEURONS_HIDDEN_LAYER];
	public static double[] biasesOL = new double[NEURONS_OUTPUT_LAYER];
	
	//UNDONE - p. 34 how the weight matrices should be initilised?
	public static double[][] weights1 = new double[][];
	public static double[][] weights2 = new double[][];
	
	public void init() {
		
		//biases randomly initialised for hidden and output layers
		Random random = new Random();
		for (double bias : biasesHL) {
			bias = random.nextGaussian();
		}
		for (double bias : biasesOL) {
			bias = random.nextGaussian();
		}
	}

}
