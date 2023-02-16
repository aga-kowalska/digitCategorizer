package digitCategorization;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;

public class DigitCategorizer {
	
	public static final int ROWS = 2810;
	public static final int COLUMNS = 65;
	public static final int INPUT_DIMENSION = COLUMNS-1;
	public static int[][] test_fold = new int[ROWS][COLUMNS];
	public static int[][] train_fold = new int[ROWS][COLUMNS];

	public static final String FILE_LOCATION = System.getProperty("user.dir");
	public static final String TEST_FILE_PATH = FILE_LOCATION + File.separator + "cw2DataSet1.csv";
	public static final File TEST_FILE = new File(TEST_FILE_PATH);
	public static final String TRAIN_FILE_PATH = FILE_LOCATION + File.separator + "cw2DataSet2.csv";
	public static final File TRAIN_FILE = new File(TRAIN_FILE_PATH);

	public static void main(String[] args) {
		
		readFile(TEST_FILE, test_fold);
		readFile(TRAIN_FILE, train_fold);
		
		System.out.println("Digit Categorization Task, UCI digit dataset.\n");
		
		//Euclidian distance
		System.out.printf("Euclidean distance two fold verification: %.2f%% \n", euclidianDistanceTwoFoldCrossValidation(train_fold, test_fold));
		
		//MLP
		//switch isPrinted to false if you don't want to see the progress of learning, just the final result of the two fold verification
		boolean isPrinted = false;
		int iteration = 500;
		System.out.printf("\nMulti-Layer Perceptron two fold verification: %.2f%% ", MLPTwoFoldVerification(train_fold, test_fold, iteration, isPrinted));
	}
	
	//method to perform two fold verification of training MLP algorithm 
	public static double MLPTwoFoldVerification(int[][] trainfold, int[][] testfold, int numberOfIterations, boolean isPrinted) {
		double finalResult;
		double partialResult;
		double max = 0.0;
		MLP mlp = new MLP();
		//first fold
		mlp.init();
		if (isPrinted) System.out.println("\n----------------------------------------------------MLP First Fold-------------------------------------------------- ");
		for (int iteration = 0; iteration < numberOfIterations; iteration++) {
			mlp.updateWeights(trainfold);
			partialResult = mlp.test(testfold);
			if (partialResult > max) {
				max = partialResult;
				if (isPrinted) System.out.printf("MLP. Accuracy. 1st fold: %.2f%% Iteration: %d\n", partialResult, iteration);
			}
		}
		finalResult = max;
		//second fold
		mlp.init();
		max = 0.0;
		if (isPrinted) System.out.println("\n----------------------------------------------------MLP Second Fold-------------------------------------------------- ");
		for (int iteration = 0; iteration < numberOfIterations; iteration++) {
			mlp.updateWeights(testfold);
			partialResult = mlp.test(testfold);
			if (partialResult > max) {
				max = partialResult;
				if (isPrinted) System.out.printf("MLP. Accuracy. 2nd fold: %.2f%% Iteration: %d\n", partialResult, iteration);
			}
		}
		finalResult += max;
		return finalResult/2.0;
	}
	
	//method to perform two fold verification of the nearest neighbour algorithm using Euclidean distance
	public static double euclidianDistanceTwoFoldCrossValidation(int[][] train_fold, int[][] test_fold) {
		
		// first fold
		int[] resultArray = findTheClosest(train_fold, test_fold);
		int[] expectedArray = extractCategories(test_fold);
		double firstFoldEval = evaluate(resultArray, expectedArray);
		
		//second fold
		resultArray = findTheClosest(test_fold, train_fold);
		expectedArray = extractCategories(train_fold);
		double secondFoldEval = evaluate(resultArray, expectedArray);
		
		return (firstFoldEval+secondFoldEval)/2.0;
	}
	
	//read and prepare the data sets
	public static void readFile(File file, int[][] array) {
		Scanner input;
		String inputString = "";
		String inputStringArray[][] = new String[ROWS][COLUMNS];
		int row = 0;
		try {
			input = new Scanner(file);
			while (input.hasNext()) {
				inputString = input.nextLine().trim();
				inputStringArray[row] = inputString.split(",");
				row++;
			}
			input.close();
		} catch (FileNotFoundException fnf) {
			System.out.println("File not found");
		}
		
		for (int row1 = 0; row1 < ROWS; row1++) {
			for (int column = 0; column < COLUMNS; column++) {
				array[row1][column] = Integer.parseInt(inputStringArray[row1][column]);

			}

		}
	}
	

	//methods below are used for Euclidean distance categorization
	public static double euclideanDistance(int[] digit1, int[] digit2) {
		double sum = 0.0;
		for (int valueIndex = 0; valueIndex < INPUT_DIMENSION; valueIndex++) {
			sum += Math.pow(digit1[valueIndex] - digit2[valueIndex], 2);
		}
		return sum;
	}
	
	public static int[] findTheClosest(int[][] train_set, int[][] test_set) {
		double minDistance = Double.POSITIVE_INFINITY; 
		int[] match = new int[ROWS]; 
		
		for (int test_digit = 0; test_digit < ROWS; test_digit++) {
			minDistance = Double.POSITIVE_INFINITY; 
			for (int train_digit = 0; train_digit < ROWS; train_digit++) {
				double distance = euclideanDistance(test_set[test_digit], train_set[train_digit]);
				if (distance < minDistance) {
					minDistance = distance;
					match[test_digit] = train_set[train_digit][64];
				}
			}
		}
		
		return match;
	}
	
	//used to evaluate Euclidean distance categorization output
	public static double evaluate(int[] result, int[] expected) {
		double correctResults = 0.0;
		for (int resultDigit = 0; resultDigit < result.length; resultDigit++) {
			if (result[resultDigit] == expected[resultDigit]) {
				correctResults++;
			}
		}
		return correctResults/ROWS * 100.00;
	}
	
	//creates a separate array to store the corresponding categories of the input data
	public static int[] extractCategories(int[][] data_set) {
		int[] categories = new int[ROWS];
		
		for (int digit = 0; digit < ROWS; digit++) {
			categories[digit] = data_set[digit][64];
		}
		
		return categories;
	}
	

	
	//printArray methods used for testing and verification
	public static void printArray(int[][] array) {
		for (int row = 0; row < ROWS; row++) {
			System.out.print("\n" + row + ". ");
			for (int column = 0; column < COLUMNS; column++) {
				System.out.print(array[row][column] + " ");
			}
			
		}
	}
	
	public static void printArray(double[][] array) {
		for (int row = 0; row < ROWS; row++) {
			System.out.print("\n" + row + ". ");
			for (int column = 0; column < COLUMNS; column++) {
				System.out.print(array[row][column] + " ");
			}
			
		}
	}

}
