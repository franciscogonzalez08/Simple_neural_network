package neuralNetwork;

import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.io.PrintWriter;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;


import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;

import javax.imageio.ImageIO;

/**
 * This class creates a neural network with 3 layers: an input layer, hidden layer and output layer.
 * @author franciscogonzalez08 and drophy at GitHub
 *
 */
public class Network {
	private DenseDoubleMatrix1D inputs;
	private DenseDoubleMatrix1D expected_outputs;
	private DenseDoubleMatrix2D weights1;
	private DenseDoubleMatrix2D weights2;
	private DenseDoubleMatrix1D outputs;
	private DenseDoubleMatrix1D hidden_layer;
	private final double LEARNING_RATE; // valid range: [0.001, 0.1]
	private Algebra algebra = new Algebra();
	private Map<String, Integer> labelsMap = null;
	
	// CONSTRUCTORS
	/**
	 * Constructs a network with the specified number of neurons for the input and output layers.
	 * The number of neurons the hidden layer has is obtained as ceil((inputSize + outputSize)/2.0),
	 * which is the recommended size for this layer.
	 * Its learning rate will be set to 0.1 which is its highest possible value.
	 * This allows the network to learn faster (require less training to get good results), 
	 * but may perform poorly compared to a network with a lower learning rate.
	 * @param inputSize - size of the input layer
	 * @param outputSize - size of the output layer
	 */
	public Network(int inputSize, int outputSize) {
		this(inputSize, outputSize, 0.1);
	}
	
	/**
	 * Constructs a network with the specified learning rate and 
	 * number of neurons for the input and output layers.
	 * The number of neurons the hidden layer has is obtained as ceil((inputSize + outputSize)/2.0),
	 * which is the recommended size for this layer.
	 * @param inputSize - size of the input layer
	 * @param outputSize - size of the output layer
	 * @param learning_rate - number in range [0.001, 0.1] that specifies how strongly this network is affected by a training case
	 */
	public Network(int inputSize, int outputSize, double learning_rate) {
		this(inputSize, (int)Math.ceil((inputSize + outputSize)/2.0), outputSize, learning_rate);
	}
	
	/**
	 * Constructs a network with the specified learning rate and number of neurons for each layer.
	 * @param inputSize - size of the input layer
	 * @param middleNeurons - size of the hidden layer
	 * @param outputSize - size of the output layer
	 * @param learning_rate - number in range [0.001, 0.1] that specifies how strongly this network is affected by a training case
	 */
	public Network(int inputSize, int middleNeurons, int outputSize, double learning_rate) {
		LEARNING_RATE = learning_rate >= 0.001 && learning_rate <= 0.1? learning_rate : 0.1;
		
		double[][] weights1 = new double[middleNeurons][inputSize];
		double[][] weights2 = new double[outputSize][middleNeurons];
		
		for(int i = 0; i < weights1.length; i++)
			for(int j = 0; j < weights1[i].length; j++)
				weights1[i][j] = (Math.random() / 5) - 0.1;

		for(int i = 0; i < weights2.length; i++)
			for(int j = 0; j < weights2[i].length; j++)
				weights2[i][j] = (Math.random() / 5) - 0.1;
		
		inputs = new DenseDoubleMatrix1D(inputSize); // alternatively, build in each test method
		expected_outputs = new DenseDoubleMatrix1D(outputSize);
		this.weights1 = new DenseDoubleMatrix2D(weights1);
		hidden_layer = new DenseDoubleMatrix1D(middleNeurons);
		this.weights2 = new DenseDoubleMatrix2D(weights2);
		outputs = new DenseDoubleMatrix1D(outputSize);
	}
	
	// SETTER
	/**
	 * By default, the network is configured so that the outputs correspond to a 0 based 
	 * base 10 number. In any other scenario, the user must provide a map with the class (as a
	 * String) as its key and 0 based indexes as its values.
	 * @param m
	 */
	public void configureMapping(Map<String, Integer> m) {
		labelsMap = m;
	}
	
	// TRAIN
	/**
	 * Trains this network using the file specified in "path".
	 * Only the rows in the range specified by "from" and "to" will be used. 
	 * The file must be a CSV file with a row per training case.
	 * The first column of each row must contain the class the image belongs to.
	 * The following columns must contain a value in range [0, 255] corresponding
	 * to the greyscale value of a pixel.
	 * Pixels are expected to be ordered as consecutive rows of the original image.  
	 * @param path - path to the CSV file
	 * @param from - first row that will be used 
	 * @param to - last row that will be used
	 */
	public void trainCSV(String path, int from, int to) {
		try {
            FileReader fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);

            long lineCount = Files.lines(Paths.get(path)).count();
            if(to > lineCount) {
            	System.out.println("The file does not have the requested number of rows. The network was not trained.");
            	br.close();
            	return;
            }
            
            String str;
            String arrLine[];
            int lineNumber = 1;
            while((str = br.readLine()) != null && lineNumber <= to)
            {
            	if(lineNumber < from) {
            		lineNumber++;
            		continue;
            	}
            	
            	// We'll save the line in an array
            	arrLine = str.split(",");
            	
            	// We'll feed in the inputs
            	for(int i = 1; i < inputs.size()+1; i++) 
            		inputs.setQuick(i-1, Double.parseDouble(arrLine[i]));
            	
            	// We update the expected outputs for the current case
            	expected_outputs.assign(0);
            	if(labelsMap == null) 
            		// If no special mapping is configured, we'll assume it works with 0-9
            		expected_outputs.setQuick(Integer.parseInt(arrLine[0]), 1.0);
            	else
            		expected_outputs.setQuick(labelsMap.get(arrLine[0]), 1.0);
            	
            	// We're ready to train the network with the new data
            	train();
            	lineNumber++;
            }
            
            br.close();
        } catch(IOException e){
            System.out.println("Couldn't find or read the file.");
        }
	}
	
	/**
	 * Trains this network using the file specified in "path"
	 * and the first "quantity" rows. 
	 * The file must be a CSV file with a row per training case.
	 * The first column of each row must contain the class the image belongs to.
	 * The following columns must contain a value in range [0, 255] corresponding
	 * to the greyscale value of a pixel.
	 * Pixels are expected to be ordered as consecutive rows of the original image.  
	 * @param path - path to the CSV file
	 * @param quantity - number of rows that will be used from the CSV file
	 */
	public void trainCSV(String path, int quantity) {
		trainCSV(path, 1, quantity);
	}
	
	/**
	 * Trains this network using the file specified in "path". 
	 * The file must be a CSV file with a row per training case.
	 * The first column of each row must contain the class the image belongs to.
	 * The following columns must contain a value in range [0, 255] corresponding
	 * to the greyscale value of a pixel.
	 * Pixels are expected to be ordered as consecutive rows of the original image. 
	 * @param path - path to the CSV file
	 */
	public void trainCSV(String path) {
		try {
            FileReader fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);

            int lineCount = (int)Files.lines(Paths.get(path)).count();
            br.close();
            trainCSV(path, 1, lineCount);
            
        } catch(IOException e){
            System.out.println("Couldn't find or read the file.");
        }
	}
	
	// TEST
	/**
	 * Tests the accuracy of the network using the file specified in "path".
	 * Only the rows in the range specified by "from" and "to" will be used. 
	 * The method will print in the console the number of cases this network solved correctly.
	 * The file must be a CSV file with a row per training case.
	 * The first column of each row must contain the class the image belongs to.
	 * The following columns must contain a value in range [0, 255] corresponding
	 * to the greyscale value of a pixel.
	 * Pixels are expected to be ordered as consecutive rows of the original image.
	 * @param path - path to the CSV file
	 * @param from - first row that will be used
	 * @param to - last row that will be used
	 * @return An array with the inverse of the average error of each output. 
	 * Note that while higher values do correlate to a better performance, high values are usually
	 * obtained regardless of the network's accuracy. 
	 */
	public double[] testCSV(String path, int from, int to) {
		int quantity = to - from + 1;
		int correctGuesses = 0;
		
		DenseDoubleMatrix1D avg_error_vector = new DenseDoubleMatrix1D(outputs.size());
        avg_error_vector.assign(0);
        DenseDoubleMatrix1D temp_vector;
		
		try {
            FileReader fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);

            long lineCount = Files.lines(Paths.get(path)).count();
            if(to > lineCount) {
            	System.out.println("The file does not have the requested number of rows. The network was not trained.");
            	br.close();
            	return null;
            }
            
            String str;
            String arrLine[];
            int lineNumber = 1;
            double[] arrOutputs;
            double desiredOutput;
            
            while((str = br.readLine()) != null && lineNumber <= to)
            {	
            	if(lineNumber < from) {
            		lineNumber++;
            		continue;
            	}
            	
            	// We'll save the line in an array
            	arrLine = str.split(",");
            	
            	// We'll feed in the inputs
            	for(int i = 1; i < inputs.size()+1; i++) 
            		inputs.setQuick(i-1, Double.parseDouble(arrLine[i]));
            	
            	// We update the expected outputs for the current case
            	expected_outputs.assign(0);
            	if(labelsMap == null) 
            		// If no special mapping is configured, we'll assume it works with 0-9
            		expected_outputs.setQuick(Integer.parseInt(arrLine[0]), 1.0);
            	else
            		expected_outputs.setQuick(labelsMap.get(arrLine[0]), 1.0);
            	
            	// We'll calculate the outputs
            	feed_forward();
            	
            	// Now we'll see how far off they were and calculate an average error
            	temp_vector = subtract(expected_outputs, outputs);
            	scalar_product(temp_vector, 1.0/quantity);
            	
            	// An error of 1 and -1 shouldn't cancel out, they should add up
            	for(int i = 0; i < outputs.size(); i++)
            		temp_vector.setQuick(i, Math.abs(temp_vector.getQuick(i)));
            	
            	avg_error_vector = sum(avg_error_vector, temp_vector);
            	
            	// Alternative: just give points if the answer is right
            	// (right being the higher output matching the label)
            	arrOutputs = outputs.toArray();
            	if(labelsMap == null) 
            		desiredOutput = arrOutputs[Integer.parseInt(arrLine[0])];
            	else
            		desiredOutput = arrOutputs[labelsMap.get(arrLine[0])];
            	Arrays.sort(arrOutputs);
            	if(arrOutputs[arrOutputs.length-1] == desiredOutput) // note: even if another class has the max value, we'll consider that to be a correct answer
            		correctGuesses++;
            		
            	lineNumber++;
            }
            br.close();
            
            // Now we have an average error per class, but it's in range [-1, 1]
            for(int i = 0; i < outputs.size(); i++)
            	avg_error_vector.setQuick(i, (1-avg_error_vector.getQuick(i))*100);
            
        } catch(IOException e){
            System.out.println("Couldn't find or read the file.");
        }
		
		System.out.println(	"Score: " + correctGuesses + "/" + quantity + 
							" (" + correctGuesses*100/quantity + "%)");
		
		return avg_error_vector.toArray();
	}
	
	/**
	 * Tests the accuracy of the network using the file specified in "path"
	 * and the first "quantity" rows. 
	 * The method will print in the console the number of cases this network solved correctly.
	 * The file must be a CSV file with a row per training case.
	 * The first column of each row must contain the class the image belongs to.
	 * The following columns must contain a value in range [0, 255] corresponding
	 * to the greyscale value of a pixel.
	 * Pixels are expected to be ordered as consecutive rows of the original image.
	 * @param path - path to the CSV file
	 * @param quantity - number of rows that will be used from the CSV file
	 * @return An array with the inverse of the average error of each output. 
	 * Note that while higher values do correlate to a better performance, high values are usually
	 * obtained regardless of the network's accuracy.  
	 */
	public double[] testCSV(String path, int quantity) {
		return testCSV(path, 1, quantity);
	}
	
	/**
	 * Tests the accuracy of the network using the file specified in "path". 
	 * The method will print in the console the number of cases this network solved correctly.
	 * The file must be a CSV file with a row per training case.
	 * The first column of each row must contain the class the image belongs to.
	 * The following columns must contain a value in range [0, 255] corresponding
	 * to the greyscale value of a pixel.
	 * Pixels are expected to be ordered as consecutive rows of the original image.
	 * @param path - path to the CSV file
	 * @return An array with the inverse of the average error of each output. 
	 * Note that while higher values do correlate to a better performance, high values are usually
	 * obtained regardless of the network's accuracy.  
	 */
	public double[] testCSV(String path) {
		try {
            FileReader fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);

            int lineCount = (int)Files.lines(Paths.get(path)).count();
            br.close();
            return testCSV(path, 1, lineCount);
            
        } catch(IOException e){
            System.out.println("Couldn't find or read the file.");
            return null;
        }

	}
	
	/**
	 * Receives an image and prints to the console the network's guess.
	 * A line is printed per possible class. 
	 * Each line contains the class's index followed by a number in range [0, 1].
	 * The higher the number, the more certain the network is that the image given
	 * belongs to that class.
	 * The image must have exactly as many pixels as the input layer.  
	 * @param path - path to the image
	 */
	// Evaluate
	public void evaluateIMG(String path) {
		File imageFile = new File(path);
		try {
			BufferedImage image = ImageIO.read(imageFile);
			int width = image.getWidth();
			int height = image.getHeight();
			if(width*height != inputs.size()) 
			{
				System.out.println("Could not evaluate the given image. Image has a different number of pixels than the network's input layer.");
				return;
			}
			Color pixel;
			
			for(int i = 0; i < height; i++)
				for(int j = 0; j < width; j++) {
					pixel = new Color(image.getRGB(j, i));
					inputs.setQuick(i * width + j, 255 - ((pixel.getRed() +
														   pixel.getGreen() + 
														   pixel.getBlue())/3));
				}
			
			feed_forward();
			int i = 0;
			for(double guess : outputs.toArray()) {
				System.out.println(i + ") " + guess);
			}
			
		} catch(IOException e) {
			System.out.println("Couldn't read the given image.");
		}
	}
	
	/**
	 * Creates a .txt file containing the current configuration of the network.
	 * The file can be used by the load static method to create an identical network. 
	 * This is specially useful so that the network's training is not lost after
	 * the application's execution ends.
	 * @param path_name - path and name for the new file 
	 */
	public void save(String path_name) {
		if(path_name == null) 
		{
			System.out.println("path_name is not specified. Save file not created.");
			return;
		}
		
		try {
			FileWriter file_writer = new FileWriter(path_name);
			PrintWriter print_writer = new PrintWriter(file_writer);
			
			int inputsSize = inputs.size(),
				middleSize = hidden_layer.size(),
				outputsSize = outputs.size();
			
			print_writer.println(inputsSize + " " + middleSize + " " + outputsSize);
			
			//print_writer.printf("%f\n\n", LEARNING_RATE);
			print_writer.println(LEARNING_RATE);
			print_writer.println();
			
			// weights1 dimensions: [middle_neurons][input_size]
			int i, j;
			for(i = 0; i < middleSize; i++)
			{
				for(j = 0; j < inputsSize-1; j++)
					print_writer.printf("%f ", weights1.getQuick(i, j));
				print_writer.println(weights1.getQuick(i, j));
			}	
			print_writer.println();	
			
			// weights2 dimensions: [outputSize][middleNeurons]
			for(i = 0; i < outputsSize; i++)
			{
				for(j = 0; j < middleSize-1; j++)
					print_writer.printf("%f ", weights2.getQuick(i, j));
				print_writer.println(weights2.getQuick(i, j));
			}		
			
			// Map
			if(labelsMap != null)
			{
				print_writer.println();
				Set<String> keys = labelsMap.keySet();
				Iterator<String> iterator = keys.iterator();
				String key;
				while(iterator.hasNext()) {
					key = iterator.next();
					print_writer.println(key + " " + labelsMap.get(key));
				}
			}

			print_writer.close();
		} catch (IOException e) {
			System.out.println("Error. Possible causes are: the named file exists but is a directory rather than a regular file, does not exist but cannot be created, or cannot be opened for any other reason");
		}
	}
	
	/**
	 * Creates and returns a neural network with the configuration given by the specified .txt file.
	 * Such a file can be created using the save method on an already existing neural network.
	 * @param path - path to the .txt file
	 * @return a new neural network object with the configuration given by the specified .txt file.
	 */
	public static Network load(String path) {
		Network nn = null;
		try {
			FileReader fr = new FileReader(path);
			BufferedReader br = new BufferedReader(fr);
			
			String[] arrLine = br.readLine().split(" ");
			String lr = br.readLine();
			
			int inputSize = Integer.parseInt(arrLine[0]);
			int middleSize = Integer.parseInt(arrLine[1]);
			int outputsSize = Integer.parseInt(arrLine[2]);
			double learningRate = Double.parseDouble(lr);
			
			nn = new Network(inputSize, middleSize, outputsSize, learningRate);
			
			br.readLine(); //Empty space
			
			for(int i = 0; i < middleSize; i++) {
				arrLine = br.readLine().split(" ");
				for(int j = 0; j < inputSize; j++)	
					nn.weights1.setQuick(i, j, Double.parseDouble(arrLine[j]));
			}
			
			br.readLine();
			
			for(int i = 0; i < outputsSize; i++) {
				arrLine = br.readLine().split(" ");
				for(int j = 0; j < middleSize; j++)	
					nn.weights2.setQuick(i, j, Double.parseDouble(arrLine[j]));
			}	
			
			// We'll check if there's a Map
			br.readLine();
			
			String line = br.readLine();
			if(line != null)
			{
				Map<String, Integer> m = new HashMap<>();
				do
				{
					arrLine =  line.split(" ");
					m.put(arrLine[0], Integer.parseInt(arrLine[1]));
				} while((line = br.readLine()) != null);
				nn.configureMapping(m);
			}
			
			br.close();
			
		} catch(IOException e) {
			System.out.println("File not found.");
		}
		return nn;
	}
	
	// PRIVATE METHODS
	// Auxiliary Methods
	private void feed_forward() {
		hidden_layer = (DenseDoubleMatrix1D)algebra.mult(weights1, inputs);
		sigmoid(hidden_layer);
		outputs = (DenseDoubleMatrix1D)algebra.mult(weights2, hidden_layer);
		sigmoid(outputs);
	}
	
	private void train() {
		feed_forward();
		
		//Calculate the errors
		DenseDoubleMatrix1D output_errors = subtract(expected_outputs, outputs);
		
		//Adjust weights2 
		
		// delta = sigmoid'(outputs)*output_errors
		DenseDoubleMatrix1D output_delta = cross_product(dSigmoid(outputs), output_errors);
		
		// new_weight = hidden_layer * output_layer_delta * learning_rate
		scalar_product(output_delta, LEARNING_RATE);
		
		int oSize = outputs.size();
		int hSize = hidden_layer.size();
		for(int o = 0; o < oSize; o++)
			for(int h = 0; h < hSize; h++)
				weights2.setQuick(o, h, weights2.getQuick(o, h) +
				hidden_layer.getQuick(h)*output_delta.getQuick(o));
		
		// Adjust weights1
		// hidden error = weights2 * output_errors
		DenseDoubleMatrix1D hidden_error = (DenseDoubleMatrix1D)algebra.mult(weights2.viewDice(), output_errors);
		
		// new_weight = LR * hidden error * dSigmoid(hidden_layer) * input_layer 
		DenseDoubleMatrix1D hidden_delta = cross_product(dSigmoid(hidden_layer), hidden_error);
		scalar_product(hidden_delta, LEARNING_RATE);
		
		int iSize = inputs.size();
		for(int h = 0; h < hSize; h++)
			for(int i = 0; i < iSize; i++)
				weights1.setQuick(h, i, weights1.getQuick(h, i) + 
				inputs.getQuick(i)*hidden_delta.getQuick(h));
		
	}
	
	//Auxiliary auxiliary methods
	private void sigmoid(DenseDoubleMatrix1D v) {
		int size = v.size();
		for(int i = 0; i < size; i++)
			v.setQuick(i, 1/(1+Math.pow(Math.E, -v.getQuick(i))));
	}
	
	private DenseDoubleMatrix1D dSigmoid(DenseDoubleMatrix1D v) {
		int size = v.size();
		DenseDoubleMatrix1D u = new DenseDoubleMatrix1D(size);
		
		for(int i = 0; i < size; i++)
			u.setQuick(i, v.getQuick(i)*(1-v.getQuick(i)));
		
		return u;
	}
	
	private DenseDoubleMatrix1D cross_product(DenseDoubleMatrix1D u, DenseDoubleMatrix1D v) {
		int size = u.size();
		DenseDoubleMatrix1D w = new DenseDoubleMatrix1D(size);
		
		for(int i = 0; i < size; i++)
			w.setQuick(i, u.getQuick(i) * v.getQuick(i));
		
		return w;
	}
	
	private void scalar_product(DenseDoubleMatrix1D v, double scalar) {
		int size = v.size();
		
		for(int i = 0; i < size; i++)
			v.setQuick(i, v.getQuick(i) * scalar);
	}
	
	private static DenseDoubleMatrix1D sum(DenseDoubleMatrix1D v1, DenseDoubleMatrix1D v2) {
		DenseDoubleMatrix1D v3 = new DenseDoubleMatrix1D(v1.size());
		for(int i = 0; i < v1.size(); i++)
			v3.setQuick(i, v1.getQuick(i) + v2.getQuick(i));
		return v3;
	}
	
	private static DenseDoubleMatrix1D subtract(DenseDoubleMatrix1D v1, DenseDoubleMatrix1D v2) {
		DenseDoubleMatrix1D v3 = new DenseDoubleMatrix1D(v1.size());
		for(int i = 0; i < v1.size(); i++)
			v3.setQuick(i, v1.getQuick(i) - v2.getQuick(i));
		return v3;
	}
}
