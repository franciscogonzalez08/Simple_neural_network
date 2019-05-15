package neuralNetwork;

import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;


import java.util.Arrays;
import java.util.Map;

import javax.imageio.ImageIO;

public class Network {
	private DenseDoubleMatrix1D inputs;
	private DenseDoubleMatrix1D expected_outputs;
	private DenseDoubleMatrix2D weights1;
	private DenseDoubleMatrix2D weights2;
	private DenseDoubleMatrix1D outputs;
	private DenseDoubleMatrix1D hidden_layer;
	private final double LEARNING_RATE; // valid range: [0.001, 0.1]
	Algebra algebra = new Algebra();
	Map<String, Integer> labelsMap = null;
	
	//Builders
	public Network(int inputSize, int outputSize, double learning_rate) {
		this(inputSize, (int)Math.ceil((inputSize + outputSize)/2.0), outputSize, learning_rate);
	}
	
	public Network(int inputSize, int middleNeurons, int outputSize, double learning_rate) {
		LEARNING_RATE = learning_rate;
		
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
	
	//Train
	public void trainCSV(String path, int from, int to) {
		try {
            FileReader fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);

            String str;
            String arrLine[];
            int lineNumber = 1;
            while((str = br.readLine()) != null && lineNumber <= to) // TODO: maybe check there's actually that many rows?
            {
            	if(lineNumber < from) {
            		lineNumber++;
            		continue;
            	}
            	
            	// We'll save the line in an array
            	arrLine = str.split(",");
            	
            	// TODO: Validate the size or arrLine is = inputs+1 (bc label)
            	
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
	
	public void trainCSV(String path, int quantity) {
		trainCSV(path, 1, quantity);
	}
	
	// TODO: maybe do one with just path too
	
	// Test
	public double[] testCSV(String path, int from, int to) {
		int quantity = to - from + 1;
		int correctGuesses = 0;
		
		DenseDoubleMatrix1D avg_error_vector = new DenseDoubleMatrix1D(outputs.size());
        avg_error_vector.assign(0);
        DenseDoubleMatrix1D temp_vector;
		
		try {
            FileReader fr = new FileReader(path);
            BufferedReader br = new BufferedReader(fr);

            String str;
            String arrLine[];
            int lineNumber = 1;
            double[] arrOutputs;
            double desiredOutput;
            
            while((str = br.readLine()) != null && lineNumber <= to) // TODO: maybe check there's actually that many rows?
            {	
            	if(lineNumber < from) {
            		lineNumber++;
            		continue;
            	}
            	
            	// We'll save the line in an array
            	arrLine = str.split(",");
            	
            	// TODO: Validate the size or arrLine is = inputs+1 (bc label)
            	
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
            	// TODO: consider adding a threshold
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
		
		System.out.println(Arrays.toString(avg_error_vector.toArray())); // DBUG
		System.out.println(	"Score: " + correctGuesses + "/" + quantity + 
							" (" + correctGuesses*100/quantity + "%)"); // DBUG
		
		return avg_error_vector.toArray();
	}
	
	// Evaluate
	public void evaluateIMG(String path) {
		File imageFile = new File(path);
		try {
			BufferedImage image = ImageIO.read(imageFile);
			int width = image.getWidth();
			int height = image.getHeight(); //TODO: Validar las dimensiones de la imagen con el tamanio de inputs
			Color pixel;
			
			for(int i = 0; i < height; i++)
				for(int j = 0; j < width; j++) {
					pixel = new Color(image.getRGB(j, i));
					inputs.setQuick(i * width + j, 255 - ((pixel.getRed() +
														   pixel.getGreen() + 
														   pixel.getBlue())/3));
				}
			feed_forward();
		} catch(IOException e) {
			System.out.println("Couldn't read the given image.");
		}
	}
	
	public void save(String path_name) {
		if(path_name == null) return; // TODO: properly handle this
		
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
			print_writer.println();	
			
			print_writer.close();
		} catch (IOException e) {
			System.out.println("Error. Possible causes are: the named file exists but is a directory rather than a regular file, does not exist but cannot be created, or cannot be opened for any other reason");
		}
	}
	
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
			br.close();
		} catch(IOException e) {
			System.out.println("File not found.");
		}
		return nn;
	}
	
	public void configureMapping(Map<String, Integer> m) {
		labelsMap = m;
	}
	
	// Auxiliary Methods
	private void feed_forward() {
		hidden_layer = (DenseDoubleMatrix1D)algebra.mult(weights1, inputs);
		sigmoid(hidden_layer);
		outputs = (DenseDoubleMatrix1D)algebra.mult(weights2, hidden_layer);
		sigmoid(outputs);
		
//		System.out.println(outputs.toString()); //dbug - see outputs
//		System.out.println("Expected: \n" + expected_outputs.toString()); //dbug0
//		if(expected_outputs.getQuick(1) == 1) {
//			System.out.println(outputs.toString()); //dbug - see outputs
//			System.out.println("Expected: \n" + expected_outputs.toString());
//		}
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
