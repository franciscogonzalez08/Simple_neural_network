package neuralNetwork;

import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public class Network {
	private DenseDoubleMatrix1D inputs;
	private DenseDoubleMatrix2D weights1;
	private DenseDoubleMatrix2D weights2;
	private DenseDoubleMatrix1D outputs;
	private DenseDoubleMatrix1D hidden_layer;
	private final double LEARNING_RATE; // valid range: [0.001, 0.1]
	Algebra algebra = new Algebra();
	
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
		
		this.weights1 = new DenseDoubleMatrix2D(weights1);
		outputs = new DenseDoubleMatrix1D(outputSize);
		hidden_layer = new DenseDoubleMatrix1D(middleNeurons);
		this.weights2 = new DenseDoubleMatrix2D(weights2);
	}
	
	//Train
	public void train(TestCase t) {
		//feed forward
		this.inputs = t.getInputs();
		hidden_layer = (DenseDoubleMatrix1D)algebra.mult(weights1, inputs);
		sigmoid(hidden_layer);
		outputs = (DenseDoubleMatrix1D)algebra.mult(weights2, hidden_layer);
		sigmoid(outputs);
		System.out.println(outputs.toString()); //dbug
		
		//Calculate the errors
		DenseDoubleMatrix1D output_errors = subtract(t.getOutputs(), outputs);
		
		//Adjust weights2 
		
		// delta = sigmoid'(outputs)*output_errors
		DenseDoubleMatrix1D output_delta = cross_product(dSigmoid(outputs), output_errors);
		
		// new_weight = hidden_layer * output_layer_delta * learning_rate
		scalar_product(output_delta, LEARNING_RATE);
		
		int oSize = outputs.size();
		int hSize = hidden_layer.size();
		for(int o = 0; o < oSize; o++)
			for(int h = 0; h < hSize; h++)
				weights2.setQuick(o, h, weights2.getQuick(o, h) + /*sum?*/
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
				weights1.setQuick(h, i, weights1.getQuick(h, i) + /*sum?*/
				inputs.getQuick(i)*hidden_delta.getQuick(h));
		
	}
	
	//Evaluate
	public void classify(TestCase t) {
		this.inputs = t.getInputs();
		hidden_layer = (DenseDoubleMatrix1D)algebra.mult(weights1, inputs);
		sigmoid(hidden_layer);
		outputs = (DenseDoubleMatrix1D)algebra.mult(weights2, hidden_layer);
		sigmoid(outputs);
		System.out.println(outputs.toString()); //dbug
	}
	
	//Auxiliary methods
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
	
	private DenseDoubleMatrix1D subtract(DenseDoubleMatrix1D m1, DenseDoubleMatrix1D m2) {
		DenseDoubleMatrix1D m3 = new DenseDoubleMatrix1D(m1.size());
		for(int i = 0; i < m1.size(); i++)
			m3.setQuick(i, m1.getQuick(i) - m2.getQuick(i));
		return m3;
	}
}
