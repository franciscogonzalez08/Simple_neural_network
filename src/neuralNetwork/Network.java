package neuralNetwork;

import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public class Network {
	private DenseDoubleMatrix1D inputs;
	private DenseDoubleMatrix2D weights1;
	private DenseDoubleMatrix2D weights2;
	private DenseDoubleMatrix1D outputs;
	private DenseDoubleMatrix1D layer1;
	Algebra algebra = new Algebra();
	
	//Builders
	public Network(int inputSize, int outputSize) {
		this(inputSize, (int)Math.ceil((inputSize + outputSize)/2.0), outputSize);		
	}
	
	public Network(int inputSize, int middleNeurons, int outputSize) {
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
		layer1 = new DenseDoubleMatrix1D(middleNeurons);
		this.weights2 = new DenseDoubleMatrix2D(weights2);
	}
	
	//Train
	public void train(TestCase t) {
		//feed forward
		this.inputs = t.getInputs();
		layer1 = (DenseDoubleMatrix1D)algebra.mult(weights1, inputs);
		sigmoid(layer1);
		outputs = (DenseDoubleMatrix1D)algebra.mult(weights2, layer1);
		sigmoid(outputs);
		System.out.println(outputs.toString()); //dbug
		
		//Calculate the errors
		DenseDoubleMatrix1D output_errors = subtract(t.getOutputs(), outputs);
		
		//Adjust weights
		
	}
	//Auxiliary methods
	private void sigmoid(DenseDoubleMatrix1D m1) {
		for(int i = 0; i < m1.size(); i++)
			m1.setQuick(i, 1/(1+Math.pow(Math.E, -m1.getQuick(i))));
	}
	
	private DenseDoubleMatrix1D subtract(DenseDoubleMatrix1D m1, DenseDoubleMatrix1D m2) {
		DenseDoubleMatrix1D m3 = new DenseDoubleMatrix1D(m1.size());
		for(int i = 0; i < m1.size(); i++)
			m3.setQuick(i, m1.getQuick(i) - m2.getQuick(i));
		return m3;
	}
}
