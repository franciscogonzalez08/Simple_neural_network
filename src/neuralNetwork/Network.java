package neuralNetwork;

public class Network {
	private double inputs[];
	private double weights1[][];
	private double weights2[][];
	private double outputs[];
	private double layer1[];
	
	
	//Builders
	public Network(int inputSize, int outputSize) {
		this(inputSize, (int)Math.ceil((inputSize + outputSize)/2.0), outputSize);		
	}
	
	public Network(int inputSize, int middleNeurons, int outputSize) {
		weights1 = new double[inputSize][middleNeurons];
		outputs = new double[outputSize];
		layer1 = new double[middleNeurons];
		weights2 = new double[middleNeurons][outputSize];
		
		for(int i = 0; i < weights1.length; i++)
			for(int j = 0; j < weights1[i].length; j++)
				weights1[i][j] = (Math.random() / 5) - 0.1;

		for(int i = 0; i < weights2.length; i++)
			for(int j = 0; j < weights2[i].length; j++)
				weights2[i][j] = (Math.random() / 5) - 0.1;
	}
	
	//Train
	public double[] train(TestCase t/*double inputs[], double expectedoutput*/) {
		this.inputs = t.getInputs().clone();
		layer1 = matrixMult(new double[][] {this.inputs}, weights1)[0];
		sigmoid(layer1);
		outputs = matrixMult(new double[][] {this.layer1}, weights2)[0];
		sigmoid(outputs);
		return outputs;
	}

	private double[][] matrixMult(double m1[][], double m2[][]) {
		double m3[][] = new double[m1.length][m2[0].length];
		double dotproduct = 0;
		for(int k = 0; k < m1.length; k++)
			for(int j = 0; j < m2[0].length; j++) {
				for(int i = 0; i < m1[0].length; i++)
					dotproduct += m1[k][i] * m2[i][j];
				m3[k][j] = dotproduct;
			}
		return m3;
	}
	
	private void sigmoid(double [] m1) {
		for(int i = 0; i < m1.length; i++)
			m1[i] = 1/(1+Math.pow(Math.E, -m1[i]));
	}
}
