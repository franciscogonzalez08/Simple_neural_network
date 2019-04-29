package neuralNetwork;

public class TestCase {
	private double inputs[];
	private double outputs[];
	
	public TestCase(double inputs[][], double output[]) {
		this.inputs = new double[inputs.length * inputs[0].length];
		
		for(int i = 0; i < inputs.length; i++)
			for(int j = 0; j < inputs[0].length; j++)
				this.inputs[i * inputs[0].length + j] = inputs[i][j];
		
		this.outputs = output.clone();
	}

	public double[] getInputs() {
		return inputs;
	}

	public double[] getOutput() {
		return outputs;
	}
}
