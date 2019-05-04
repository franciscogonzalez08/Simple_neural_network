package neuralNetwork;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;

public class TestCase {
	private DenseDoubleMatrix1D inputs;
	private DenseDoubleMatrix1D outputs;
	
	//Builder
	public TestCase(double inputs[][], double outputs[]) {
		this.inputs = new DenseDoubleMatrix1D(inputs.length * inputs[0].length);
		
		for(int i = 0; i < inputs.length; i++)
			for(int j = 0; j < inputs[0].length; j++)
				this.inputs.setQuick(i * inputs[0].length + j, inputs[i][j]);
		
		this.outputs = new DenseDoubleMatrix1D(outputs);
	}

	//Getters
	public DenseDoubleMatrix1D getInputs() {
		return inputs;
	}

	public DenseDoubleMatrix1D getOutputs() {
		return outputs;
	}
}
