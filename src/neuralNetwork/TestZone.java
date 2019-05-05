package neuralNetwork;

public class TestZone {

	public static void main(String[] args) {
		TestCase c1 = new TestCase(new double[][] {{1,1,1,1,1}, 
												   {1,0,0,0,1}, 
												   {1,0,0,0,1}, 
												   {1,0,0,0,1},
												   {1,1,1,1,1}}, new double[] {1.0, 0.0});		

		TestCase c2 = new TestCase(new double[][] {{1,0,0,0,1}, 
									               {0,1,0,1,0}, 
									               {0,0,1,0,0}, 
									               {0,1,0,1,0},
									               {1,0,0,0,1}}, new double[] {0.0, 1.0});
		
		TestCase c3 = new TestCase(new double[][] {{0,1,1,0,0}, 
									               {1,0,0,0,1}, 
									               {1,0,0,0,1}, 
									               {1,0,0,0,1},
									               {1,1,0,1,0}}, new double[] {1.0, 0.0});
		
		TestCase c4 = new TestCase(new double[][] {{1,0,1,0,1}, 
									               {0,1,1,0,0}, 
									               {0,0,1,0,0}, 
									               {0,1,0,1,1},
									               {1,1,1,0,1}}, new double[] {0.0, 1.0});
		
		TestCase c5 = new TestCase(new double[][] { {1,1,1,1,1}, 
										            {1,0,0,0,0}, 
										            {1,0,0,0,1}, 
										            {0,0,1,0,1},
										            {1,1,1,1,1}}, new double[] {1.0, 0.0});
		
		TestCase c6 = new TestCase(new double[][] { {1,0,0,0,0}, 
										            {0,1,0,1,0}, 
										            {0,0,1,0,0}, 
										            {0,1,0,1,0},
										            {1,0,0,0,1}}, new double[] {0.0, 1.0});
		
		TestCase c9 = new TestCase(new double[][] {{0.1,0,0,0.1,0.2}, 
									               {0,0.6,1,0.5,0}, 
									               {0,0.2,0.8,0.8,0}, 
									               {0,1,0.2,1,0},
									               {0.1,0,0,0.1,0.2}}, new double[] {0.0, 1.0});
		
		Network X0 = new Network(25, 2, 0.1); // valid range: [0.001, 0.1]
		
		training(X0, c1, c2, c3, c4);
		
		System.out.println("*Drum rolls*");
		X0.classify(c5);
		X0.classify(c6);
		X0.classify(c9);
	}
	
	public static void training(Network network, TestCase ... cases) {
		int testQuantity = cases.length;
		
		int i = 10000;
		int testIndex;
		while(i > 0) {
			testIndex = (int)Math.floor(Math.random()*testQuantity);
			System.out.println("Test case: " + (testIndex+1));
			network.train(cases[testIndex]);
			i--;
		}
	}
}
