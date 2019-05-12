package neuralNetwork;

public class TestZone {

	public static void main(String[] args) {
		Network firstCSVNetwork = new Network(784, 10, 0.1);
		
		// TRAINING
		int times = 1_000;
		while(times > 0)
		{
			firstCSVNetwork.trainCSV("C:\\Users\\panch\\cursoJava\\workspace\\NN\\train_digits2.csv", 1, 100);
			times--;
		}
		
		// TESTING
		System.out.println("\n\n///// TESTING /////\n\n");
		firstCSVNetwork.testCSV("C:\\Users\\panch\\cursoJava\\workspace\\NN\\train_digits2.csv", 1, 100);
		
		//EVALUATE
		System.out.println("EVALUATE");
		firstCSVNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\6v2.png");
	}
}
