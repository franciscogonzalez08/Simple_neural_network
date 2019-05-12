package neuralNetwork;

public class TestZone {

	public static void main(String[] args) {
		Network firstCSVNetwork = new Network(784, 10, 0.1);
		
		
		// TRAINING
		int times = 100;
		while(times > 0)
		{
			firstCSVNetwork.trainCSV("C:\\Users\\panch\\cursoJava\\workspace\\NN\\train_digits2.csv", 1, 100);
			times--;
		}
		
		// TESTING
		System.out.println("\n\n///// TESTING /////\n\n");
		firstCSVNetwork.testCSV("C:\\Users\\panch\\cursoJava\\workspace\\NN\\train_digits2.csv", 1, 100);
		
		//EVALUATE
		System.out.println("EVALUATE firstCSVNetwork");
		firstCSVNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
		
		
		// SAVE
		firstCSVNetwork.save("NNlog.txt");
		Network newNetwork = Network.load("NNlog.txt");
		
		//EVALUATE
		System.out.println("EVALUATE newNetwork");
		newNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
	}
}
