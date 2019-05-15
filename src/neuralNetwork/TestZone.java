package neuralNetwork;

public class TestZone {

	public static void main(String[] args) {
		Network firstCSVNetwork = new Network(784, 10, 0.1);
		
		
		// TRAINING
		int times = 10;
		while(times > 0)
		{
			firstCSVNetwork.trainCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\Letters_train.csv", 1, 1000);
			times--;
		}
		
		// TESTING
		System.out.println("\n\n///// TESTING /////\n\n");
		firstCSVNetwork.testCSV("C:\\\\Users\\\\Asus\\\\OneDrive\\\\Documents\\\\Learning\\\\5-Universidad\\\\4 Semestre\\\\Orientada a Objetos\\\\Proyecto\\\\Data Sets\\\\Letters_train.csv", 1001, 200);
		
		/*
		//EVALUATE
		System.out.println("EVALUATE firstCSVNetwork");
		firstCSVNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
		*/
		
		
		// SAVE
		firstCSVNetwork.save("NNlog.txt");
		Network newNetwork = Network.load("NNlog.txt");
		
		//EVALUATE
		System.out.println("EVALUATE newNetwork");
		newNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
	}
}
