package neuralNetwork;

import java.util.HashMap;

public class TestZone {

	public static void main(String[] args) {
		Network n = new Network(784, 26, 0.1);
		
		// MAPPING
		HashMap<String, Integer> offset1Map = new HashMap<>();
		for(int i = 0; i < 26; i++) 
			offset1Map.put(""+(i+1), i);
		
		n.configureMapping(offset1Map);
		
		// TRAINING
		int times = 0;
		while(times < 10)
		{
			n.trainCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\Letters_train.csv", 1, 1000);
			times++;
			// Save every other iteration
			if(times%5 == 0)
				n.save("Letters_" + times + ".txt");
		}
		
		// TESTING
		System.out.println("///// TESTING /////\n\n");
		n.testCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\Digits_test.csv", 1, 1000);
		
		/*
		//EVALUATE
		System.out.println("EVALUATE firstCSVNetwork");
		firstCSVNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
	
		// SAVE
		Network newNetwork = Network.load("NNlog.txt");
		
		//EVALUATE
		System.out.println("EVALUATE newNetwork");
		newNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
		*/
	}
}
