package neuralNetwork;

import java.util.HashMap;

public class TestZone {

	public static void main(String[] args) {
		Network n_high_lr = new Network(784, 10, 0.1);
		Network n_low_lr = new Network(784, 10, 0.001);
		
		/*
		// MAPPING
		HashMap<String, Integer> offset1Map = new HashMap<>();
		for(int i = 0; i < 26; i++) 
			offset1Map.put(""+(i+1), i);
		
		n.configureMapping(offset1Map);
		*/
		
		// TRAINING
		int times = 0;
		while(times < 1_000)
		{
			n_high_lr.trainCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\train_digits_full.csv", 1, 33_600);
			n_low_lr.trainCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\train_digits_full.csv", 1, 33_600);
			times++;
			
			System.out.println("Times = " + times);
			if(times%100 == 0)
			{
				n_high_lr.save("Numbers_High_LR_" + times + ".txt");
				n_low_lr.save("Numbers_Low_LR_" + times + ".txt");
			}
		}
		
		// TESTING
		System.out.println("///// TESTING /////\n\n");
		n_high_lr.testCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\train_digits_full.csv", 33_601, 42_000);
		n_low_lr.testCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\train_digits_full.csv", 33_601, 42_000);
		
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
