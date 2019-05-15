package neuralNetwork;

import java.util.HashMap;
import java.util.Set;
import java.util.Iterator;

public class TestZone {

	public static void main(String[] args) {
		/*
		Network n_high_lr = Network.load("C:\\Users\\Asus\\eclipse-workspace\\X0 NN\\Numbers_High_LR_100.txt");
		Network n_low_lr = Network.load("C:\\Users\\Asus\\eclipse-workspace\\X0 NN\\Numbers_Low_LR_100.txt");
		*/
		
		Network n = new Network(784, 26);
		
		// MAPPING
		HashMap<String, Integer> offset1Map = new HashMap<>();
		for(int i = 0; i < 26; i++) 
			offset1Map.put(""+(i+1), i);
		
		n.configureMapping(offset1Map);
		
		
		/*
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
		*/
		
		/*
		// TESTING
		System.out.println("///// TESTING /////\n\n");
		n_high_lr.testCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\train_digits_full.csv", 33_601, 42_000);
		n_low_lr.testCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Proyecto\\Data Sets\\train_digits_full.csv", 33_601, 42_000);
		*/
		
		/*
		//EVALUATE
		System.out.println("EVALUATE firstCSVNetwork");
		firstCSVNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
		 */
	
		// SAVE
		n.save("hashSavedTest.txt");
		Network newNetwork = Network.load("hashSavedTest.txt");
		
		/*
		//EVALUATE
		System.out.println("EVALUATE newNetwork");
		newNetwork.evaluateIMG("C:\\Users\\panch\\cursoJava\\workspace\\NN\\4.png");
		*/
	}
}
