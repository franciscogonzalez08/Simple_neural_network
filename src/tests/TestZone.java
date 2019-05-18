package tests;

import java.util.HashMap;

import neuralNetwork.Network;

public class TestZone {

	public static void main(String[] args) {	
//		//We'll create a new neural network
//		Network loserNetwork = new Network(784, 26, 0.001);
//		
//		// We'll configure the mapping
//		HashMap<String, Integer> offset1Map = new HashMap<>();
//		for(int i = 0; i < 26; i++) 
//			offset1Map.put(""+(i+1), i);
//
//		loserNetwork.configureMapping(offset1Map);
//		
//		// We'll have it do a couple push-ups
//		int times = 0;
//        do
//        {
//            times++;
//            loserNetwork.trainCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 1, 71_040); //80% is used for training and 20% for testing
//        } while(times <= 2);
//		
//		// And we'll save it just in case it turns out to be really good (plot twist: it won't but shhh)
//		loserNetwork.save("C:\\Users\\panch\\cursoJava\\workspace\\X0 NN\\savedNetworks\\loserNetwork.txt");
//		
//		//We'll import more experienced networks to compete
//		Network sassyNetwork200 = Network.load("C:\\Users\\panch\\cursoJava\\workspace\\X0 NN\\savedNetworks\\Letters_200.txt");
//		Network moodyNetwork400 = Network.load("C:\\Users\\panch\\cursoJava\\workspace\\X0 NN\\savedNetworks\\Letters_400.txt");
//		Network wiseNetwork600 = Network.load("C:\\Users\\panch\\cursoJava\\workspace\\X0 NN\\savedNetworks\\Letters_600.txt");
		Network kawaiiNetwork800 = Network.load("C:\\Users\\panch\\cursoJava\\workspace\\X0 NN\\savedNetworks\\Letters_800.txt");
//		Network winnerNetwork1000 = Network.load("C:\\Users\\panch\\cursoJava\\workspace\\X0 NN\\savedNetworks\\Letters_1000.txt");
//		
//		// Let the competition begin! May the best network win c:
//		System.out.println("Loser network");
//		loserNetwork.testCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 71_041, 88_000);
//		
//		System.out.println("Sassy network");
//		sassyNetwork200.testCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 71_041, 88_000);
//		
//		System.out.println("Moody network");
//		moodyNetwork400.testCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 71_041, 88_000);
//		
//		System.out.println("Wise network");
//		wiseNetwork600.testCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 71_041, 88_000);
//		
//		System.out.println("Kawaii network");
//		kawaiiNetwork800.testCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 71_041, 88_000);
//		
//		System.out.println("Winner network");
//		winnerNetwork1000.testCSV("C:\\Users\\panch\\Downloads\\Letters_train.csv", 71_041, 88_000);
		
		//EVALUATE
		kawaiiNetwork800.evaluateIMG("C:\\Users\\panch\\Desktop\\TESTCASES\\test.png");
	
	}
}
