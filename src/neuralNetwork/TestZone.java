package neuralNetwork;

public class TestZone {

	public static void main(String[] args) {
		Network firstCSVNetwork = new Network(784, 10, 0.1);
		
		int times = 100;
		while(times > 0)
		{
			firstCSVNetwork.trainCSV("C:\\Users\\Asus\\OneDrive\\Documents\\Learning\\5-Universidad\\4 Semestre\\Orientada a Objetos\\Number Identificating NN Project\\Data Sets\\train_digits2.csv", 1, 2);
			times--;
		}
	}
}
