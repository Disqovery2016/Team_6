

import java.util.ArrayList;
import java.util.HashMap;


public class KNN{
	public static void main(String[] args){
		ArrayList<KNN.DataEntry> data = new ArrayList<KNN.DataEntry>();
		//data.add(new DataEntry(new double[]{1,8.1,89} "safe"));
		data.add(new DataEntry(new double[]{1,8.1,89}, "safe"));
		data.add(new DataEntry(new double[]{1,8.5,122}, "safe"));
		data.add(new DataEntry(new double[]{1,9.0,195}, "safe"));
		data.add(new DataEntry(new double[]{1,9.2,207}, "safe"));
		data.add(new DataEntry(new double[]{1.3,9.5,236}, "safe"));
		data.add(new DataEntry(new double[]{1.4,8.8,336}, "safe"));
		data.add(new DataEntry(new double[]{2.1,7.9,380}, "safe"));
		data.add(new DataEntry(new double[]{1.8,8.3,408}, "safe"));
		data.add(new DataEntry(new double[]{20.2,0.8,899}, "unsafe"));
		data.add(new DataEntry(new double[]{13.1,0.9,714}, "unsafe"));
		data.add(new DataEntry(new double[]{14.5,4.1,1162}, "unsafe"));
		data.add(new DataEntry(new double[]{8.7,6.9,1211}, "unsafe"));
		data.add(new DataEntry(new double[]{8.9,6.4,1132}, "unsafe"));
		data.add(new DataEntry(new double[]{9.6,9.3,1252}, "unsafe"));
		data.add(new DataEntry(new double[]{17.7,5.0,1174}, "unsafe"));
		data.add(new DataEntry(new double[]{9.7,9.5,1044}, "unsafe"));
		data.add(new DataEntry(new double[]{8.7,10.5,640}, "unsafe"));
		data.add(new DataEntry(new double[]{4.0,9.4,470}, "safe"));
		data.add(new DataEntry(new double[]{1.8,7.4,540}, "safe"));
		KNN nn = new KNN(data, 3); //3 neighbours
		System.out.println("Classified as: "+nn.classify(new DataEntry(new double[]{20,10,1000},"Ignore")));
	}
	
	
	private int k;
	private ArrayList<Object> classes;
	private ArrayList<DataEntry> dataSet;
	
	public KNN(ArrayList<DataEntry> dataSet, int k){
		this.classes = new ArrayList<Object>();
		this.k = k;
		this.dataSet = dataSet;
		
		//Load different classes
		for(DataEntry entry : dataSet){
			if(!classes.contains(entry.getY())) classes.add(entry.getY());
		}
	}
	
	private DataEntry[] getNearestNeighbourType(DataEntry x){
		DataEntry[] retur = new DataEntry[this.k];
		double fjernest = Double.MIN_VALUE;
		int index = 0;
		for(DataEntry tse : this.dataSet){
			double distance = distance(x,tse);
			if(retur[retur.length-1] == null){ //Hvis ikke fyldt
				int j = 0;
				while(j < retur.length){
					if(retur[j] == null){
						retur[j] = tse; break;
					}
					j++;
				}
				if(distance > fjernest){
					index = j;
					fjernest = distance;
				}
			}
			else{
				if(distance < fjernest){
					retur[index] = tse;
					double f = 0.0;
					int ind = 0;
					for(int j = 0; j < retur.length; j++){
						double dt = distance(retur[j],x);
						if(dt > f){
							f = dt;
							ind = j;
						}
					}
					fjernest = f;
					index = ind;
				}
			}
		}
		return retur;
	}
	
	private static double convertDistance(double d){
		return 1.0/d;
	}

	
	public static double distance(DataEntry a, DataEntry b){
		double distance = 0.0;
		int length = a.getX().length;
		for(int i = 0; i < length; i++){
			double t = a.getX()[i]-b.getX()[i];
			distance = distance+t*t;
		}
		return Math.sqrt(distance);
	}
	
	public Object classify(DataEntry e){
		HashMap<Object,Double> classcount = new HashMap<Object,Double>();
		DataEntry[] de = this.getNearestNeighbourType(e);
		for(int i = 0; i < de.length; i++){
			double distance = KNN.convertDistance(KNN.distance(de[i], e));
			if(!classcount.containsKey(de[i].getY())){
				classcount.put(de[i].getY(), distance);
			}
			else{
				classcount.put(de[i].getY(), classcount.get(de[i].getY())+distance);
			}
		}
		//Find right choice
		Object o = null;
		double max = 0;
		for(Object ob : classcount.keySet()){
			if(classcount.get(ob) > max){
				max = classcount.get(ob);
				o = ob;
			}
		}
		
		return o;
	}

public static class DataEntry{
	private double[] x;
	private Object y;
	
	public DataEntry(double[] x, Object y){
		this.x = x;
		this.y = y;
	}
	
		public double[] getX(){
			return this.x;
		}
	
		public Object getY(){
			return this.y;
		}
	}
}
