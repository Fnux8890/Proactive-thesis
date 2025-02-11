public class Reading {
	
	private double timestamp;
	private double value;

	public Reading(double timestamp, double value) {
		this.timestamp = timestamp;
		this.value = value;
	}
	
	public double getTimeStamp() {
		return this.timestamp;
	}
	
	public double getValue() {
		return this.value;
	}
	
	public String toString() {
		return (new Double(timestamp)).longValue() + ": " + value;
	}
}