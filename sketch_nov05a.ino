//DISQovery 2016
//Team Jalvayu..
//Device name-- Jal Niyantrak
//by Abhishek Baghel
#include <SoftwareSerial.h>
#include <stdlib.h>
// special random variable
long randNumber;
// LED
int ledPin = 13;

// replace with your channel's thingspeak API key
String apiKey = "03NUI4WGDRGQ0GOP";

// connect 2 to TX of Serial USB
// connect 3 to RX of serial USB
SoftwareSerial ser(2,3); // RX, TX

// this runs once
void setup() {                
  // initialize the digital pin as an output.
  pinMode(ledPin, OUTPUT);    

  // enable debug serial
  Serial.begin(115200);
  // enable software serial
  ser.begin(115200);
  //random data generator
  randomSeed(analogRead(0));
  // reset ESP8266
  ser.println("AT+RST");
}


// the loop
void loop() {

  // blink LED on board
  digitalWrite(ledPin, HIGH);   
  delay(200);               
  digitalWrite(ledPin, LOW);
// print a random number from 0 to 12 is the range of BOD
  randNumber = random(0,12);
  float bod   = randNumber;  

  // print a random number from 0 to 25 is the range of DO
  randNumber = random(0, 25);
  float disso   = randNumber;
  // print a random number from 0 to 2500 is the range of cunductivity
  randNumber = random(0, 2500);
  float cund   = randNumber;
  


  //data to string conversion
  char buf[16];
  String strbod = dtostrf(bod, 2, 2, buf);
  String strdisso = dtostrf(disso, 2, 2, buf);
  String strcund = dtostrf(cund, 2, 2, buf);
  
  Serial.println(bod);
  Serial.println(disso);
  Serial.println(cund);

  // TCP connection
  String cmd = "AT+CIPSTART=\"TCP\",\"";
  cmd += "184.106.153.149"; // api.thingspeak.com
  cmd += "\",80";
  ser.println(cmd);

  if(ser.find("Error")){
    Serial.println("AT+CIPSTART error");
    return;
  }

  // prepare GET string
  String getStr = "GET /update?api_key=";
  getStr += apiKey;
  getStr +="&field1=";
  getStr += String(strbod);
  getStr +="&field2=";
  getStr += String(strdisso);
  getStr +="&field3=";
  getStr += String(strcund);
  getStr += "\r\n\r\n";

  // send data length
  cmd = "AT+CIPSEND=";
  cmd += String(getStr.length());
  ser.println(cmd);

  if(ser.find(">")){
    ser.print(getStr);
  }
  else{
    ser.println("AT+CIPCLOSE");
    // alert user
    Serial.println("AT+CIPCLOSE");
  }

  // thingspeak needs 15 sec delay between updates
  delay(16000);  
}
