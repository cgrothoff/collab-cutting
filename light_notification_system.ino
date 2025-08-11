#include <SoftwareSerial.h>
#include <Adafruit_NeoPixel.h>

// Bluetooth Module Set up
SoftwareSerial BTSerial(10, 11); // RX | TX

// LED Light Strip Set Up
#define PIN 6 
#define NUM_LED 8 
int color = 0;
Adafruit_NeoPixel strip(NUM_LED, PIN, NEO_GRB + NEO_KHZ800); 

// Force Sensor Set Up
const int force_pin = A0;
const int force_threshold = 450; // Resting Value: 200-400
unsigned long force_exceeded_start = 0;
bool force_triggered = false;

void setup() {
  Serial.begin(9600);
  BTSerial.begin(9600);

  strip.begin();

  Serial.println("Ready to receive Bluetooth commands!");
}

void loop() {
  int force_val = analogRead(force_pin);
  //Serial.println(force_val);
  
  if(!force_triggered && force_val > force_threshold){
    if(force_exceeded_start == 0) {
      force_exceeded_start = millis();
    } else if (millis() - force_exceeded_start > 500){
      force_triggered = true;

      BTSerial.println("1");
      Serial.println("Force exceeded for 0.5s - sending STOP");

      setLEDColor(255, 0, 0);
    }
  } 
  
  if(force_val <= force_threshold) {
    force_exceeded_start = 0;
  } 

  // Check for Bluetooth
  if (BTSerial.available()) {
    char receivedChar = BTSerial.read();  // Read one byte of data
    //Serial.print("Received: ");
    //Serial.println(receivedChar);
    
    if(receivedChar == 'q') {
      Serial.println("Resetting force trigger state.");
      force_triggered = false;
      force_exceeded_start = 0;
      setLEDColor(0, 255, 0);
    }

    if(!force_triggered) {
      switch(receivedChar) {
      case '1':   // RED
        setLEDColor(255, 0, 0);
        delay(50); 
      break;
      
      case '2':   // YELLOW
        setLEDColor(255, 255, 0);
        delay(50);
      break;

      case '3':   // GREEN
        setLEDColor(0, 255, 0);
        delay(50); 
      break;
    }
    }
  }
}

void setLEDColor(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < NUM_LED; i++) {
    strip.setPixelColor(i, r, g, b);
  }
  strip.show();
}
