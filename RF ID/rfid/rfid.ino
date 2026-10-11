#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 10
#define RST_PIN 9

MFRC522 rfid(SS_PIN, RST_PIN);

// =================================
// REGISTERED EMPLOYEES
// =================================

// Employee 1
byte employee1UID[] = {0xB3, 0xC7, 0x01, 0x07};
String employee1Name = "SOUMYA";
String employee1ID = "EMP102";

// Employee 2
byte employee2UID[] = {0x6D, 0x7C, 0xF7, 0x06};
String employee2Name = "SMRUTI";
String employee2ID = "EMP103";

// Employee 3
byte employee3UID[] = {0xB6, 0x2B, 0xF5, 0x06};
String employee3Name = "GEETA";
String employee3ID = "EMP104";

// Employee 4
byte employee4UID[] = {0x55, 0xAB, 0xF5, 0x06};
String employee4Name = "SIBA";
String employee4ID = "EMP105";

// Employee 5
byte employee5UID[] = {0xAC, 0x89, 0xF5, 0x06};
String employee5Name = "OMM";
String employee5ID = "EMP106";

// Employee 6
byte employee6UID[] = {0xCF, 0xAB, 0x42, 0x1C};
String employee6Name = "SOURAV";
String employee6ID = "EMP101";


// =================================
// SETUP
// =================================

void setup() {

  Serial.begin(9600);

  SPI.begin();
  rfid.PCD_Init();

  Serial.println("==============================");
  Serial.println(" RFID EMPLOYEE ATTENDANCE");
  Serial.println("==============================");

  Serial.println("6 Employees Registered");
  Serial.println("Scan Employee RFID Card...");
}


// =================================
// LOOP
// =================================

void loop() {

  // Check if RFID card is present
  if (!rfid.PICC_IsNewCardPresent()) {
    return;
  }

  // Read RFID card
  if (!rfid.PICC_ReadCardSerial()) {
    return;
  }

  Serial.println();
  Serial.println("------------------------------");

  Serial.print("CARD UID: ");

  for (byte i = 0; i < rfid.uid.size; i++) {

    if (rfid.uid.uidByte[i] < 0x10) {
      Serial.print("0");
    }

    Serial.print(rfid.uid.uidByte[i], HEX);

    if (i < rfid.uid.size - 1) {
      Serial.print(":");
    }
  }

  Serial.println();


  // =================================
  // CHECK EMPLOYEE 1
  // =================================

  if (checkUID(employee1UID)) {

    attendance(employee1Name, employee1ID);
  }


  // =================================
  // CHECK EMPLOYEE 2
  // =================================

  else if (checkUID(employee2UID)) {

    attendance(employee2Name, employee2ID);
  }


  // =================================
  // CHECK EMPLOYEE 3
  // =================================

  else if (checkUID(employee3UID)) {

    attendance(employee3Name, employee3ID);
  }


  // =================================
  // CHECK EMPLOYEE 4
  // =================================

  else if (checkUID(employee4UID)) {

    attendance(employee4Name, employee4ID);
  }


  // =================================
  // CHECK EMPLOYEE 5
  // =================================

  else if (checkUID(employee5UID)) {

    attendance(employee5Name, employee5ID);
  }


  // =================================
  // CHECK EMPLOYEE 6
  // =================================

  else if (checkUID(employee6UID)) {

    attendance(employee6Name, employee6ID);
  }


  // =================================
  // UNKNOWN CARD
  // =================================

  else {

    Serial.println("==============================");
    Serial.println(" UNKNOWN CARD");
    Serial.println("==============================");

    Serial.println("EMPLOYEE: UNKNOWN");
    Serial.println("EMPLOYEE ID: UNKNOWN");
    Serial.println("STATUS: NOT REGISTERED");

    Serial.println("==============================");

    // Do NOT open camera for unknown card
    Serial.println("CAMERA: NOT_OPENED");
  }


  // Stop RFID communication
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  delay(2000);

  Serial.println();
  Serial.println("Scan Employee RFID Card...");
}


// =================================
// CHECK UID FUNCTION
// =================================

bool checkUID(byte registeredUID[]) {

  if (rfid.uid.size != 4) {
    return false;
  }

  for (byte i = 0; i < 4; i++) {

    if (rfid.uid.uidByte[i] != registeredUID[i]) {
      return false;
    }
  }

  return true;
}


// =================================
// ATTENDANCE FUNCTION
// =================================

void attendance(String employeeName, String employeeID) {

  Serial.println("==============================");
  Serial.println(" EMPLOYEE ATTENDANCE");
  Serial.println("==============================");

  Serial.println("RFID CARD: MATCHED");

  Serial.print("EMPLOYEE NAME: ");
  Serial.println(employeeName);

  Serial.print("EMPLOYEE ID: ");
  Serial.println(employeeID);

  Serial.println("STATUS: PRESENT");

  // =================================
  // SEND CAMERA COMMAND TO LAPTOP
  // =================================

  Serial.println("OPEN_CAMERA");

  Serial.println("==============================");
}
