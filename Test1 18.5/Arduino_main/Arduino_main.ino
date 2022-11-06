int ENA = 7;
int IN1 = 6;
int IN2 = 5;

int IN3 = 4;
int IN4 = 3;
int ENB = 2;
int VELOCIDAD;

String x;
void setup () {

  Serial.begin(9600);

  pinMode (ENA, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
 
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
  pinMode (ENB, OUTPUT);
}

void adelante (){
 
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, HIGH);
  analogWrite  (ENA, 255);
  analogWrite  (ENB, 255);
  delay (500);
}
void atras (){

  digitalWrite (IN1, HIGH);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
  analogWrite  (ENA, 255);
  analogWrite  (ENB, 255);
  delay (500);
}
void izquierda (){

  digitalWrite (IN1, HIGH);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, HIGH);
  analogWrite  (ENA, 255);
  analogWrite  (ENB, 255);
  delay (500);
}
void derecha (){

  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
  analogWrite  (ENA, 255);
  analogWrite  (ENB, 255);
  delay (500);
}

void loop () { 
  while (Serial.available() == 0) {}     //wait for data available
  String teststr = Serial.readString();  //read until timeout
  teststr.trim();                        // remove any \r \n whitespace at the end of the String

  if (teststr == "w") {
    adelante();
    Serial.println("Adelante");
  } 

  if (teststr == "a") {
    izquierda();
    Serial.println("Izquierda");
 }
  if (teststr == "d") {
    derecha();
    Serial.println("Derecha");
 }
   if (teststr == "s") {
    atras();
    Serial.println("Atras");
 }
 
}