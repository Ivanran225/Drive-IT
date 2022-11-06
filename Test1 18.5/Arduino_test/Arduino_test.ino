int ENA = 7;
int IN1 = 6;
int IN2 = 5;

int IN3 = 4;
int IN4 = 3;
int ENB = 2;
void setup() {
  // put your setup code here, to run once:
 Serial.begin(9600);

  pinMode (ENA, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
 
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
  pinMode (ENB, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, HIGH);
  digitalWrite (ENA, HIGH);
  digitalWrite (ENB, HIGH);
}
