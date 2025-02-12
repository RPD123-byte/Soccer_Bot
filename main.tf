provider "aws" {
  region = "us-east-1"
}

resource "aws_security_group" "soccer_bot_sg" {
  name        = "soccer_bot_sg"
  description = "Allow SSH and HTTP traffic"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "soccer_instance" {
  ami                         = "ami-0c55b159cbfafe1f0"
  instance_type               = "t2.micro"
  key_name                    = "my-key-pair"
  vpc_security_group_ids      = [aws_security_group.soccer_bot_sg.id]
  associate_public_ip_address = true

  tags = {
    Name = "SoccerBot-EC2"
  }
}
