/* Begin variables.tf update */

// Define AWS region variable
variable "aws_region" {
  type        = string
  default     = "ap-south-1"
  description = "The AWS region where resources will be deployed."
}

// Define EKS cluster related variables
variable "cluster_name" {
  type        = string
  default     = "example-eks-cluster"
  description = "Name of the EKS cluster."
}

variable "cluster_version" {
  type        = string
  default     = "1.21"
  description = "The version of the EKS cluster."
}

variable "instance_types" {
  type        = list(string)
  default     = ["t3.medium"]
  description = "A list of instance types for the EKS managed node group."
}

variable "node_count" {
  type        = number
  default     = 2
  description = "Desired number of nodes in the EKS managed node group."
}

/* Optional: Remove or comment out legacy variables that are no longer used */
// variable "region" {
//   type    = string
//   default = "us-east-1"
// }

// variable "instance_type" {
//   type    = string
//   default = "t2.micro"
// }

// variable "ami_id" {
//   type    = string
//   default = "ami-0c94855ba95c574c8" # Replace with your desired AMI ID
// }

// variable "key_name" {
//   type    = string
//   default = "my-key-pair"
// }

/* End variables.tf update */
