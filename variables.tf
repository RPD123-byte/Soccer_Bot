variable "aws_region" {
  type        = string
  default     = "ap-south-1"
  description = "The AWS region where resources will be deployed."
}

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

variable "environment" {
  type        = string
  default     = "dev"
  description = "Environment name to append to resources to avoid conflicts (dev, staging, prod)."
}

variable "deploy_prometheus" {
  type        = bool
  default     = false
  description = "Whether to deploy Prometheus monitoring stack."
}

variable "alert_email" {
  type        = string
  default     = "your-email@example.com"
  description = "Email address for Prometheus alerts."
}

variable "existing_role_arn" {
  type        = string
  default     = ""
  description = "ARN of existing IAM role to use."
}

variable "oidc_provider_arn" {
  type        = string
  default     = ""
  description = "ARN of the OIDC provider for the EKS cluster."
}

variable "oidc_provider_url" {
  type        = string
  default     = ""
  description = "URL of the OIDC provider for the EKS cluster."
}
