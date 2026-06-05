# outputs.tf with correct references

output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = local.ecr_repo_url
}

output "cluster_endpoint" {
  description = "Endpoint for EKS cluster"
  value       = data.aws_eks_cluster.existing.endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = data.aws_eks_cluster.existing.certificate_authority[0].data
}

output "vpc_id" {
  description = "ID of the VPC used"
  value       = data.aws_vpc.existing_vpc.id
}

output "subnet_ids" {
  description = "IDs of the subnets used"
  value       = data.aws_subnets.existing.ids
}

output "service_account_name" {
  description = "Name of the Kubernetes service account"
  value       = kubernetes_service_account.example_sa.metadata[0].name
}

output "iam_role_arn" {
  description = "ARN of the IAM role used for RBAC"
  value       = var.existing_role_arn
}
