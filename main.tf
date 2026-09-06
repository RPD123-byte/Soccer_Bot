# main.tf
# Provider configuration
provider "aws" {
  region = var.aws_region
}

# Reference the existing EKS cluster
data "aws_eks_cluster" "existing" {
  name = var.cluster_name
}

# Reference the existing ECR repository
data "aws_ecr_repository" "existing_repo" {
  name = "example-eks-repo"
}

# Reference existing VPC 
data "aws_vpc" "existing_vpc" {
  default = true  # You can change this to find a specific VPC by ID or tag
}

# Find existing subnets
data "aws_subnets" "existing" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.existing_vpc.id]
  }
}

# Configure the Kubernetes provider to use the correct authentication method
provider "kubernetes" {
  host                   = data.aws_eks_cluster.existing.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.existing.certificate_authority[0].data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"  # Updated from v1alpha1
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
  }
}

# Configure the Helm provider
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.existing.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.existing.certificate_authority[0].data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"  # Updated from v1alpha1
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
    }
  }
}

# Create local values for use in outputs and resources
locals {
  ecr_repo_url = data.aws_ecr_repository.existing_repo.repository_url
}

# Update Kubernetes ServiceAccount to link with the IAM role using annotation
resource "kubernetes_service_account" "example_sa" {
  metadata {
    name = "example-sa"
    annotations = {
      "eks.amazonaws.com/role-arn" = var.existing_role_arn
    }
  }
}

# Example Horizontal Pod Autoscaler (HPA) for a deployment named "example-deployment"
resource "kubernetes_horizontal_pod_autoscaler" "example_hpa" {
  metadata {
    name      = "example-hpa"
    namespace = "default"
  }
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "soccer-bot-deployment"
    }
    min_replicas = 2
    max_replicas = 5
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 50
        }
      }
    }
  }
}

# RBAC Configuration: Create a Role and RoleBinding
resource "kubernetes_role" "example_role" {
  metadata {
    name = "example-role"
  }
  rule {
    api_groups = [""]
    resources  = ["pods"]
    verbs      = ["get", "list", "watch"]
  }
}

resource "kubernetes_role_binding" "example_rb" {
  metadata {
    name = "example-role-binding"
  }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.example_role.metadata[0].name
  }
  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.example_sa.metadata[0].name
    namespace = "default"
  }
}

# Network Policy: Restrict ingress to pods with label app=example-app from pods labeled allowed-app
resource "kubernetes_network_policy" "example_np" {
  metadata {
    name = "deny-all-except-allowed"
    namespace = "default"
  }
  spec {
    pod_selector {
      match_labels = {
        app = "example-app"
      }
    }
    ingress {
      from {
        pod_selector {
          match_labels = {
            app = "allowed-app"
          }
        }
      }
      ports {
        port = 80
      }
    }
    policy_types = ["Ingress"]
  }
}

# Prometheus Monitoring: Deploy kube-prometheus-stack via a Helm release (conditionally)
resource "helm_release" "prometheus" {
  count            = var.deploy_prometheus ? 1 : 0
  name             = "prometheus-${var.environment}"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  
  values = [
    <<EOF
alertmanager:
  alertmanagerSpec:
    route:
      group_by: ['alertname']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 1h
    config: |
      global:
        resolve_timeout: 5m
      route:
        receiver: 'default-receiver'
        group_wait: 30s
        group_interval: 5m
        repeat_interval: 1h
      receivers:
      - name: 'default-receiver'
        email_configs:
        - to: '${var.alert_email}'

prometheus:
  prometheusSpec:
    ruleSelectorNilUsesHelmValues: false

additionalPrometheusRules:
  custom:
    groups:
      - name: custom.rules
        rules:
          - alert: HighCpuUsage
            expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 75
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High CPU usage detected"
              description: "CPU usage is over 75% on instance {{ $labels.instance }}"
          - alert: HighMemoryUsage
            expr: (node_memory_Active_bytes / node_memory_MemTotal_bytes * 100) > 75
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High Memory usage detected"
              description: "Memory usage is over 75% on instance {{ $labels.instance }}"
EOF
  ]
}
