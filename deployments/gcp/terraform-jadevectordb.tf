# Terraform configuration for JadeVectorDB deployment on GCP

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name for the GKE cluster"
  type        = string
  default     = "jadevectordb-cluster"
}

variable "node_count" {
  description = "Number of nodes in the cluster"
  type        = number
  default     = 3
}

variable "node_machine_type" {
  description = "Machine type for cluster nodes"
  type        = string
  default     = "n1-standard-4"
}

variable "disk_size_gb" {
  description = "Disk size for each node in GB"
  type        = number
  default     = 100
}

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster"
  type        = string
  default     = "1.27"
}

# Create a VPC network
resource "google_compute_network" "jadevectordb_network" {
  name                    = "jadevectordb-network"
  auto_create_subnetworks = false
}

# Create subnetwork
resource "google_compute_subnetwork" "jadevectordb_subnetwork" {
  name          = "jadevectordb-subnetwork"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.jadevectordb_network.id
}

# Create GKE cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = "${var.region}-a"

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.jadevectordb_network.name
  subnetwork = google_compute_subnetwork.jadevectordb_subnetwork.name

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = var.disk_size_gb
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# Create node pool for the cluster
resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.cluster_name}-node-pool"
  location   = "${var.region}-a"
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = var.disk_size_gb
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }
}

# Kubernetes provider configuration to deploy JadeVectorDB
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
}

# Deploy JadeVectorDB as a StatefulSet
resource "kubernetes_stateful_set_v1" "jadevectordb" {
  metadata {
    name      = "jadevectordb"
    namespace = "default"
  }

  spec {
    service_name = "jadevectordb-service"
    replicas     = var.node_count

    selector {
      match_labels = {
        app = "jadevectordb"
      }
    }

    template {
      metadata {
        labels = {
          app = "jadevectordb"
        }
      }

      spec {
        container {
          name  = "jadevectordb"
          image = "jadevectordb/jadevectordb:latest"

          port {
            container_port = 8080
            name           = "http"
          }

          port {
            container_port = 8081
            name           = "rpc"
          }

          env {
            name  = "JADE_DB_PORT"
            value = "8080"
          }

          env {
            name  = "JADE_DB_RPC_PORT"
            value = "8081"
          }

          env {
            name  = "JADE_DB_LOG_LEVEL"
            value = "INFO"
          }

          env {
            name  = "JADE_DB_CLUSTER_SIZE"
            value = var.node_count
          }

          env {
            name  = "JADE_DB_DATA_DIR"
            value = "/data"
          }

          env {
            name  = "JADE_DB_CONFIG_DIR"
            value = "/config"
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 60
            period_seconds        = 30
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          volume_mount {
            name       = "data-volume"
            mount_path = "/data"
          }

          volume_mount {
            name       = "config-volume"
            mount_path = "/config"
          }
        }

        volume {
          name = "config-volume"

          empty_dir {}
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "data-volume"
      }

      spec {
        access_modes = ["ReadWriteOnce"]

        resources {
          requests = {
            storage = "100Gi"
          }
        }
      }
    }
  }

  depends_on = [
    google_container_node_pool.primary_nodes
  ]
}

# Create a LoadBalancer service for JadeVectorDB
resource "kubernetes_service_v1" "jadevectordb_service" {
  metadata {
    name      = "jadevectordb-service"
    namespace = "default"
  }

  spec {
    selector = {
      app = kubernetes_stateful_set_v1.jadevectordb.metadata[0].labels.app
    }

    port {
      name        = "http"
      port        = 8080
      target_port = 8080
    }

    port {
      name        = "rpc"
      port        = 8081
      target_port = 8081
    }

    type = "LoadBalancer"
  }

  depends_on = [
    kubernetes_stateful_set_v1.jadevectordb
  ]
}

# Output values
output "cluster_endpoint" {
  value = google_container_cluster.primary.endpoint
}

output "cluster_ca_certificate" {
  value = google_container_cluster.primary.master_auth.0.cluster_ca_certificate
}

output "service_external_ip" {
  value = kubernetes_service_v1.jadevectordb_service.status.0.load_balancer.0.ingress.0.0.ip
}