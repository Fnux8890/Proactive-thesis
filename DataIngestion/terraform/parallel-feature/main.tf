terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
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

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "use_preemptible" {
  description = "Use preemptible instances to save costs"
  type        = bool
  default     = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "feature-extraction-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "feature-extraction-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

# Firewall rules
resource "google_compute_firewall" "allow_internal" {
  name    = "allow-internal"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/24"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "allow_monitoring" {
  name    = "allow-monitoring"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["3001", "9090", "9400"]  # Grafana, Prometheus, DCGM
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["monitoring"]
}

# Service Account
resource "google_service_account" "feature_extraction" {
  account_id   = "feature-extraction-sa"
  display_name = "Feature Extraction Service Account"
}

resource "google_project_iam_member" "compute_admin" {
  project = var.project_id
  role    = "roles/compute.admin"
  member  = "serviceAccount:${google_service_account.feature_extraction.email}"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.feature_extraction.email}"
}

# Cloud Storage bucket for data
resource "google_storage_bucket" "data_bucket" {
  name          = "${var.project_id}-feature-extraction-data"
  location      = var.region
  force_destroy = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# A2 High-GPU Instance for Production Pipeline
resource "google_compute_instance" "production_pipeline" {
  name         = "greenhouse-production-a2"
  machine_type = "a2-highgpu-4g"
  zone         = var.zone

  tags = ["feature-extraction", "gpu", "monitoring"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 500
      type  = "pd-ssd"
    }
  }

  # Attach 4x A100 GPUs
  guest_accelerator {
    type  = "nvidia-tesla-a100"
    count = 4
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible        = var.use_preemptible
  }

  network_interface {
    network    = google_compute_network.vpc.id
    subnetwork = google_compute_subnetwork.subnet.id

    access_config {
      // Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.feature_extraction.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin = "TRUE"
    startup-script = file("${path.module}/startup.sh")
  }

  metadata_startup_script = templatefile("${path.module}/startup.sh", {
    project_id = var.project_id
    region     = var.region
    db_host    = google_sql_database_instance.timescale.private_ip_address
    db_pass    = random_password.db_password.result
  })
}

# Cloud SQL for TimescaleDB
resource "google_sql_database_instance" "timescale" {
  name             = "timescale-feature-extraction"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier = "db-highmem-8"  # 8 vCPUs, 52 GB RAM
    
    disk_size       = 500
    disk_type       = "PD_SSD"
    disk_autoresize = true

    database_flags {
      name  = "shared_preload_libraries"
      value = "timescaledb"
    }

    database_flags {
      name  = "max_connections"
      value = "500"
    }

    database_flags {
      name  = "shared_buffers"
      value = "13107200"  # 25% of RAM
    }

    database_flags {
      name  = "work_mem"
      value = "52428"  # 50MB
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      require_ssl     = true
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Database password
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "google_sql_user" "postgres" {
  name     = "postgres"
  instance = google_sql_database_instance.timescale.name
  password = random_password.db_password.result
}

resource "google_sql_database" "greenhouse" {
  name     = "greenhouse"
  instance = google_sql_database_instance.timescale.name
}

# Private VPC connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Outputs
output "instance_ip" {
  value = google_compute_instance.production_pipeline.network_interface[0].access_config[0].nat_ip
}

output "db_private_ip" {
  value = google_sql_database_instance.timescale.private_ip_address
}

output "db_connection_name" {
  value = google_sql_database_instance.timescale.connection_name
}

output "ssh_command" {
  value = "gcloud compute ssh --zone ${var.zone} ${google_compute_instance.production_pipeline.name} --project ${var.project_id}"
}

output "monitoring_urls" {
  value = {
    grafana    = "http://${google_compute_instance.production_pipeline.network_interface[0].access_config[0].nat_ip}:3001"
    prometheus = "http://${google_compute_instance.production_pipeline.network_interface[0].access_config[0].nat_ip}:9090"
  }
}