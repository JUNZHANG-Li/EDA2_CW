# --- Data Sources ---
data "harvester_image" "img" {
  display_name = var.img_display_name
  namespace    = "harvester-public"
}

data "harvester_ssh_key" "mysshkey" {
  name      = var.keyname
  namespace = var.namespace
}

# --- Common Resources ---
resource "random_id" "secret" {
  byte_length = 5
}

resource "harvester_cloudinit_secret" "cloud-config" {
  name      = "cloud-config-${random_id.secret.hex}"
  namespace = var.namespace

  user_data = templatefile("cloud-init.tmpl.yml", {
    public_key_openssh = data.harvester_ssh_key.mysshkey.public_key
  })
}

# --- Host Virtual Machine (Matches Requirement Table: Host) ---
resource "harvester_virtualmachine" "hostvm" {
  count = 1 # Requirement: 1 Host VM

  name                 = "${var.username}-host-${random_id.secret.hex}"
  namespace            = var.namespace
  restart_after_update = true

  description = "Cluster Head Node"

  cpu    = 2    # Requirement: 2 Cores
  memory = "4Gi" # Requirement: 4GB RAM

  efi         = true
  secure_boot = false

  run_strategy    = "RerunOnFailure"
  hostname        = "${var.username}-host-${random_id.secret.hex}"
  reserved_memory = "100Mi"
  machine_type    = "q35"

  tags = {
    condenser_ingress_isAllowed           = true
    condenser_ingress_isEnabled           = true
    condenser_ingress_prometheus_hostname = "prometheus-${var.username}"
    condenser_ingress_prometheus_port     = 9090
    condenser_ingress_nodeexporter_hostname = "nodeexporter-${var.username}"
    condenser_ingress_nodeexporter_port   = 9100
    condenser_ingress_grafana_hostname    = "grafana-${var.username}"
    condenser_ingress_grafana_port        = 3000
    condenser_ingress_daskboard_hostname  = "daskboard-${var.username}"
    condenser_ingress_daskboard_port      = 8787
  }

  network_interface {
    name           = "nic-1"
    wait_for_lease = true
    type           = "bridge"
    network_name   = var.network_name
  }

  disk {
    name       = "rootdisk"
    type       = "disk"
    size       = "10Gi" # Requirement: 10GB HDD1
    bus        = "virtio"
    boot_order = 1

    image       = data.harvester_image.img.id
    auto_delete = true
  }
  # No HDD2 as per requirement

  cloudinit {
    user_data_secret_name = harvester_cloudinit_secret.cloud-config.name
  }
}

# --- Worker Virtual Machines (Matches Requirement Table: Worker) ---
resource "harvester_virtualmachine" "workervm" {
  count = 4 # Requirement: 4 Worker VMs

  name                 = "${var.username}-worker-${format("%02d", count.index + 1)}-${random_id.secret.hex}"
  namespace            = var.namespace
  restart_after_update = true

  description = "Cluster Compute Node ${count.index + 1}"

  cpu    = 4     # Requirement: 4 Cores
  memory = "32Gi" # Requirement: 32GB RAM

  efi         = true
  secure_boot = false

  run_strategy    = "RerunOnFailure"
  hostname        = "${var.username}-worker-${format("%02d", count.index + 1)}-${random_id.secret.hex}"
  reserved_memory = "100Mi"
  machine_type    = "q35"

  tags = {
    condenser_ingress_isAllowed     = true
    condenser_ingress_isEnabled     = true
    condenser_ingress_node_hostname = "node-${var.username}" # Consider making this unique per node if needed
    condenser_ingress_node_port     = 9100
  }

  network_interface {
    name           = "nic-1-${count.index}" # Ensure unique name per VM if needed, though Harvester might handle this
    wait_for_lease = true
    type           = "bridge"
    network_name   = var.network_name
  }

  disk {
    name       = "rootdisk"
    type       = "disk"
    size       = "50Gi" # Requirement: 50GB HDD1
    bus        = "virtio"
    boot_order = 1

    image       = data.harvester_image.img.id
    auto_delete = true
  }

  disk {
    name       = "datadisk" # Requirement: HDD2
    type       = "disk"
    size       = "200Gi" # Requirement: (up to 200GB) - Set to max specified
    bus        = "virtio"
    boot_order = 2      # Make sure root disk is boot_order = 1

    auto_delete = true # Usually desired for non-persistent data disks
  }

  cloudinit {
    user_data_secret_name = harvester_cloudinit_secret.cloud-config.name
  }
}
