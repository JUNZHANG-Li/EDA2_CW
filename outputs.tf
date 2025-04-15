output worker_vm_ips {
  value = harvester_virtualmachine.workervm[*].network_interface[0].ip_address
}

output worker_vm_ids {
  value = harvester_virtualmachine.workervm.*.id
}
output storage_vm_ips {
  value = harvester_virtualmachine.storagevm[*].network_interface[0].ip_address
}

output storage_vm_ids {
  value = harvester_virtualmachine.storagevm.*.id
}
output host_vm_ips {
  value = harvester_virtualmachine.hostvm[*].network_interface[0].ip_address
}

output host_vm_ids {
  value = harvester_virtualmachine.hostvm.*.id
}
