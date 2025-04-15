## IP Address
`10.134.12.113`

# Deployment Steps

### 0. **Generate Key Pairs**
    - Upload to Rancher.
    - Edit `variables.tf` to use your own SSH keys.

### 1. **Configure cnc machine**

    cd cnc-provisioning
    ansible-playbook -i local-inventory.ini provision.yaml
    
### 2. **(Optional) Update Packages on All Machines**

    ansible-playbook -i generate_inventory.py update-almalinux.yaml


### 3. **Setup Monitoring Systems**  
   (**Prometheus**, **Grafana**)

    ansible-playbook -i generate_inventory.py setup-monitoring.yaml

Once the monitoring systems are set up, you can access:
- **Grafana** at [https://grafana-ucabiqx-x.comp0235.condenser.arc.ucl.ac.uk/](https://grafana-ucabiqx-x.comp0235.condenser.arc.ucl.ac.uk/)
  - **Default username:** `admin`
  - **Default password:** `william200212`

- **Prometheus** at [https://prometheus-ucabiqx-x.comp0235.condenser.arc.ucl.ac.uk/graph](https://prometheus-ucabiqx-x.comp0235.condenser.arc.ucl.ac.uk/graph)

### 4. **Configure All Nodes**

    ansible-playbook -i generate_inventory.py config-nodes.yaml

### 5. **Setup Minio**

    ansible-playbook -i generate_inventory.py setup-minio.yaml


### 6. **Download Datasets**  
   (**Human**, **Ecoli**, **Cath**)

    ansible-playbook -i generate_inventory.py setup-datasets.yaml


### 7. **Setup Pipeline and Parser**

    ansible-playbook -i generate_inventory.py setup-pipeline-parser.yaml


### 8. **Merizo Analysis**

    python analyser.py


---

### 9. **Install Hadoop**

    ansible-playbook -i generate_inventory.py setup-hadoop.yaml


Unfortunately, due to the failure in regenerate pyspark summarisation, only results are provided in the repository, in **Summary & plddt Generator** and **pyspark-files**
