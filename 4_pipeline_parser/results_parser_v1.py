import sys
import csv
import json
from collections import defaultdict
import statistics
# fhOut = open(sys.argv[1]."summmary", "w")
# fhOut.write('ID,domain_count\n')

cath_ids = defaultdict(int)
plDDT_values = []
id = sys.argv[1].split("_search.tsv")[0]
with open(sys.argv[1], "r") as fhIn:
    next(fhIn)
    msreader = csv.reader(fhIn, delimiter='\t',) 
    tot_entries = 0
    for i, row in enumerate(msreader):
        tot_entries = i+1
        plDDT_values.append(float(row[3]))
        meta = row[15]
        data = json.loads(meta)
        cath_ids[data["cath"]] += 1

    with open(id+".parsed", "w", encoding="utf-8") as fhOut:
        if len(plDDT_values) > 0:
            fhOut.write(f"#{sys.argv[1]} Results. mean plddt: {statistics.mean(plDDT_values)}\n")
        else:
             fhOut.write(f"#{sys.argv[1]} Results. mean plddt: 0\n")
        fhOut.write("cath_id,count\n")
        for cath, v in cath_ids.items():
            fhOut.write(f"{cath},{v}\n")

