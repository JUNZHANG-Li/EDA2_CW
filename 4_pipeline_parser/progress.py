def task_num(file_path):
    try:
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0

def write_prometheus_metric(prometheus_file, result):
    try:
        with open(prometheus_file, 'w') as file:
            file.write(result)
    except Exception:
        return 0

done_list = '/home/almalinux/data/clean.log'
task_list = '/home/almalinux/paths.txt'
prometheus_file = '/home/almalinux/prom_metrics/progression_ratio.prom'

done = task_num(done_list)
total = task_num(task_list)
ratio = (done/total)*100 if total != 0 else 0
result = f"""
# HELP progression_ratio Ratio of files processed
# TYPE progression_ratio gauge
progression_ratio {ratio:.3f}

# HELP progression_number Number of files processed
# TYPE progression_number gauge
progression_number {done:d}
"""
write_prometheus_metric(prometheus_file, result)

