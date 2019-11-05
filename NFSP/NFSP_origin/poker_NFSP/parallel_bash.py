import subprocess

template = 'python NFSP_test.py --seed={}'

args = [i for i in range(50)]

processes = []

for arg in args:
    print("arg",arg)
    command = template.format(arg)
    print(command)
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

output = [p.wait() for p in processes]
