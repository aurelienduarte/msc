#!/bin/bash

# ./ports_tcp.py | sort -n | uniq > data/ports_tcp.txt
echo "out2host_port_count"
./out2host_port_count.py data/ip2.nmap.online.hosts.masscan > data/ip2.nmap.online.hosts.masscan.csv

echo "out2csv"
./out2csv.py data/ip2.nmap.online.hosts.masscan > data/ip2.nmap.online.hosts.masscan.csv

echo "out2tsv"
./out2tsv.py data/ip2.nmap.online.hosts.masscan > data/ip2.nmap.online.hosts.masscan.tsv

echo "csv_optimise"
./csv_optimise.py data/ip2.nmap.online.hosts.masscan.csv

# ./nmap.out2csv.py > data/random_ip.nmap.csv
# ./nmap.out2csv.py -open_only > data/random_ip.nmap.open_only.csv
# ./out2cartesian_csv.py data/random_ip.out > data/random_ip.our.cartesian.csv

echo "out2ports"
./out2ports.py data/ip2.nmap.online.hosts.masscan > data/ip2.nmap.online.hosts.masscan.ports

echo "out2hosts"
./out2hosts.py data/ip2.nmap.online.hosts.masscan > data/ip2.nmap.online.hosts.masscan.hosts

echo "out2port_count"
./out2port_count.py data/ip2.nmap.online.hosts.masscan > data/ip2.nmap.online.hosts.masscan.port_count
