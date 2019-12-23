import pyfpgrowth
import pickle
import sys
sys.setrecursionlimit(100000)

transactions = []

with open ('data/ip2.nmap.online.hosts.masscan.transactions', 'rb') as fp:
    transactions = pickle.load(fp)

patterns = pyfpgrowth.find_frequent_patterns(transactions, 1)

rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
