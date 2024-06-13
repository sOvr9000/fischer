
from fischer.factoriobalancers import BeltGraph
from construct_m_n_alt import construct_m_n_alt
import json
from clipboard import paste





def analyze_string() -> BeltGraph:

    # big_balancers = json.load(open('examples/construct_m_n_alt_output.json', 'r'))

    graph = BeltGraph()

    # graph.load_compact_string(big_balancers['3-12'])
    # print(graph)

    # graph.load_common_balancer('3-3')
    # print(graph)

    # graph.load_blueprint_string('0eNqlXNtu20oM/Bc9O4V2uVf/ShEcJK1QCEhkw5aLBoH//di59OjEHoviPAUJ4iGXS3J2xbFem8enQ7fd9cPYrF+b/sdm2Dfr76/Nvv81PDyd/za+bLtm3fRj99ysmuHh+fxb92e76/b7u3H3MOy3m91499g9jc1x1fTDz+5Ps3bHlRpkv33qx7HbTT7uj/erphvGfuy7d4fefnn5Zzg8P57+c+3mXFk1283+9PHNcLZ/grzLVb7FVfPSrLNvv8Xj2cMvqN6C6udQBa/4qpcfaO46WliG5m6jRcuK3dyKkwW1nUPNBtRS51DLoniWejue1eJjmfPRtRewh1Oh7H7tNqefGDj/BT67u/osw81h3B7O1XppyC2LRrodDWcpqInXKBxigU2zsMECG2dhTTUW/sIKgE0crH/LiJ/9rvvx/i/+mpHM9VlRGSlc29UZqdxKPGAMW2VOvPe6yvSO69Qgi7y3+d9+if68/8L5j+K/kA7bGbhoa7R1cTgSRxRoOzMHi8JSuD5+UaHpmpHKdXWVETEyaVpar+I4MtGtxsSvs9QiwvkOskgC561XhSQuawjhAzwAn0mWjQA2s7QRpmnYDygLSWpF7leb+/I/YEUVhZZjjahJmUBSa/hqJFwz4rk7UVCtRGyXGbDNIXA3Lp3PkaOsoDn9hcQRWFQZyRyBoV0gyVeXnpXlRV1Dii3HAaBNR5JtEaztdDwBVva5KDZDYbGhwPFZAoGK3O0qaTpFTCxpXprRhCxzHJo19ReL7R6TwX5UjtKyZj9Sy94dk6pnJMcxRNY07+Q5rgOFkYyFXa7HCGdpChxLgERKkeWFy/RX7DhJ10m14yRdJ01dp8IySNbVSOV4FWx/bln3b7RcuJjsuMWo2CR7jgkriJhwsEWTujlwlFQ0qZsje8mrqtTNibVzGTOFVZLVqyqEtuqfEHHVdf9cbccHkMSltQ0ZC4BztmspgvPL4MrMYoWluIopDm9ZCexMr6jKq9AMfqO8biwvcTeyqumDJXNEgTKicL6jvCU5GsDW1vTEU8C4sjqKwAQNxjltxZu3s/lQhTPiNAeHGqi7LgwQJ8VQ+s4pMy6NXN2FzK1Et9WFWwnK/kpdA9Hmural7rAY10atAoUmnGJDl4Wu5QQcWivclRjHKHLetzrv07Kdnc2UTJEajkYhH11+xGPuHOPayhETmEo6Z+RPpMtxjiMhr7lZOJO0asJCOvmPIwlVJ81xNk0j3lGSSuHOkuwJcbOtiUO8QqpYxOtK0lXy6abWkG+5Vq7LQ5PGatLY0X54b1JVwvz2YqMG6B8nzMB+Gp8g/YcclLJVz8k0JKp6rufUkBJQnDiVhqjm7M6zoo3LKGnGZ05YiaQoZ5vuiuZq2eM2vSVWNqm3JOyadIN5JzYlJUxqidxNKiLcxHKOOvTZ9JwRR6SQT/z0nrNqjluljlPIpNaaXDFUai0XuOkQTKzAij3UbBWEWwFKsMCJO0Q1NnWBZnXdSNOFZTfgCWFkzTJIMlcGq3BWss6KbfwjCX1jhZNdKr02ibYmFJqR956lzGQYKLooHOHB5QTu0aFyMyJLq8kyJnJx4WOuvKjII6fxEJV2y8XCWdG1ksiNj2B+JZK1URNJJE9DXE64IRXhiu05YEF4nEhDVCINlzjlJY5G4ryvqrNcyuw9SqcxcYlVYlxuh+rym8h5EtqeTM6TIK5xnoSKIJPzJF0aZXKepLTCKjZu5RBO3UyOm4pucYm9/SgrMXOaDZy5hcOFGcyNnQKam5WWatxBN0VcKL36fPlEQFPE4smHeUE57StimglhxwNJAGrHo+kuhvMkmToyDkQmD/rqQCx8s0aec7yS7Vfr+EJB1ee4Bu5gdaZxDQxE5b71i/0UriOhMVDlDsIYlxvzYlxuzItxuRfYYNxCnQUxbqXObgG+MqOlDjQYl/sSIMYl6+0N9371/var9eSNW6vmd7fbv9/4iwunnD99rEqq6Xj8Fz2Gvn0=')
    graph.load_blueprint_string(paste()) # load from current clipboard content
    print(graph)
    print(graph.evaluate())

    # graph.load_factorio_sat_network('examples/networks/13x11')
    # print(graph)
    # print(graph.evaluate())

    print('Removable vertices:')
    for u in graph.removable_vertices():
        print(u)

    print('Graph summary:')
    print(graph.advanced_summary)
    print('Graph is solved:')
    print(graph.is_solved())



def main():
    analyze_string()



if __name__ == '__main__':
    main()







