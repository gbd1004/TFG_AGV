import csv
import sqlite3

def insert(cur, con):
    with open('prototipos/datos/vix-daily_csv.csv') as csv_file:
        items = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                items.append((
                    row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])
                ))
            line_count += 1
        cur.executemany("INSERT INTO financial_analysis VALUES(?, ?, ?, ?, ?)", items)
        con.commit()

def query(cur):
    res = cur.execute("SELECT MAX(open) max_open, MAX(high) max_high, MAX(low) max_low, MAX(close) max_close FROM financial_analysis")
    
    for row in res:
        print("max open = " + str(row[0]))
        print("max high = " + str(row[1]))
        print("max low = " + str(row[2]))
        print("max close = " + str(row[3]))

if __name__ == "__main__":
    con = sqlite3.connect("prototipos/sqlite/test.db")

    cur = con.cursor()

    try:
        cur.execute("DROP TABLE financial_analysis")
        con.commit()
    except:
        print("La tabla no existe")

    cur.execute("CREATE TABLE financial_analysis(date, open, high, low, close)")

    try:
        insert(cur, con)
    except:
        print("Los datos ya se han introducido")

    query(cur)
