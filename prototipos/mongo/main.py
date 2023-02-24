from pymongo import MongoClient

import csv

def insert(collection_name):
    with open('prototipos/datos/vix-daily_csv.csv') as csv_file:
        items = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                items.append({
                    "_id": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4])
                })
            line_count += 1
        collection_name.insert_many(items)

def query(collection_name):
    # item_details = collection_name.find().sort({'open':-1}).limit(1)
    max_open = collection_name.find_one(sort=[("open", -1)])["open"]
    max_high = collection_name.find_one(sort=[("high", -1)])["high"]
    max_low = collection_name.find_one(sort=[("low", -1)])["low"]
    max_close = collection_name.find_one(sort=[("close", -1)])["close"]

    print("max open = " + str(max_open))
    print("max high = " + str(max_high))
    print("max low = " + str(max_low))
    print("max close = " + str(max_close))

def get_database():
    client = MongoClient("mongodb://localhost:27017")

    return client["mydatabase"]

if __name__ == "__main__":
    dbname = get_database()

    collection_name = dbname["financial-analysis"]

    print("Insertando datos")
    # try:    
    insert(collection_name)
    # except:
    #     print("Datos ya insertados")
    
    print("Datos insertados")

    query(collection_name)