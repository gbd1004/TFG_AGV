import psycopg2

CONNECTION = "postgres://postgres:password@localhost:5432/postgres"

with psycopg2.connect(CONNECTION) as conn:
    cursor = conn.cursor()
    # create sensor data hypertable
    query_create_sensordata_table = """CREATE TABLE sensor_data (
                                            time TIMESTAMPTZ NOT NULL,
                                            sensor_id INTEGER,
                                            temperature DOUBLE PRECISION,
                                            cpu DOUBLE PRECISION
                                        );
                                        """
    query_create_sensordata_hypertable = "SELECT create_hypertable('sensor_data', 'time');"

    cursor.execute(query_create_sensordata_table)
    cursor.execute(query_create_sensordata_hypertable)

    conn.commit()
    cursor.close()