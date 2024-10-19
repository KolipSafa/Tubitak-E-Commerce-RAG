import sqlite3

class LaptopSQLiteDb:
    def __init__(self, db_path):
        self.db_path = db_path
        self.create_table()

    def create_connection(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS laptops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    processor_brand TEXT,
                    processor_name TEXT,
                    graphic_processor TEXT,
                    ram_capacity TEXT,
                    storage_type TEXT,
                    storage_capacity TEXT,
                    screen_size TEXT
                )
            ''')
            conn.commit()

    # Laptop sınıfının örneğini veritabanına ekleyen fonksiyon
    def add_laptop(self, laptop):
        if self.is_product_id_in_database(laptop.product_id):
            return None

        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO laptops (product_id, name, url, processor_brand, processor_name, 
                                     graphic_processor, ram_capacity, storage_type, 
                                     storage_capacity, screen_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (laptop.product_id, laptop.name, laptop.url, laptop.processor_brand, 
                  laptop.processor_name, laptop.graphic_processor, laptop.ram_capacity, 
                  laptop.storage_type, laptop.storage_capacity, laptop.screen_size))
            conn.commit()
            return c.lastrowid

    def get_all_laptops(self):
        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM laptops')
            return c.fetchall()
        
    def is_product_id_in_database(self, product_id):
        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT 1 FROM laptops WHERE product_id = ?
            ''', (product_id,))
            result = c.fetchone()
            return result is not None
