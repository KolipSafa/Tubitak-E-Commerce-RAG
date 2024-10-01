import sqlite3

class LaptopDatabase:
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
                    name TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE
                )
            ''')
            conn.commit()

    def add_laptop(self, name, url):
        if self.is_url_in_database(url):
            return None

        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO laptops (name, url)
                VALUES (?, ?)
            ''', (name, url))
            conn.commit()
            return c.lastrowid

    def get_all_laptops(self):
        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM laptops')
            return c.fetchall()
        
    def is_url_in_database(self, url):
        with self.create_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT 1 FROM laptops WHERE url = ?
            ''', (url,))
            result = c.fetchone()
            return result is not None