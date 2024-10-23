import sqlite3

class DBFetcher:
    def __init__(self, db_path):
        self.db_path = db_path

    def create_connection(self):
        return sqlite3.connect(self.db_path)

    def get_by_id(self, table_name, record_id, columns):
        with self.create_connection() as conn:
            c = conn.cursor()
            query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE id = ?"
            c.execute(query, (record_id,))
            result = c.fetchone()
            if result:
                return dict(zip(columns, result))
            return None
        
    def to_xml(self,data):
        xml = []
        for key, value in data.items():
            xml.append(f"<{key}>{value}</{key}>\n")
        return ''.join(xml)
