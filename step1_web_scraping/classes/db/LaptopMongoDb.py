import os
from bson import ObjectId
from mongoengine import Document, StringField, connect, DoesNotExist, ValidationError, IntField
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv('LAPTOP_MONGO_URI')
mongo_db_name = os.getenv('LAPTOP_MONGO_DB_NAME')

# Laptop Entity
class LaptopEntity(Document):
    name = StringField(required=True)
    url = StringField(required=True, unique=True)
    processor_brand = StringField()
    processor_name = StringField()
    graphic_processor = StringField()
    ram_capacity = StringField()
    storage_type = StringField()
    storage_capacity = StringField()
    screen_size = StringField()

class LaptopMongoDb:
    def __init__(self):
        try:
            connect(host=mongo_uri + mongo_db_name)
        except Exception as e:
            print(f"Error connecting to the database: {e}")

    # CREATE - Add new laptop
    def create(self, name, url):
        try:
            if not LaptopEntity.objects(url=url).first():
                laptop = LaptopEntity(name=name, url=url)
                laptop.save()
                return {"status": "success", "data": str(laptop.id)}
            else:
                return {"status": "error", "message": f"A laptop with this URL already exists: {url}"}
        except ValidationError as ve:
            return {"status": "error", "message": f"Validation error: {ve}"}
        except Exception as e:
            return {"status": "error", "message": f"Error while adding laptop: {e}"}

    # GET ALL - Retrieve all laptops
    def get_all_laptops(self):
        try:
            laptops = LaptopEntity.objects()
            return {"status": "success", "data": laptops}
        except Exception as e:
            return {"status": "error", "message": f"Error retrieving laptops: {e}"}

    # GET BY URL - Retrieve laptop by URL
    def get_laptop_by_url(self, url):
        try:
            laptop = LaptopEntity.objects(url=url).first()
            if laptop:
                return {"status": "success", "data": laptop}
            else:
                return {"status": "error", "message": "Laptop not found."}
        except Exception as e:
            return {"status": "error", "message": f"Error retrieving laptop: {e}"}

    def update(self, laptop_id, update_data):
        try:
            laptop = LaptopEntity.objects(id=ObjectId(laptop_id)).first()
            if laptop:
                laptop.update(**update_data)
                laptop.reload()
                return {"status": "success", "message": "Laptop updated", "data": laptop}
            else:
                return {"status": "error", "message": "Laptop not found."}
        except ValidationError as ve:
            return {"status": "error", "message": f"Validation error: {ve}"}
        except Exception as e:
            return {"status": "error", "message": f"Error while updating laptop: {e}"}

    # DELETE - Delete laptop
    def delete(self, laptop_id):
        try:
            laptop = LaptopEntity.objects(id=ObjectId(laptop_id)).first()
            if laptop:
                laptop.delete()
                return {"status": "success", "message": f"Laptop deleted: {laptop_id}"}
            else:
                return {"status": "error", "message": "Laptop not found."}
        except Exception as e:
            return {"status": "error", "message": f"Error while deleting laptop: {e}"}