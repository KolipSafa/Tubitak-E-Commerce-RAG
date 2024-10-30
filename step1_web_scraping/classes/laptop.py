import json
import textwrap
from classes.product import Product

class Laptop(Product):
    def __init__(self, id=None,product_id=None, name=None, url=None, processor_name="Unknown", processor_brand="Unknown", 
                 graphic_processor="Unknown", ram_capacity="Unknown", storage_type="Unknown", 
                 storage_capacity="Unknown", screen_size="Unknown", reviews="Unknown"):
        self.id = str(id)
        self.product_id = product_id
        self.name = name
        self.url = url
        self.processor_brand = processor_brand
        self.processor_name = processor_name
        self.graphic_processor = graphic_processor
        self.ram_capacity = ram_capacity
        self.storage_type = storage_type
        self.storage_capacity = storage_capacity
        self.screen_size = screen_size
        self.reviews = reviews

    def to_dict(self):
        return self.__dict__
    
    def review_to_md_text(self,review):
        md_text = ''
        if 'title' in review and review['title']:
            md_text += f"### Review's Title: {review['title']}\n\n"
        if 'content' in review and review['content']:
            md_text += f"**Review's Content**:\n{review['content']}\n\n"
        if 'numberOfLikes' in review and review['numberOfLikes']:
            md_text += f"**Review's Likes**: {review['numberOfLikes']}\n"
        if 'numberOfDislikes' in review and review['numberOfDislikes']:
            md_text += f"**Review's Dislikes**: {review['numberOfDislikes']}\n"

        return textwrap.dedent(md_text)
    
    def review_to_json(self, review):
        json_data = {}
        if 'title' in review and review['title']:
            json_data['title'] = review['title']
        if 'content' in review and review['content']:
            json_data['content'] = review['content']
        if 'numberOfLikes' in review and review['numberOfLikes']:
            json_data['numberOfLikes'] = review['numberOfLikes']
        if 'numberOfDislikes' in review and review['numberOfDislikes']:
            json_data['numberOfDislikes'] = review['numberOfDislikes']

        return json.dumps(json_data, indent=4)



     