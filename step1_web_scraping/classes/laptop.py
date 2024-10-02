import textwrap
from classes.product import Product

class Laptop(Product):
    def __init__(self, id=None, name=None, url=None, processor_name=None, processor_brand=None, 
                 ram_capacity=None, storage_type=None, storage_capacity=None, screen_size=None, reviews=None):
        self.id = str(id)
        self.name = name
        self.url = url
        self.processor_brand = processor_brand
        self.processor_name = processor_name
        self.ram_capacity = ram_capacity
        self.storage_type = storage_type
        self.storage_capacity = storage_capacity
        self.screen_size = screen_size
        self.reviews = reviews

    def to_dict(self):
        return self.__dict__
    
    def features_to_md_text(self):
        return textwrap.dedent(f"""
        ---
        laptop_id: {self.id}
        ---

        **Laptop ID**: {self.id} 
        **Laptop Name**: {self.name} 
        **Processor Brand**: {self.processor_brand}  
        **Processor Name**: {self.processor_name}  
        **RAM**: {self.ram_capacity}  
        **Storage Type**: {self.storage_type}  
        **Storage Capacity**: {self.storage_capacity}
        **Screen Size**: {self.screen_size}  
        """).strip()
    
    def review_to_md_text(self,review):
        md_text = ''

        metadata = '---\n'
        metadata += f"laptop_id: {self.id}\n"
        metadata += '---\n\n'
        md_text += metadata

        if 'title' in review and review['title']:
            md_text += f"### Review's Title: {review['title']}\n\n"
        if 'rating' in review and review['rating']:
            md_text += f"**Review's Rating of the Laptop**: {review['rating']}\n\n"
        # if 'writtenBy' in review and review['writtenBy']:
        #     md_text += f"**Written By**: {review['writtenBy']}\n\n"
        if 'content' in review and review['content']:
            md_text += f"**Review's Content**:\n{review['content']}\n\n"
        if 'numberOfLikes' in review and review['numberOfLikes']:
            md_text += f"**Review's Likes**: {review['numberOfLikes']}\n"
        if 'numberOfDislikes' in review and review['numberOfDislikes']:
            md_text += f"**Review's Dislikes**: {review['numberOfDislikes']}\n"

        return textwrap.dedent(md_text)



     