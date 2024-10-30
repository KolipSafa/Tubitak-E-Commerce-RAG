class Product:
    def to_dict(self):
        raise NotImplementedError
    
    def features_to_md_text(self):
        """
        Converts the product's attributes into a Markdown-formatted string.

        This method is intended to take the product's details (such as name, price, etc.) 
        and format them in a structured Markdown text. The exact attributes used for the conversion 
        will depend on the implementation of the product class.

        Returns:
            str: A Markdown-formatted string containing the product's details.
        
        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError

    def review_to_md_text(review):
        """
        Converts a review's content into a Markdown-formatted string.

        This method takes the review's details (such as rating, comments, author, etc.)
        and formats them into structured Markdown text. The exact attributes used for 
        the conversion will depend on the implementation of the review class.

        Returns:
            str: A Markdown-formatted string containing the review's details.
        
        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError
    
    def review_to_json(review):
        raise NotImplementedError

