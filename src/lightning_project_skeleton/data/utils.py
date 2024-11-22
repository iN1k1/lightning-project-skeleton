def is_image_file(filename:str):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'])
