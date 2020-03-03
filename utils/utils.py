import os
from PIL import Image

def loadImages(path):
   loaded_images = [Image.open(os.path.join(path, image)) for image in os.listdir(path)]

   return loaded_images

def loadAndResizeImages(top_path, output_directory, output_size=(256,256)):
   '''
   Assumes that images are of the size 1024x1024. Loads one image at a time,
   rescales it, and saves it back to the output folder.

   'top_path' is the folder containing folders with images (assumes that
   images are spread out among many folders).
   '''

   all_image_locations = []
   for folder_name in os.listdir(top_path):
      print(folder_name)
      if os.path.isfile(os.path.join(top_path, folder_name)):
         continue
      file_names = os.listdir(os.path.join(top_path, folder_name))
      file_locations = [os.path.join(top_path, folder_name, f) for f in file_names]
      all_image_locations += file_locations

   all_image_locations = list(set(all_image_locations))

   for image_loc in all_image_locations:
      image_name = image_loc.split(os.sep)[-1]
      if os.path.exists(os.path.join(output_directory, image_name)):
         continue

      #print("image loc:", image_loc)
      try:
         image = Image.open(image_loc)
         image.thumbnail(output_size, Image.ANTIALIAS)
         image.save(os.path.join(output_directory, image_name))
      except Exception as ex:
         print(ex)


if __name__ == "__main__":
   loadAndResizeImages("/home/jg/Documents/stylegan/ffhq-dataset/images1024x1024", "/home/jg/Documents/stylegan/ffhq-dataset/images256x256")