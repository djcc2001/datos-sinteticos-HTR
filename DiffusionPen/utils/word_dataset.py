import io,os
import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from utils.auxilary_functions import image_resize_PIL, centered_PIL
import tqdm
from torchvision.utils import save_image
import json
import random
import pickle
import time
#import sys
#import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

MAX_CHARS = 64
OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

class WordLineDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)   
    '''
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    '''
    def __init__(self, 
        basefolder: str = 'datasets/',                #Root folder
        subset: str = 'all',                          #Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = 'line',             #Type of data to load ('line' or 'word')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        tokenizer = None,
        text_encoder = None,
        feat_extractor = None,
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        args = None,                                  #Optional args container for downstream datasets
                                #Feature extractor to be used for text encoding
        ):
        
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None                             # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.max_transcr_len = 0
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.output_max_len = OUTPUT_MAX_LEN
        self.feat_extractor = feat_extractor
        self.args = args
        self._indices_by_writer = None
        self._long_indices_by_writer = None

    CACHE_VERSION = 2

    @staticmethod
    def _normalize_image(img: Image.Image, target_h: int = 64, target_w: int = 256) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.height != target_h:
            img = image_resize_PIL(img, height=target_h)
        if img.width > target_w:
            img = image_resize_PIL(img, width=target_w)
        img = centered_PIL(img, (target_h, target_w), border_value=255.0)
        return img

    def _load_item_image(self, item0):
        if isinstance(item0, Image.Image):
            img = item0
        elif isinstance(item0, str):
            img = Image.open(item0)
        else:
            raise TypeError(f"Unsupported image type in dataset: {type(item0)}")
        return self._normalize_image(img)

    def _build_writer_indices(self):
        indices_by_writer = {}
        long_indices_by_writer = {}
        for idx, (_, transcr, wid, _) in enumerate(self.data):
            indices_by_writer.setdefault(wid, []).append(idx)
            if isinstance(transcr, str) and len(transcr) > 3:
                long_indices_by_writer.setdefault(wid, []).append(idx)
        self._indices_by_writer = indices_by_writer
        self._long_indices_by_writer = long_indices_by_writer
    def __finalize__(self):
        '''
        Will call code after descendant class has specified 'key' variables
        and ran dataset-specific code
        '''
        assert(self.setname is not None)
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(','))
            self.stopwords = self.stopwords[0]
        
        save_path = f'./saved_{self.setname.lower()}_data'
        
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        save_file = '{}/{}_{}_{}_v{}.pt'.format(
            save_path, self.subset, self.segmentation_level, self.setname, self.CACHE_VERSION
        )
        
        def _atomic_torch_save(obj, path: str):
            tmp = f"{path}.tmp"
            torch.save(obj, tmp)
            os.replace(tmp, path)

        if isfile(save_file) is False:
            data = self.main_loader(self.subset, self.segmentation_level)
            _atomic_torch_save(data, save_file)
        else:
            try:
                data = torch.load(save_file, map_location="cpu")
            except (EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as e:
                # Cache file is corrupted or partially written (common in Colab when interrupted).
                ts = int(time.time())
                backup = f"{save_file}.corrupt.{ts}"
                try:
                    os.replace(save_file, backup)
                except OSError:
                    try:
                        os.remove(save_file)
                    except OSError:
                        pass
                print(f"Warning: failed to load cached dataset '{save_file}' ({type(e).__name__}); rebuilding. Backup: {backup}")
                data = self.main_loader(self.subset, self.segmentation_level)
                _atomic_torch_save(data, save_file)
        
        #data = self.main_loader(self.subset, self.segmentation_level)
        self.data = data
        #print('data', self.data)
        self.initial_writer_ids = [d[2] for d in data]
        
        writer_ids,_  = np.unique([d[2] for d in data], return_inverse=True)
       
        self.writer_ids = writer_ids
        
        self.wclasses = len(writer_ids)
        print('Number of writers', self.wclasses)
        if self.character_classes is None:
            res = set()
             #compute character classes given input transcriptions
            for _,transcr,_,_ in tqdm.tqdm(data):
                #print('legth transcr = ', len(transcr))
                res.update(list(transcr))
                self.max_transcr_len = max(self.max_transcr_len, len(transcr))
                #print('self.max_transcr_len', self.max_transcr_len)
                
            res = sorted(list(res))
            res.append(' ')
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            print('Max transcription length: {}'.format(self.max_transcr_len))
            self.character_classes = res
            self.max_transcr_len = self.max_transcr_len
        #END FINALIZE
        self._build_writer_indices()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def draw_word(word: str) -> Image:
        # Define the target image width and height
        target_width = 256
        target_height = 64

        # Calculate the appropriate font size based on the target width and word length
        max_font_size = 45
        text_width, text_height = float('inf'), float('inf')
        font_size = max_font_size
        while text_width > target_width or text_height > target_height:
            font_size -= 1
            font = ImageFont.truetype('./Roboto-Regular.ttf', font_size)
            _,_,text_width, text_height = font.getbbox(word)
            
        # Create a white image with the target dimensions
        img = Image.new('RGB', (target_width, target_height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # Calculate the position to center the text
        position = ((target_width - text_width) / 2, (target_height - text_height) / 2)

        # Draw the text onto the image
        d.text(position, word, font=font, fill=0)

        return img

    @staticmethod
    def find_text_bounding_box(image):
            # Load the image
        #image = cv2.imread(image_path)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Threshold the image to separate black text from the background
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("Number of contours detected:", len(contours))
        cnts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(cnts)
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
        #cv2.imwrite('./new.png', image)
        
        return (x, y, w, h)
    
    @staticmethod
    def draw_word_in_bounding_box(word: str, bounding_box: tuple) -> Image:
        # bounding_box is a tuple (x1, y1, x2, y2) specifying the top-left (x1, y1) and
        # bottom-right (x2, y2) coordinates of the bounding box

        # Create a white image with the target dimensions (64x256)
        target_width = 256
        target_height = 64
        img = Image.new('RGB', (target_width, target_height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # Calculate the width and height of the bounding box
        box_width = bounding_box[2] - bounding_box[0]
        box_height = bounding_box[3] - bounding_box[1]

        # Calculate the appropriate font size based on the bounding box dimensions and word length
        max_font_size = 50
        font_size = max_font_size

        while True:
            # Load the font
            font = ImageFont.truetype('./Roboto-Regular.ttf', font_size)

            # Get the size of the text with the current font
            text_width, text_height = d.textsize(word, font=font)

            # Check if the text fits within the bounding box
            if text_width <= box_width and text_height <= box_height:
                break  # The text fits, exit the loop
            else:
                font_size -= 1  # Reduce font size and try again

        # Calculate the position to center the text within the bounding box
        x = bounding_box[0] + (box_width - text_width) / 2
        y = bounding_box[1] + (box_height - text_height) / 2
        position = (x, y)

        # Draw the text onto the image
        d.text(position, word, font=font, fill=0)

        return img
    
    def __getitem__(self, index):
        # Robust image loading (lazy) with a few retries on IO/corruption.
        tries = 0
        while True:
            item0, transcr, wid, img_path = self.data[index]
            try:
                img_pil = self._load_item_image(item0)
                break
            except Exception:
                tries += 1
                if tries >= 3:
                    raise
                index = random.randint(0, len(self.data) - 1)

        img = self.transforms(img_pil) if self.transforms is not None else img_pil

        # Fast style sampling: precomputed indices per writer.
        if self._indices_by_writer is None:
            self._build_writer_indices()

        long_ids = self._long_indices_by_writer.get(wid, [])
        all_ids = self._indices_by_writer.get(wid, [])
        source_ids = long_ids if len(long_ids) > 0 else all_ids

        num_style_samples = int(getattr(self.args, "num_style_samples", 5) or 5)
        if len(source_ids) >= num_style_samples:
            style_ids = random.sample(source_ids, k=num_style_samples)
        else:
            style_ids = (
                random.choices(source_ids, k=num_style_samples)
                if len(source_ids) > 0
                else [index] * num_style_samples
            )

        st_imgs = []
        for sid in style_ids:
            s_item0 = self.data[sid][0]
            try:
                s_pil = self._load_item_image(s_item0)
                st_imgs.append(self.transforms(s_pil) if self.transforms is not None else s_pil)
            except Exception:
                # fallback: reuse main image
                st_imgs.append(img)
        s_imgs = torch.stack(st_imgs) if torch.is_tensor(st_imgs[0]) else st_imgs

        # A correlated image from same writer (used by some sampling modes; keep for compatibility).
        cor_idx = random.choice(source_ids) if len(source_ids) > 0 else index
        try:
            cor_pil = self._load_item_image(self.data[cor_idx][0])
            cor_im = self.transforms(cor_pil) if self.transforms is not None else cor_pil
        except Exception:
            cor_im = img
        
        '''
        pos_image = random.sample(positive_samples, k=1)
        neg_image = random.sample(negative_samples, k=1)
        
        pos_image = pos_image[0][0]
        neg_image = neg_image[0][0]
        pos_image = self.transforms(pos_image)
        neg_image = self.transforms(neg_image)
        '''
        return img, transcr, wid, s_imgs, img_path, cor_im
    
    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        img, transcr, wid, s_imgs, img_path, cor_im = zip(*batch)

        #context = [item.detach() for item in transcr]  # Detach context tensors
        transcr = torch.stack(transcr)
        #context = tok_transcr#torch.stack(tok_transcr)
        
        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        
        s_imgs = torch.stack(s_imgs)
        
        cor_images_batch = torch.stack(cor_im)
        # pos_images_batch = torch.stack(pos_image)
        # neg_images_batch = torch.stack(neg_image)
        
        return images_batch, transcr, wid, s_imgs, img_path, cor_images_batch#, pos_images_batch, neg_images_batch#, printed_word, bbox#, context

    
    
    def main_loader(self, subset, segmentation_level) -> list:
        # This function should be implemented by an inheriting class.
        raise NotImplementedError

    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        '''
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not
        
        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        '''
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError('The requested fixed image size is invalid!')
            new_img = resize(image=img, output_shape=fixed_image_size[::-1], mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print('OUCH')
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img
    
    def print_random_sample(self, image, transcription, id, as_saved_files=True):
        import random    #   Create method that will show example images using graphics-in-console (e.g. TerminalImageViewer)
        from PIL import Image
        # Run this with a very low probability
        x = random.randint(0, 10000)
        if(x > 5):
            return
        def show_image(img):
            def get_ansi_color_code(r, g, b):
                if r == g and g == b:
                    if r < 8:
                        return 16
                    if r > 248:
                        return 231
                    return round(((r - 8) / 247) * 24) + 232
                return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
            def get_color(r, g, b):
                return "\x1b[48;5;{}m \x1b[0m".format(int(get_ansi_color_code(r,g,b)))
            h = 12
            w = int((img.width / img.height) * h)
            img = img.resize((w,h))
            img_arr = np.asarray(img)
            h,w  = img_arr.shape #,c
            for x in range(h):
                for y in range(w):
                    pix = img_arr[x][y]
                    print(get_color(pix, pix, pix), sep='', end='')
                    #print(get_color(pix[0], pix[1], pix[2]), sep='', end='')
                print()
        if(as_saved_files):
            Image.fromarray(np.uint8(image*255.)).save('/tmp/a{}_{}.png'.format(id, transcription))
        else:
            print('Id = {}, Transcription = "{}"'.format(id, transcription))
            show_image(Image.fromarray(255.0*image))
            print()

class LineListIO(object):
    '''
    Helper class for reading/writing text files into lists.
    The elements of the list are the lines in the text file.
    '''
    @staticmethod
    def read_list(filepath, encoding='utf-8'):        
        if not os.path.exists(filepath):
            raise ValueError('File for reading list does NOT exist: ' + filepath)
        
        linelist = []        
        if encoding == 'ascii':
            transform = lambda line: line.encode()
        else:
            transform = lambda line: line 

        with io.open(filepath, encoding=encoding) as stream:            
            for line in stream:
                line = transform(line.strip())
                if line != '':
                    linelist.append(line)                    
        return linelist

    @staticmethod
    def write_list(file_path, line_list, encoding='utf-8', 
                   append=False, verbose=False):
        '''
        Writes a list into the given file object
        
        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''                
        mode = 'w'
        if append:
            mode = 'a'
        
        with io.open(file_path, mode, encoding=encoding) as f:
            if verbose:
                line_list = tqdm.tqdm(line_list)
              
            for l in line_list:
                #f.write(unicode(l) + '\n')   Python 2
                f.write(l + '\n')

