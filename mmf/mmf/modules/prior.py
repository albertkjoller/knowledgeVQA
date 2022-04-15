
import torch
import pickle
from PIL import Image
import glob
from pathlib import Path
import numpy as np
from mmf.utils.model_utils.image import openImage
from mmf.utils.build import build_processors
from mmf.common.sample import Sample, SampleList
from mmf.utils.text import VocabDict




# taken from: <https://developers.google.com/drive/api/guides/manage-downloads#python>
def download_from_drive(save_dir):
    file_id = '0BwwA4oUTeiV1UVNwOHItT0xfa2M'
    # from google drive folder /pretrained
    file_id = '1fkm4c2-yI0XFIhtKK5Zrj7OgidnePzWr'
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))


def load_priors(cache_dir, data_dir, processors_config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        #file = open(Path(f'{cache_dir}/priors.pkl'), "rb")
        #priors = pickle.load(file)
        file = open(Path(f'{cache_dir}/priors.pt'), "rb")
        priors = torch.load(file)

        return priors

    except (OSError, IOError) as e:

        # building processors
        extra_params = {"data_dir": data_dir}
        processors = build_processors(processors_config, **extra_params)
        text_processor = processors['text_processor']
        image_processor = processors['image_processor']

        # extracting answer vocabulary file for current experiement
        vocab_path = processors_config.answer_processor.params.vocab_file
        #with open(Path(f'{data_dir}/{vocab_path}')) as f:
        #    answer_vocab = f.read().splitlines()
        answer_vocab = VocabDict(Path(f'{data_dir}/{vocab_path}'))

        # defining random image for candidate answers without image priors
        img_array = np.random.rand(224, 224, 3) * 255
        random_img = Image.fromarray(img_array.astype('uint8')).convert('RGB')


        # initializing dict based on answer vocabulary and witrh random priors
        #priors = dict.fromkeys(answer_vocab, img)

        priors = dict.fromkeys(list(answer_vocab.word2idx_dict.keys()), dict())

        # counting avaliable image priors
        num_priors = 0
        # container for number of images per prior
        num_images = []

        # looping each folder in priors, i.e. answer candidates
        folders = Path(f'{cache_dir}/priors').glob('*')
        for ans_cand in answer_vocab.word2idx_dict:
            # if path to current answer exists
            if Path(f'{cache_dir}/priors/{ans_cand}').is_dir():

        #for ans_cand_path in folders:
            # getting folder name, i.e. answer candidate
            #ans_cand = ans_cand_path.name
            #images = Path(prior_path).glob(folder+'/*.jpg')
            #img_paths = Path(f'prior_path/{ans_cand}').glob('*.jpg')
            #print('img paths:', img_paths)

            #if ans_cand in priors.keys():
                # process answer candidate as text input for prior
                priors[ans_cand]['input_ids'] = text_processor({'text': ans_cand})['input_ids']
                # initializing
                priors[ans_cand]['images'] = []
                # counting number of folders
                num_priors += 1
                # counting number of images per prior
                j = 0
                for img_path in Path(f'{cache_dir}/priors/{ans_cand}').glob('*.jpg'):
                    #img = Image.open(img_path)
                    #img = openImage(img_path)
                    # process image input
                    processed_image = image_processor({'image': openImage(str(img_path))})
                    processed_image = processed_image['image']
                    #processed_image = Sample(processed_image['image'])
                    #processed_image = Sample(processed_image)
                    # extending
                    #priors[ans_cand]['images'] =  torch.cat([priors[ans_cand]['images'], processed_image]) #.unsqueeze(0)])
                    # saving to dictionary
                    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

                    #processed_image.to(device)
                    priors[ans_cand]['images'].append(processed_image) #.unsqueeze(0)])
                    j += 1

            #priors[ans_cand]['images'] = torch.tensor(priors[ans_cand]['images'])
                #priors[ans_cand]['images'] = torch.cat(priors[ans_cand]['images'])

                num_images.append(j)


            # not folder for current answer and picture is random as initialized
            else:
                # process answer candidate as text input for prior
                priors[ans_cand]['input_ids'] = text_processor({'text': ans_cand})['input_ids']

                processed_image = image_processor({'image': random_img})
                processed_image = processed_image['image']
                priors[ans_cand]['images'] = processed_image


        print('\n')
        print('number of avaliable priors: {} out of {} candidate answers'.format(num_priors, len(priors)))
        print('average number of images per avaliable prior: ', np.mean(num_images))


        '''
        for subdir, dirs, files in os.walk(prior_path):
            priors[subdir] = []
            # loading each image prior
            for filename in glob.glob(subdir+'/*.jpg'):  # assuming jpg
                img = Image.open(filename)
                # saving to dictionary
                priors[subdir].append(img)
        '''

        #file = open(Path(f'{cache_dir}/priors.pkl'), "wb")
        #pickle.dump(priors, file)
        #file.close()
        file = Path(f'{cache_dir}/priors.pt')
        torch.save(priors, file)


        return priors

    raise NotImplementedError

