
'''
from PIL import Image
import glob
import os
from google_images_download import google_images_download
'''


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


def load_priors(root_dir):
    # image
    prior = {}
    for subdir, dirs, files in os.walk(rootdir):
        images[subdir] = []
        for filename in glob.glob(subdir+'/*.jpg'):  # assuming gif
            img = Image.open(filename)
            images[subdir].append(img)

    return images



