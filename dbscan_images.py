import itertools
import multiprocessing
import operator
import urllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from logging import DEBUG, getLogger

import numpy as np
import pandas as pd
from sklearn import cluster

import cv2
import holoviews as hv
from wand.display import display
from wand.image import Image

hv.extension('bokeh')
logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.propagate = False


class DbscanImages(object):

  __instance = None

  def __new__(cls, *args, **keys):
    if cls.__instance is None:
      cls.__instance = object.__new__(cls)
    return cls.__instance

  def __init__(self):
    logger.debug("init")

  def compare(self, args):
    img, img2 = args
    img = ((img - img.mean()) / img.std()) if img.std() != 0 else 1
    img2 = ((img2 - img2.mean()) / img2.std()) if img2.std() != 0 else 1
    return np.mean(np.abs(img - img2))

  def calculate_distance(self, train):
    pool = multiprocessing.Pool(8)
    distances = np.zeros((len(train), len(train)))
    for i, img in enumerate(train):
      all_imgs = [(img, f) for f in train]
      dists = pool.map(self.compare, all_imgs)
      distances[i, :] = dists
    return distances

  def url_to_image(self, url):
    ret = None
    try:
      request = urllib.request.Request(url)
      resp = urllib.request.urlopen(request)
      ret = np.asarray(bytearray(resp.read()), dtype="uint8")
      ret = cv2.imdecode(ret, cv2.IMREAD_COLOR)
    except Exception as e:
      logger.debug(e)
    return ret

  def show_images(self, image_urls, col_num=4):
    images = []
    for url in image_urls:
      img = self.url_to_image(url)
      if img is None:
        continue
      img_ = hv.Image(img)
      images.append(img_)
    obj = hv.Layout(images).cols(col_num).display('all')
    return obj

  def url_to_trim_image(self, url, size, f=1000):
    ret = None
    try:
      request = urllib.request.Request(url)
      response = urllib.request.urlopen(request)
      # print("response:" + str(response))
      with Image(file=response).clone() as img:
        img.trim(fuzz=f)
        img.resize(size, size)
        ret = img.make_blob()
        ret = np.asarray(bytearray(ret), dtype="uint8")
        ret = cv2.imdecode(ret, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
      logger.debug(e)

    return ret

  def get_train(self, image_urls, image_size=240):
    executer = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    futures = [executer.submit(self.url_to_trim_image, x, size=image_size) for x in image_urls]
    vals = [future.result() for index, future in enumerate(futures) if future.result() is not None]
    train = np.array(vals)
    return train

  def clustering(self, image_urls, min_samples=2, eps=0.4, pick_up_num=3):
    train = self.get_train(image_urls)
    print(train)
    if len(train) < min_samples:
      return None
    distances = self.calculate_distance(train)
    if distances is None:
      return None
    cls = cluster.DBSCAN(metric='precomputed', min_samples=min_samples, eps=eps)
    y = cls.fit_predict(distances)
    val = pd.Series(y).value_counts()
    target_clusters_index = [x for x in list(val.index) if x != -1][:pick_up_num]
    order = {key: i for i, key in enumerate(target_clusters_index)}
    picked_up = dict([(index, val) for (index, val) in enumerate(y.tolist()) if val in target_clusters_index])
    picked_up_ = [(order[x2], image_urls[x1]) for (x1, x2) in sorted(picked_up.items(), key=lambda x: order[x[1]])]
    ret = []
    for key, subiter in itertools.groupby(picked_up_, operator.itemgetter(0)):
      vals = [item[1] for item in subiter]
      ret.append({"row_id": int(key), "sumples_num": len(vals), "vals": vals})
    return ret
