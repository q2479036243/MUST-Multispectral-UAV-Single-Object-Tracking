import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np

davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):
    """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('Using opencv_loader instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)

default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)

        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv.imread(path, cv.IMREAD_COLOR)

            # convert to rgb and return
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None


def opencv_seg_loader(path):
    """ Read segmentation annotation using opencv's imread function"""
    try:
        return cv.imread(path)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def imread_indexed(filename):
    """ Load indexed image with given filename. Used to read segmentation annotations."""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    """ Save indexed image as png. Used to save segmentation annotation."""

    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')
    
def hsijpg_loader(path):
    """ Read hsijpg"""
    try:
        # im1 = cv.imread(path, cv.IMREAD_COLOR)
        # im2 = cv.imread(path.replace('img1','img2'), cv.IMREAD_COLOR)
        # im3 = cv.imread(path.replace('img1','img3'), cv.IMREAD_COLOR)
        # im = np.concatenate((im1, im2, im3[:,:,:2]), axis=2)
        
        im1 = jpeg4py.JPEG(path).decode()
        im2 = jpeg4py.JPEG(path.replace('img1','img2')).decode()
        im3 = jpeg4py.JPEG(path.replace('img1','img3')).decode()
        # # im = np.concatenate((im3[:,:,1:3], im2, im1), axis=2)[:, :, ::-1]
        # # im = np.concatenate((im1[:, :, ::-1], im2[:, :, ::-1], im3[:,:,1:3][:, :, ::-1]), axis=2)   
        im = np.concatenate((im1[:,:,2:3], im1[:,:,1:2], im1[:,:,0:1], im2[:,:,2:3], im2[:,:,1:2], im2[:,:,0:1], im3[:,:,1:2], im3[:,:,0:1]), axis=2)
        # # im = np.concatenate((im3[:,:,0:1], im3[:,:,1:2], im2[:,:,0:1], im2[:,:,1:2], im2[:,:,2:3], im1[:,:,0:1], im1[:,:,1:2], im1[:,:,2:3]), axis=2)
        
        # im = np.load(path.replace('HSIJPG','HSIData').replace('.img,jpg','.npy'))

        return im
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def hotjpg_loader(path):
    try:
        im1 = jpeg4py.JPEG(path).decode()
        im2 = jpeg4py.JPEG(path.replace('img1','img2')).decode()
        im3 = jpeg4py.JPEG(path.replace('img1','img3')).decode()
        im4 = jpeg4py.JPEG(path.replace('img1','img4')).decode()
        im5 = jpeg4py.JPEG(path.replace('img1','img5')).decode()
        im6 = jpeg4py.JPEG(path.replace('img1','img6')).decode()
        im = np.concatenate((im1[:,:,2:3], im1[:,:,1:2], im1[:,:,0:1], im2[:,:,2:3], im2[:,:,1:2], im2[:,:,0:1], im3[:,:,2:3], im3[:,:,1:2], im3[:,:,0:1], im4[:,:,2:3], im4[:,:,1:2], im4[:,:,0:1], im5[:,:,2:3], im5[:,:,1:2], im5[:,:,0:1], im6[:,:,0:1]), axis=2)
        return im
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None
    
    
def hotpng_loader(path):
    try:
        data = Image.open(path)
        img = np.array(data)
        
        # Parameters
        M, N = img.shape
        B = [4, 4]
        skip = [4, 4]
        bandNumber = 16
        col_extent = N - B[1] + 1
        row_extent = M - B[0] + 1
        # Get Starting block indices
        start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
        # Generate Depth indeces
        didx = M * N * np.arange(1)
        start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
        # Get all actual indices & index into input array for final output
        out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
        out = np.transpose(out)
        DataCube = out.reshape(M//4, N//4, bandNumber)  # (272, 512, 16)    <class 'numpy.int32'>
        
        # cube = ((DataCube - DataCube.min()) / (DataCube.max() - DataCube.min()) * 255).astype(np.uint8)[:,:,:3]
        return DataCube
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None