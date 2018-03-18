import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt  
import numpy as np

def plot_kpt(image, kpt):
    '''
    Args:
        image: value range:1~255. 
        kpt: n x 68. n>=2.
    '''
    # fig = plt.figure(figsize=(image.shape[0], image.shape[1]))
    plt.imshow(image)
    kpt = np.round(kpt).astype(np.int32)
    plt.plot(kpt[0,:], kpt[1,:], 'b.', linewidth = '0.1')
    plt.savefig('test.jpg', bbox_inches='tight')
    # plt.close(fig)
    return image
