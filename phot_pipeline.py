import photutils
from ccdproc import CCDData, Combiner
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from matplotlib import patches
import glob
from photutils import CircularAperture, CircularAnnulus
from photutils import centroid_2dg
import numpy as np
from astropy.time import Time
import pdb
import es_gen
from copy import deepcopy
import yaml

class phot:
    def __init__(self,paramFile='parameters/phot_parameters.yaml'):
        """ Photometry class
    
        Parameters
        ------
        paramFile: str
            Location of the YAML file that contains the photometry parameters
        Properties
        -------
        paramFile: str
            Same as paramFile above
        param: dict
            The photometry parameters like file names, aperture sizes, guess locations
        fileL: list
            The files on which photometry will be performed
        nImg: int
            Number of images in the sequence
        """
        self.paramFile = paramFile
        self.param = yaml.load(open(paramFile))
        self.fileL = glob.glob(self.param['procFiles'])
        self.nImg = len(self.fileL)
        xCoors, yCoors = [], []
        positions = self.param['refStarPos']
        self.nrc = len(positions)
        self.srcApertures = CircularAperture(positions,r=4.)
        self.xCoors = self.srcApertures.positions[:,0]
        self.yCoors = self.srcApertures.positions[:,1]
        self.bkgApertures = CircularAnnulus(positions,r_in=5.,r_out=8.)
        self.srcNames = np.array(np.arange(self.nrc),dtype=np.str)
        self.srcNames[0] = 'src'

    def getImg(self,path,ext=0):
        """ Load an image from a given path and extensions"""
        HDUList = fits.open(self.fileL[self.nImg /2])
        img = HDUList[ext].data
        head = HDUList[ext].header
        HDUList.close()
        return img, head

    def showStarChoices(self):
        """ Show the star choices for photometry """
        plt.close('all')
        fig, ax = plt.subplots()
        
        img, head = self.getImg(self.fileL[self.nImg/2])
        #t = Time(head['DATE-OBS']+'T'+head['TIME-OBS'])
        
        imData = ax.imshow(img,cmap='viridis',vmin=0,vmax=1.0e4,interpolation='nearest')
        ax.invert_yaxis()
        rad, txtOffset = 50, 20
        ax.scatter(self.xCoors, self.yCoors, s=rad, facecolors='none', edgecolors='r')
        for ind, onePos in enumerate(self.srcApertures.positions):
            
            #circ = plt.Circle((onePos[0], onePos[1]), rad, color='r')
            #ax.add_patch(circ)
            if ind == 0:
                name='src'
            else:
                name=str(ind)
            ax.text(onePos[0]+txtOffset,onePos[1]+txtOffset,name,color='white')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        fig.colorbar(imData,label='Counts')
        fig.show()
        fig.savefig('plots/photometry/star_labels/st_labels.pdf')

    def showStamps(self):
        """Shows the fixed apertures on the image with postage stamps surrounding sources """ 
        
        ##  Calculate approximately square numbers of X & Y positions in the grid
        numGridY = int(np.floor(np.sqrt(self.nrc)))
        numGridX = int(np.ceil(float(self.nrc) / float(numGridY)))
        fig, axArr = plt.subplots(numGridY, numGridX)
        
        ## Get the data
        img, head = self.getImg(self.fileL[self.nImg/2])
        
        boxsize = self.param['boxFindSize']
        for ind, onePos in enumerate(self.srcApertures.positions):
            ax = axArr.ravel()[ind]
            
            yStamp = np.array(onePos[1] + np.array([-1,1]) * boxsize,dtype=np.int)
            xStamp = np.array(onePos[0] + np.array([-1,1]) * boxsize,dtype=np.int)
            
            stamp = img[yStamp[0]:yStamp[1],xStamp[0]:xStamp[1]]
            
            imData = ax.imshow(stamp,cmap='viridis',vmin=0,vmax=1.2e4,interpolation='nearest')
            ax.set_title(self.srcNames[ind])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            circ = plt.Circle((onePos[0] - xStamp[0],onePos[1] - yStamp[0]),
                              self.param['apRadius'],edgecolor='red',facecolor='none')
            ax.add_patch(circ)
        
        fig.colorbar(imData,label='Counts')
        
        totStamps = numGridY * numGridX
        
        for ind in np.arange(self.nrc,totStamps):
            ## Make any extra postage stamps blank
            ax = axArr.ravel()[ind]
            
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #self.srcApertures.plot(indices=ind,color='red')
            #ax.set_xlim(onePos[0] - boxsize,onePos[0] + boxsize)
            #ax.set_ylim(onePos[1] - boxsize,onePos[1] + boxsize)
        
            
        fig.show()
        fig.savefig('plots/photometry/postage_stamps/postage_stamps.pdf')
            
        #     ax.set_xlim(xCoors[0] - 10.,xCoors[0] + 10.)
        #     ax.set_ylim(yCoors[0] - 10.,yCoors[0] + 10.)
        #     srcApertures.plot(color='cyan')
        # fig.colorbar(imData)
        # return fig, ax


# In[82]:
#
# img1 = fits.getdata(fileL[20])
# fig, ax = showFixedAps(img1)
#
#
# # In[83]:
#
# img1 = fits.getdata(fileL[0])
# fig, ax = showFixedAps(img1)
#
#
# # ## All apertures
#
# # In[88]:
#
# help(srcApertures.plot)
#
#
# # In[94]:
#
# len(srcApertures)
#
#
# # In[95]:
#
# srcApertures.positions[0]
#
#
# # In[84]:
#
# img1 = fits.getdata(fileL[0])
# fig, ax = showFixedAps(img1,fullFrame=True)
#
#
# # In[12]:
#
# def get_centroid(img,xGuess,yGuess,boxSize=10):
#     shape = img.shape
#     minX = int(np.max([xGuess - boxSize,0.]))
#     maxX = int(np.min([xGuess + boxSize,shape[1]-1]))
#     minY = int(np.max([yGuess - boxSize,0.]))
#     maxY = int(np.min([yGuess + boxSize,shape[1]-1]))
#     subimg = img[minY:maxY,minX:maxX]
#
#     xcenSub,ycenSub = centroid_2dg(subimg)
#     xcen = xcenSub + minX
#     ycen = ycenSub + minY
#
#     return xcen, ycen, subimg
#
#
# # In[13]:
#
# xcen, ycen, subimg = get_centroid(img,xCoors[0],yCoors[0])
#
#
# # In[14]:
#
# subimg.shape
#
#
# # In[15]:
#
# xcenSub,ycenSub = centroid_2dg(subimg)
#
#
# # In[16]:
#
# xcen, ycen, subimg = get_centroid(img,xCoors[0],yCoors[0])
# newAp = CircularAperture([xcen,ycen],r=4)
# (xcen, ycen)
#
#
# # In[17]:
#
# img1 = fits.getdata(fileL[0])
# fig, ax = showFixedAps(img)
# newAp.plot(ax=ax,color='red')
# fig.savefig('centering.pdf',interpolation='nearest')
#
#
# # ## Now re-do photometry with centroiding built-in
#
# # In[18]:
#
# srcApertures.positions[1] = [843.301,681.8426]
#
#
# # In[19]:
#
# srcApertures.r
#
#
# # In[20]:
#
# originalApertures = deepcopy(srcApertures)
#
#
# # In[21]:
#
# onePos = originalApertures.positions[0]
# onePos[0]
# img.shape
#
#
# # In[22]:
#
# photArr = np.zeros((nImg,nSrc))
# xPos, yPos = [], []
# jdArr = []
# for ind,oneImg in enumerate(fileL):
#     if np.mod(ind,15) == 0:
#         print("On "+str(ind)+' of '+str(len(fileL)))
#     HDUList = fits.open(oneImg)
#     img = HDUList[0].data
#     head = HDUList[0].header
#     t = Time(head['DATE-OBS']+'T'+head['TIME-OBS'])
#     jdArr.append(t.jd)
#
#     for cenInd,onePos in enumerate(originalApertures.positions):
#         xcen, ycen, subimg = get_centroid(img,onePos[0],onePos[1],boxSize=18)
#
#         srcApertures.positions[cenInd] = [xcen,ycen]
#         bkgApertures.positions[cenInd] = [xcen,ycen]
#
#     rawPhot = aperture_photometry(img,srcApertures)
#     bkgPhot = aperture_photometry(img,bkgApertures)
#     bkgVals = bkgPhot['aperture_sum'] / bkgApertures.area() * srcApertures.area()
#
#     ## Background subtracted fluxes
#     srcPhot = rawPhot['aperture_sum'] - bkgVals
#     photArr[ind,:] = srcPhot
#
#
#
#
# # In[23]:
#
# fig, ax = plt.subplots()
# for oneSrc in range(nSrc):
#     yFlux = photArr[:,oneSrc]
#     yNorm = yFlux / np.median(yFlux)
#     if oneSrc == 0:
#         pLabel = 'Src'
#     else:
#         pLabel = 'Ref '+str(oneSrc)
#     yplot = yNorm - 0.03 * oneSrc
#     yplot = yNorm
#     ax.plot(jdArr - jdRef,yNorm,label=pLabel)
#
# ax.set_xlabel('JD - '+str(jdRef))
# ax.set_ylabel('Normalized Flux + Offset')
# ax.legend(loc='best',fontsize=10)
# fig.savefig('second_t_series.pdf')
#
#
# # In[37]:
#
# def refSeries(photArr,reNorm=False,custSrc=None):
#     combRef = []
#     if custSrc == None:
#         refArray = np.arange(1,nSrc)
#     else:
#         refArray = custSrc
#
#     for oneSrc in refArray:
#         if reNorm == True:
#             refSeries = photArr[:,oneSrc]
#             refSeries = refSeries / np.median(refSeries)
#         else:
#             refSeries = photArr[:,oneSrc]
#
#         if oneSrc == 1:
#             combRef = refSeries
#         else:
#             combRef = combRef + refSeries
#     yCorrected = photArr[:,0] / combRef
#     yCorrNorm = yCorrected / np.median(yCorrected)
#     return yCorrNorm
#
#
# # In[38]:
#
# yCorrNorm = refSeries(photArr)
#
#
# # In[39]:
#
# fig, ax = plt.subplots()
# ax.plot(jdArr - jdRef,yCorrNorm)
#
#
# # In[41]:
#
# goodRef = [1,3,4,5,6]
# yCorrNorm2 = refSeries(photArr,custSrc=goodRef)
#
#
# # In[49]:
#
# yCorrNorm3 = refSeries(photArr,custSrc=goodRef,reNorm=True)
# yCorrNorm4 = refSeries(photArr,custSrc=None,reNorm=True)
#
#
# # In[52]:
#
# fig, ax = plt.subplots()
# ax.plot(jdArr - jdRef,yCorrNorm,label='All Ref Sum')
# ax.plot(jdArr - jdRef,yCorrNorm2,label='Choice Ref')
# ax.plot(jdArr - jdRef,yCorrNorm3,label='Choice Ref Norm Sum')
# ax.plot(jdArr - jdRef,yCorrNorm4,label='All Ref Norm Sum')
# ax.legend(loc='lower right')
# ax.set_xlabel('JD - '+str(jdRef))
# ax.set_ylabel('Normalized Flux')
# fig.savefig('reference_method.pdf')
#
#
# # ## Compare to SLC curve
#
# # In[63]:
#
# lc = ascii.read('slc_curve/k1255.hjd',data_start=0,names=['HJD'])
#
# quickfl = ascii.read('slc_curve/k1255.dR7.r12345689101112',data_start=0,names=['Fl'])
#
# lc.add_column(quickfl['Fl'])
#
# x = lc['HJD'] - 1.
# y = 10**(0.4 * lc['Fl'])
#
#
# # In[ ]:
#
# yfit = es_gen.yCorrNorm3
#
#
# # In[72]:
#
# fig, ax = plt.subplots(figsize=(12,12))
# #ax.plot(jdArr - jdRef,yCorrNorm3,'o',label='Choice Ref Norm Sum')
# ax.plot(jdArr - jdRef,yCorrNorm3,label='Choice Ref Norm Sum')
# #ax.plot(x,y,'o',label='slc pipelin')
# ax.plot(x,y,label='slc pipelin')
# ax.legend(loc='lower right')
# ax.set_xlabel('JD - '+str(jdRef))
# ax.set_ylabel('Normalized Flux')
# fig.savefig('reference_method.pdf')


# In[ ]:



